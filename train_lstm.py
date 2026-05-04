# train_lstm.py

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model_lstm import JV_LSTM
from build_dataset import build_dataset
import matplotlib.pyplot as plt


def train(directory):
    train_loader, val_loader, test_loader, scalers, raw_test = build_dataset(directory)

    model = JV_LSTM(input_size=10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, verbose=True
    )

    writer = SummaryWriter(log_dir="runs/jv_lstm")

    best_val_loss = float("inf")
    patience = 12
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(200):
        model.train()
        running = 0
        for X, Y in train_loader:
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
            running += loss.item()

        train_loss = running / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running = 0
        with torch.no_grad():
            for X, Y in val_loader:
                pred = model(X)
                val_running += loss_fn(pred, Y).item()

        val_loss = val_running / len(val_loader)
        val_losses.append(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", opt.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch+1} | Train {train_loss:.5f} | Val {val_loss:.5f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model": model.state_dict(), "scalers": scalers}, "jv_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    writer.close()

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.grid()
    plt.savefig("training_curve.png")
    plt.close()


if __name__ == "__main__":
    train("C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw")
