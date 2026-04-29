import torch
import torch.nn as nn
from model_lstm import VocJscLSTM
from preprocess_jv import get_dataloaders
import matplotlib.pyplot as plt#
import os

def train_model(model, train_loader, val_loader, epochs=40, lr=1e-3, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_running += loss.item()

        val_loss = val_running / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # FIXED: unpack all 3 returned values
    train_loader, val_loader, scaler_dict = get_dataloaders()

    model = VocJscLSTM()

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=40, lr=1e-3, device=device
    )

    torch.save({
        "model_state": model.state_dict(),
        "scalers": scaler_dict
    }, "outputs/voc_jsc_lstm.pth")
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/training_curves.png")
    plt.close()

    print("Training curves saved to outputs/training_curves.png")