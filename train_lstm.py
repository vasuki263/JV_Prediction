# train_lstm.py — UPDATED FOR INFLUXDB-BASED JV EXTRACTION

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

from model_lstm import JV_LSTM
from build_dataset import build_dataset


# ---------------------------------------------------------
# DIRECTORY HELPERS
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# TRAIN FUNCTION (UPDATED)
# ---------------------------------------------------------
def train(dataset,
          tech,
          model_path="jv_model.pth",
          run_dir=None,
          epochs=40,
          lr=1e-3):

    # -----------------------------------------------------
    # Output directories
    # -----------------------------------------------------
    if run_dir is None:
        base_plot_dir = "outputs/training/plots"
        base_metric_dir = "outputs/training/metrics"
    else:
        base_plot_dir = os.path.join(run_dir, "training_plots")
        base_metric_dir = os.path.join(run_dir, "training_metrics")

    reset_dir(base_plot_dir)
    reset_dir(base_metric_dir)

    # -----------------------------------------------------
    # Dataset is already built in pipeline.py
    # -----------------------------------------------------
    train_loader, val_loader, test_loader, scalers, raw_test = dataset

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = JV_LSTM(input_size=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for X, Y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X, Y in val_loader:
                pred = model(X)
                val_batch_losses.append(criterion(pred, Y).item())

        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------
    ensure_dir(os.path.dirname(model_path))
    torch.save({"model": model.state_dict(), "scalers": scalers}, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # -----------------------------------------------------
    # Save loss curves
    # -----------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss ({tech})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/loss_curve.png", dpi=150)
    plt.close()

    # Save metrics CSV
    df = pd.DataFrame({
        "epoch": np.arange(1, epochs+1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    df.to_csv(f"{base_metric_dir}/training_losses.csv", index=False)

    print("[INFO] Training complete.")
    return model
