# train_autoencoder.py

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from model_autoencoder import JVAutoencoder


class JVAutoencoderDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.jv_input = data["jv_input"].astype(np.float32)
        self.jv_target = data["jv_target"].astype(np.float32)
        self.weather = data["weather"].astype(np.float32)

    def __len__(self):
        return len(self.jv_input)

    def __getitem__(self, idx):
        return (
            self.jv_input[idx],
            self.jv_target[idx],
            self.weather[idx],
        )


def train(
    dataset_path, model_path,
    out_dir="models/autoencoder",
    batch_size=64,
    lr=1e-3,
    epochs=100,
    val_split=0.2,
    latent_dim=32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds = JVAutoencoderDataset(dataset_path)
    n_total = len(full_ds)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    jv_dim = full_ds.jv_input.shape[1]
    weather_dim = full_ds.weather.shape[1]

    model = JVAutoencoder(jv_dim, weather_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for jv_in, jv_tgt, weather in train_loader:
            jv_in, jv_tgt, weather = jv_in.to(device), jv_tgt.to(device), weather.to(device)

            opt.zero_grad()
            jv_recon, _ = model(jv_in, weather)
            loss = loss_fn(jv_recon, jv_tgt)
            loss.backward()
            opt.step()

            train_loss += loss.item() * jv_in.size(0)

        train_loss /= n_train

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for jv_in, jv_tgt, weather in val_loader:
                jv_in, jv_tgt, weather = jv_in.to(device), jv_tgt.to(device), weather.to(device)
                jv_recon, _ = model(jv_in, weather)
                loss = loss_fn(jv_recon, jv_tgt)
                val_loss += loss.item() * jv_in.size(0)

        val_loss /= n_val
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "jv_dim": jv_dim,
                    "weather_dim": weather_dim,
                    "latent_dim": latent_dim,
                },
                model_path,
            )
            print("  -> Saved best model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/jv_autoencoder_dataset.npz")
    parser.add_argument("--out_dir", type=str, default="models/autoencoder")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=32)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
    )
