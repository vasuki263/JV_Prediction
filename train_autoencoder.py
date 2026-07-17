import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from model_autoencoder import PhysicsInformedJVModel


class JVPhysicsDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        self.jv_input = data["jv_input"].astype(np.float32)
        self.jv_target = data["jv_target"].astype(np.float32)
        self.weather = data["weather"].astype(np.float32)
        self.V_grid = data["V_grid"].astype(np.float32)
        self.diode_params = data["diode_params"].astype(np.float32)
        # self.diode_params[:, 1] = np.log(self.diode_params[:, 1])

        # compute normalization stats for diode params
        self.diode_mean = self.diode_params.mean(axis=0)
        self.diode_std = self.diode_params.std(axis=0) + 1e-8

        self.diode_params_norm = (self.diode_params - self.diode_mean) / self.diode_std

    def __len__(self):
        return len(self.jv_input)

    def __getitem__(self, idx):
        return (
            self.jv_input[idx],
            self.jv_target[idx],
            self.weather[idx],
            self.diode_params_norm[idx],
        )


def train(
    dataset_path,
    model_path,
    out_dir="models",
    batch_size=64,
    lr=1e-3,
    epochs=100,
    val_split=0.2,
    latent_dim=32,
    lambda_params=1e-3,
    lambda_residual=1e-3,
    lambda_latent=1e-5,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds = JVPhysicsDataset(dataset_path)
    V_grid = full_ds.V_grid

    n_total = len(full_ds)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    jv_dim = full_ds.jv_input.shape[1]
    weather_dim = full_ds.weather.shape[1]

    model = PhysicsInformedJVModel(
        jv_dim=jv_dim,
        weather_dim=weather_dim,
        voltage_grid=V_grid,
        latent_dim=latent_dim,
        diode_mean=full_ds.diode_mean,
        diode_std=full_ds.diode_std,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn_jv = nn.MSELoss()
    loss_fn_params = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss = 0.0

        for jv_in, jv_tgt, weather, diode_norm in train_loader:

            jv_in = jv_in.to(device)
            jv_tgt = jv_tgt.to(device)
            weather = weather.to(device)
            diode_norm = diode_norm.to(device)

            opt.zero_grad()

            I_pred, I_phys, delta_I, z, diode_pred_norm = model(
                jv_in,
                weather,
                diode_params_norm=diode_norm,
            )

            loss_recon = loss_fn_jv(I_pred, jv_tgt)
            loss_params = loss_fn_params(diode_pred_norm, diode_norm)
            loss_residual = (delta_I ** 2).mean()
            loss_latent = (z ** 2).mean()
            loss_phys = loss_fn_jv(I_phys, jv_tgt)

            loss = (
                loss_recon
                + lambda_params * loss_params
                + lambda_residual * loss_residual
                + lambda_latent * loss_latent
                + 0.2 * loss_phys
            )

            loss.backward()
            opt.step()

            train_loss += loss.item() * jv_in.size(0)

        train_loss /= n_train

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for jv_in, jv_tgt, weather, diode_norm in val_loader:

                jv_in = jv_in.to(device)
                jv_tgt = jv_tgt.to(device)
                weather = weather.to(device)
                diode_norm = diode_norm.to(device)

                I_pred, I_phys, delta_I, z, diode_pred_norm = model(
                    jv_in,
                    weather,
                    diode_params_norm=diode_norm,
                )

                loss_recon = loss_fn_jv(I_pred, jv_tgt)
                loss_params = loss_fn_params(diode_pred_norm, diode_norm)
                loss_residual = (delta_I ** 2).mean()
                loss_latent = (z ** 2).mean()
                loss_phys = loss_fn_jv(I_phys, jv_tgt)

                loss = (
                    loss_recon
                    + lambda_params * loss_params
                    + lambda_residual * loss_residual
                    + lambda_latent * loss_latent
                    + 0.02 * loss_phys
                )

                val_loss += loss.item() * jv_in.size(0)

        val_loss /= n_val

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "jv_dim": jv_dim,
                    "weather_dim": weather_dim,
                    "latent_dim": latent_dim,
                    "V_grid": V_grid,
                    "diode_mean": full_ds.diode_mean,
                    "diode_std": full_ds.diode_std,
                },
                model_path,
            )

            print("  -> Saved best model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/jv_autoencoder_dataset_perovskite.npz")
    parser.add_argument("--model_path", type=str, default="models/jv_autoencoder_perovskite.pth")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
    )