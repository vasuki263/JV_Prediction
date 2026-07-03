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

    def __len__(self):
        return len(self.jv_input)

    def __getitem__(self, idx):
        return (
            self.jv_input[idx],
            self.jv_target[idx],
            self.weather[idx],
            self.diode_params[idx],
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
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds = JVPhysicsDataset(dataset_path)
    V_grid = full_ds.V_grid

    n_total = len(full_ds)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size
    )

    jv_dim = full_ds.jv_input.shape[1]
    weather_dim = full_ds.weather.shape[1]

    model = PhysicsInformedJVModel(
        jv_dim=jv_dim,
        weather_dim=weather_dim,
        voltage_grid= V_grid,
        latent_dim=latent_dim,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        # ----------------------------------
        # Training
        # ----------------------------------

        model.train()

        train_loss = 0

        for jv_in, jv_tgt, weather, diode_params in train_loader:

            jv_in = jv_in.to(device)
            jv_tgt = jv_tgt.to(device)
            weather = weather.to(device)
            diode_params = diode_params.to(device)

            opt.zero_grad()

            jv_pred, I_phys, delta_I, z = model(
                jv_in,
                weather,
                diode_params
            )
            if epoch % 10 == 0:
                print(
                    f"Residual RMS: "
                    f"{torch.sqrt((delta_I ** 2).mean()).item():.5f}"
                )

            loss_recon = loss_fn(
                jv_pred,
                jv_tgt
            )

            loss_residual = 5e-5 * (
                    delta_I ** 2
            ).mean()

            d2 = (
                    delta_I[:, 2:]
                    - 2 * delta_I[:, 1:-1]
                    + delta_I[:, :-2]
            )

            loss_smooth = 1e-4 * (d2 ** 2).mean()

            loss_latent = 1e-5 * (
                    z ** 2
            ).mean()

            loss = (
                    loss_recon
                    + loss_residual
                    + loss_latent + loss_smooth
            )

            loss.backward()
            opt.step()

            train_loss += (
                loss.item()
                * jv_in.size(0)
            )

        train_loss /= n_train

        # ----------------------------------
        # Validation
        # ----------------------------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for jv_in, jv_tgt, weather, diode_params in val_loader:

                jv_in = jv_in.to(device)
                jv_tgt = jv_tgt.to(device)
                weather = weather.to(device)
                diode_params = diode_params.to(device)

                jv_pred, I_phys, delta_I, z = model(
                    jv_in,
                    weather,
                    diode_params
                )

                loss = loss_fn(
                    jv_pred,
                    jv_tgt
                )

                val_loss += (
                    loss.item()
                    * jv_in.size(0)
                )

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
                },
                model_path,
            )

            print("  -> Saved best model")

