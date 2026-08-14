import argparse
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

from pathlib import Path

from model_autoencoder import (
    PhysicsInformedJVModel
)


# ============================================================
# Dataset
# ============================================================

class JVPhysicsDataset(Dataset):

    def __init__(
        self,
        npz_path
    ):

        data = np.load(
            npz_path,
            allow_pickle=True
        )

        # ----------------------------------------------------
        # Measured JV target
        # ----------------------------------------------------

        self.jv_target = (
            data["jv_target"]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        self.weather = (
            data["weather"]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Voltage grid
        # ----------------------------------------------------

        self.V_grid = (
            data["V_grid"]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Fitted diode parameters
        #
        # These came from fitting the measured JV curves.
        # They are NOT model inputs.
        # ----------------------------------------------------

        if "diode_params" in data:

            self.diode_params = (
                data["diode_params"]
                .astype(np.float32)
            )

        else:

            raise KeyError(
                "Dataset must contain 'diode_params'."
            )

        # ----------------------------------------------------
        # Normalize diode parameters
        # ----------------------------------------------------

        self.diode_mean = (
            self.diode_params.mean(
                axis=0
            )
        )

        self.diode_std = (
            self.diode_params.std(
                axis=0
            )
            + 1e-8
        )

        self.diode_params_norm = (
            self.diode_params
            - self.diode_mean
        ) / self.diode_std

    def __len__(self):

        return len(
            self.jv_target
        )

    def __getitem__(
        self,
        idx
    ):

        return (
            self.jv_target[idx],
            self.weather[idx],
            self.diode_params_norm[idx],
        )


# ============================================================
# Training
# ============================================================

def train(

    dataset_path,

    model_path,

    out_dir="models",

    batch_size=64,

    lr=1e-4,

    epochs=100,

    val_split=0.2,

    latent_dim=64,

    # --------------------------------------------------------
    # Loss weights
    # --------------------------------------------------------

    lambda_phys=0.02,

    lambda_diode=0.020,

    lambda_residual=1e-3,

    lambda_latent=1e-5,

):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Output directory
    # ========================================================

    out_dir = Path(
        out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Dataset
    # ========================================================

    full_ds = JVPhysicsDataset(
        dataset_path
    )

    V_grid = (
        full_ds.V_grid
    )

    n_total = len(
        full_ds
    )

    n_val = max(
        1,
        int(
            val_split
            * n_total
        )
    )

    n_train = (
        n_total
        - n_val
    )

    train_ds, val_ds = random_split(
        full_ds,
        [
            n_train,
            n_val
        ]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ========================================================
    # Dimensions
    # ========================================================

    # jv_dim is OUTPUT dimension only.
    jv_dim = (
        full_ds.jv_target.shape[1]
    )

    weather_dim = (
        full_ds.weather.shape[1]
    )

    # ========================================================
    # Model
    # ========================================================

    model = PhysicsInformedJVModel(

        jv_dim=jv_dim,

        weather_dim=weather_dim,

        voltage_grid=V_grid,

        latent_dim=latent_dim,

        diode_mean=full_ds.diode_mean,

        diode_std=full_ds.diode_std,

    ).to(device)

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # ========================================================
    # Loss
    # ========================================================

    mse = nn.MSELoss()

    best_val = float(
        "inf"
    )

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        train_total = 0.0
        train_jv = 0.0
        train_phys = 0.0
        train_diode = 0.0
        train_residual = 0.0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for (
            jv_true,
            weather,
            diode_true_norm,
        ) in train_loader:

            jv_true = (
                jv_true.to(device)
            )

            weather = (
                weather.to(device)
            )

            diode_true_norm = (
                diode_true_norm.to(device)
            )

            optimizer.zero_grad()

            # =================================================
            # WEATHER ONLY
            # =================================================

            (
                jv_pred,
                jv_phys,
                delta_I,
                z,
                diode_pred_norm,
            ) = model(
                weather
            )

            # =================================================
            # 1. FINAL JV LOSS
            #
            # Main objective:
            # predicted JV vs measured JV
            # =================================================

            loss_jv = mse(
                jv_pred,
                jv_true
            )

            # =================================================
            # 2. PHYSICS JV LOSS
            #
            # Forces the single-diode component itself to
            # approximate the measured JV.
            # =================================================

            loss_phys = mse(
                jv_phys,
                jv_true
            )

            # =================================================
            # 3. DIODE RESIDUAL LOSS
            #
            # Fitted diode parameters are supervision only.
            #
            # Residual:
            #
            # predicted fitted parameter
            #        -
            # fitted parameter from measured JV
            #
            # =================================================

            diode_residual = (
                diode_pred_norm
                - diode_true_norm
            )

            loss_diode_residual = (
                diode_residual ** 2
            ).mean()

            # =================================================
            # 4. JV RESIDUAL REGULARIZATION
            #
            # Prevents the neural residual from becoming
            # unnecessarily large.
            # =================================================

            loss_residual = (
                delta_I ** 2
            ).mean()

            # =================================================
            # 5. Latent regularization
            # =================================================

            loss_latent = (
                z ** 2
            ).mean()

            # =================================================
            # TOTAL LOSS
            # =================================================

            loss = (

                loss_jv

                + lambda_phys
                * loss_phys

                + lambda_diode
                * loss_diode_residual

                + lambda_residual
                * loss_residual

                + lambda_latent
                * loss_latent
            )

            # =================================================
            # Backprop
            # =================================================

            loss.backward()

            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            # =================================================
            # Accumulate
            # =================================================

            bs = (
                jv_true.size(0)
            )

            train_total += (
                loss.item()
                * bs
            )

            train_jv += (
                loss_jv.item()
                * bs
            )

            train_phys += (
                loss_phys.item()
                * bs
            )

            train_diode += (
                loss_diode_residual.item()
                * bs
            )

            train_residual += (
                loss_residual.item()
                * bs
            )

        # ====================================================
        # Average training losses
        # ====================================================

        train_total /= n_train
        train_jv /= n_train
        train_phys /= n_train
        train_diode /= n_train
        train_residual /= n_train

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_total = 0.0
        val_jv = 0.0
        val_phys = 0.0
        val_diode = 0.0
        val_residual = 0.0

        with torch.no_grad():

            for (
                jv_true,
                weather,
                diode_true_norm,
            ) in val_loader:

                jv_true = (
                    jv_true.to(device)
                )

                weather = (
                    weather.to(device)
                )

                diode_true_norm = (
                    diode_true_norm.to(device)
                )

                (
                    jv_pred,
                    jv_phys,
                    delta_I,
                    z,
                    diode_pred_norm,
                ) = model(
                    weather
                )

                # ------------------------------------------------
                # Final JV
                # ------------------------------------------------

                loss_jv = mse(
                    jv_pred,
                    jv_true
                )

                # ------------------------------------------------
                # Physics JV
                # ------------------------------------------------

                loss_phys = mse(
                    jv_phys,
                    jv_true
                )

                # ------------------------------------------------
                # Diode residual
                # ------------------------------------------------

                diode_residual = (
                    diode_pred_norm
                    - diode_true_norm
                )

                loss_diode_residual = (
                    diode_residual ** 2
                ).mean()

                # ------------------------------------------------
                # Residual regularization
                # ------------------------------------------------

                loss_residual = (
                    delta_I ** 2
                ).mean()

                # ------------------------------------------------
                # Latent
                # ------------------------------------------------

                loss_latent = (
                    z ** 2
                ).mean()

                # ------------------------------------------------
                # Total
                # ------------------------------------------------

                loss = (

                    loss_jv

                    + lambda_phys
                    * loss_phys

                    + lambda_diode
                    * loss_diode_residual

                    + lambda_residual
                    * loss_residual

                    + lambda_latent
                    * loss_latent
                )

                bs = (
                    jv_true.size(0)
                )

                val_total += (
                    loss.item()
                    * bs
                )

                val_jv += (
                    loss_jv.item()
                    * bs
                )

                val_phys += (
                    loss_phys.item()
                    * bs
                )

                val_diode += (
                    loss_diode_residual.item()
                    * bs
                )

                val_residual += (
                    loss_residual.item()
                    * bs
                )

        # ====================================================
        # Average
        # ====================================================

        val_total /= n_val
        val_jv /= n_val
        val_phys /= n_val
        val_diode /= n_val
        val_residual /= n_val

        # ====================================================
        # Print
        # ====================================================

        print(
            f"\nEpoch {epoch:03d}"
        )

        print(
            f"  Train total    : {train_total:.6f}"
        )

        print(
            f"  Train JV       : {train_jv:.6f}"
        )

        print(
            f"  Train physics  : {train_phys:.6f}"
        )

        print(
            f"  Train diode    : {train_diode:.6f}"
        )

        print(
            f"  Train residual : {train_residual:.6f}"
        )

        print(
            f"  Val total      : {val_total:.6f}"
        )

        print(
            f"  Val JV         : {val_jv:.6f}"
        )

        print(
            f"  Val physics    : {val_phys:.6f}"
        )

        print(
            f"  Val diode      : {val_diode:.6f}"
        )

        print(
            f"  Val residual   : {val_residual:.6f}"
        )

        # ====================================================
        # Save best model
        # ====================================================

        if val_total < best_val:

            best_val = val_total

            torch.save(
                {

                    "model_state_dict":
                        model.state_dict(),

                    "jv_dim":
                        jv_dim,

                    "weather_dim":
                        weather_dim,

                    "latent_dim":
                        latent_dim,

                    "V_grid":
                        V_grid,

                    "diode_mean":
                        full_ds.diode_mean,

                    "diode_std":
                        full_ds.diode_std,

                    # Save loss weights too
                    "lambda_phys":
                        lambda_phys,

                    "lambda_diode":
                        lambda_diode,

                    "lambda_residual":
                        lambda_residual,

                    "lambda_latent":
                        lambda_latent,
                },
                model_path,
            )

            print(
                "  -> Saved best model"
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=(
            "data/"
            "jv_autoencoder_dataset_perovskite.npz"
        )
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "models/"
            "jv_autoencoder_perovskite.pth"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()

    train(

        dataset_path=args.dataset_path,

        model_path=args.model_path,

        batch_size=args.batch_size,

        lr=args.lr,

        epochs=args.epochs,
    )