import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from model_autoencoder import PhysicsInformedJVModel


# --------------------------------------------------------
# Global plotting style
# --------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.5,
    "figure.dpi": 150,
})

def smooth_curve(y, window=15):
    """
    Simple moving-average smoothing.
    y: 1D numpy array
    window: number of points to average
    """
    if window < 2:
        return y

    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="same")
    return y_smooth

# --------------------------------------------------------
# Plot single sample
# --------------------------------------------------------

def plot_single_jv(
    V_grid,
    measured,
    target,
    predicted,
    sample_idx,
    rmse,
    save_path,
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # Measured JV
    ax.plot(
        V_grid,
        measured,
        color="#4C78A8",
        alpha=0.6,
        linewidth=2.0,
        label="Measured JV (raw)",
    )

    # Smoothed reference JV
    ax.plot(
        V_grid,
        target,
        color="#F58518",
        linewidth=2.8,
        label="Reference JV (clean)",
    )

    # Predicted JV
    ax.plot(
        V_grid,
        predicted,
        "--",
        color="#54A24B",
        linewidth=3.0,
        label="Physics-informed prediction",
    )

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")

    ax.set_title(
        f"JV Reconstruction #{sample_idx}",
        pad=15,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(
        loc="lower left",
        frameon=True,
    )

    ax.text(
        0.02,
        0.97,
        f"RMSE = {rmse:.4f} A",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------------
# Main plotting routine
# --------------------------------------------------------

def plot_multiple_predictions(
    dataset_path,
    model_path,
    n_samples=10,
    random_samples=True,
    save_dir="outputs/Perovskite_5&6/ae/curves",
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------
    # Load dataset
    # ----------------------------------------------------

    data = np.load(
        dataset_path,
        allow_pickle=True,
    )

    jv_input = data["jv_input"].astype(np.float32)
    jv_target = data["jv_target"].astype(np.float32)
    weather = data["weather"].astype(np.float32)
    V_grid = data["V_grid"].astype(np.float32)

    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model = PhysicsInformedJVModel(
        jv_dim=checkpoint["jv_dim"],
        weather_dim=checkpoint["weather_dim"],
        voltage_grid=V_grid,
        latent_dim=checkpoint["latent_dim"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ----------------------------------------------------
    # Choose samples
    # ----------------------------------------------------

    n_total = len(jv_input)

    if random_samples:

        indices = np.random.choice(
            n_total,
            size=min(n_samples, n_total),
            replace=False,
        )

    else:

        indices = np.arange(
            min(n_samples, n_total)
        )

    # ----------------------------------------------------
    # Generate plots
    # ----------------------------------------------------

    for idx in indices:

        jv_in = torch.tensor(
            jv_input[idx:idx + 1]
        ).to(device)

        weather_in = torch.tensor(
            weather[idx:idx + 1]
        ).to(device)

        with torch.no_grad():

            jv_pred, _, params = model(
                jv_in,
                weather_in,
            )

        jv_pred = (
            jv_pred
            .cpu()
            .numpy()
            .squeeze()
        )

        jv_pred_plot = np.maximum(jv_pred, 0)

        rmse = np.sqrt(
            np.mean(
                (
                    jv_pred
                    - jv_target[idx]
                ) ** 2
            )
        )

        save_path = (
            save_dir
            / f"sample_{idx:05d}.png"
        )

        plot_single_jv(
            V_grid=V_grid,
            measured=jv_input[idx],
            target=jv_target[idx],
            predicted=jv_pred_plot,
            sample_idx=idx,
            rmse=rmse,
            save_path=save_path,
        )

        print(
            f"\nSaved: {save_path}"
        )

        print(
            f"Iph={params['Iph'].item():.4f} | "
            f"Io={params['Io'].item():.3e} | "
            f"n={params['n'].item():.3f} | "
            f"Rs={params['Rs'].item():.3f} | "
            f"Rsh={params['Rsh'].item():.1f}"
        )


# --------------------------------------------------------
# Entry point
# --------------------------------------------------------

if __name__ == "__main__":

    plot_multiple_predictions(
        dataset_path="data/jv_autoencoder_dataset_perovskite.npz",
        model_path="models/jv_autoencoder_perovskite.pth",
        n_samples=10,
        random_samples=True,
        save_dir="outputs/ae/curves",
    )


if __name__ == "__main__":

    plot_multiple_predictions(
        dataset_path="data/jv_autoencoder_dataset_perovskite.npz",
        model_path="models/jv_autoencoder_perovskite.pth",
        n_samples=10,
        random_samples=True
    )