import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from model_autoencoder import PhysicsInformedJVModel
from scipy.signal import savgol_filter


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


# ---------------------------------------------------------
# Smooth JV curve
# ---------------------------------------------------------
def smooth_jv_curve(V, J, num_points=200):

    V = np.asarray(V, float)
    J = np.asarray(J, float)

    mask = np.isfinite(V) & np.isfinite(J)
    V, J = V[mask], J[mask]

    if len(V) < 5:
        return V, J

    idx = np.argsort(V)
    V, J = V[idx], J[idx]

    V_new = np.linspace(V.min(), V.max(), num_points)
    f = interp1d(V, J, kind="linear", fill_value="extrapolate")
    J_new = f(V_new)

    window = 31
    if window >= len(J_new):
        window = len(J_new) - 1 if (len(J_new) - 1) % 2 == 1 else len(J_new) - 2

    if window < 7:
        return V_new, J_new

    J_smooth = savgol_filter(J_new, window_length=window, polyorder=3)
    return V_new, J_smooth


# ---------------------------------------------------------
# Plot a single JV curve
# ---------------------------------------------------------
def plot_single_jv(V_grid, measured, target, predicted, sample_idx, rmse, save_path):

    V_m, J_m = smooth_jv_curve(V_grid, measured)
    V_t, J_t = smooth_jv_curve(V_grid, target)
    V_p, J_p = smooth_jv_curve(V_grid, predicted)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(V_m, J_m, alpha=0.7, label="Measured JV")
    ax.plot(V_t, J_t, label="Target JV")
    ax.plot(V_p, J_p, "--", label="Predicted JV")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title(f"JV Reconstruction #{sample_idx}", pad=15)

    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()

    # ax.text(
    #     0.02,
    #     0.97,
    #     f"RMSE = {rmse:.6f}",
    #     transform=ax.transAxes,
    #     verticalalignment="top",
    #     bbox=dict(facecolor="white", alpha=0.9),
    # )

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------
def plot_multiple_predictions(
    dataset_path,
    model_path,
    n_samples=10,
    random_samples=True,
    save_dir="outputs/ae/curves",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    data = np.load(dataset_path, allow_pickle=True)

    jv_input = data["jv_input"].astype(np.float32)
    jv_target = data["jv_target"].astype(np.float32)
    weather = data["weather"].astype(np.float32)
    V_grid = data["V_grid"].astype(np.float32)
    diode_params_true = data["diode_params"].astype(np.float32)

    # -----------------------------------------
    # Load model + normalization stats
    # -----------------------------------------
    ckpt = torch.load(model_path, map_location=device)

    model = PhysicsInformedJVModel(
        jv_dim=ckpt["jv_dim"],
        weather_dim=ckpt["weather_dim"],
        voltage_grid=V_grid,
        latent_dim=ckpt["latent_dim"],
        diode_mean=ckpt["diode_mean"],
        diode_std=ckpt["diode_std"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diode_mean = ckpt["diode_mean"]
    diode_std = ckpt["diode_std"]

    # -----------------------------------------
    # Select samples
    # -----------------------------------------
    n_total = len(jv_input)

    if random_samples:
        indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    else:
        indices = np.arange(min(n_samples, n_total))

    # -----------------------------------------
    # Plot samples
    # -----------------------------------------
    for idx in indices:

        jv_in = torch.tensor(jv_input[idx:idx + 1]).to(device)
        weather_in = torch.tensor(weather[idx:idx + 1]).to(device)

        with torch.no_grad():
            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_in, weather_in)

        # JV prediction
        jv_pred = I_pred.cpu().numpy().squeeze()

        # Diode parameter prediction (denormalized)
        diode_pred = diode_pred_norm.cpu().numpy().squeeze() * diode_std + diode_mean

        rmse = np.sqrt(np.mean((jv_pred - jv_target[idx]) ** 2))

        save_path = save_dir / f"sample_{idx:05d}.png"
        predicted_smooth = savgol_filter(
            jv_pred,
            window_length=25,
            polyorder=3
        )

        target_smooth = savgol_filter(
            jv_target[idx],
            window_length=25,
            polyorder=3
        )

        measured_smooth = savgol_filter(
            jv_input[idx],
            window_length=25,
            polyorder=3
        )

        # Plot JV curves
        plot_single_jv(
            V_grid=V_grid,
            measured=measured_smooth,
            target=target_smooth,
            predicted=predicted_smooth,
            sample_idx=idx,
            rmse=rmse,
            save_path=save_path,
        )

        # Print diode parameters
        print(f"\nSaved: {save_path}")
        print(f"RMSE: {rmse:.6f}")

        print("True diode parameters:")
        print(f"  Iph = {diode_params_true[idx,0]:.6f}")
        print(f"  Io  = {diode_params_true[idx,1]:.6e}")
        print(f"  n   = {diode_params_true[idx,2]:.6f}")
        print(f"  Rs  = {diode_params_true[idx,3]:.6f}")
        print(f"  Rsh = {diode_params_true[idx,4]:.6f}")

        print("Predicted diode parameters:")
        print(f"  Iph = {diode_pred[0]:.6f}")
        print(f"  Io  = {diode_pred[1]:.6e}")
        print(f"  n   = {diode_pred[2]:.6f}")
        print(f"  Rs  = {diode_pred[3]:.6f}")
        print(f"  Rsh = {diode_pred[4]:.6f}")


if __name__ == "__main__":

    plot_multiple_predictions(
        dataset_path="data/jv_autoencoder_dataset_perovskite.npz",
        model_path="models/jv_autoencoder_perovskite.pth",
        n_samples=10,
        random_samples=True,
        save_dir="outputs/Perovskite_5&6/ae/curves",
    )
