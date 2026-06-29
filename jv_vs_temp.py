import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

from model_autoencoder import JVAutoencoder


# ---------------------------------------------------------
# Directory helper
# ---------------------------------------------------------
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Full JV Temperature Analysis for Autoencoder
# ---------------------------------------------------------
def analyze_temperature_AE(
    dataset_path,
    model_path,
    tech,
    temp_range=np.arange(20, 70, 10)
):

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------
    outdir = Path(f"outputs/{tech}/jv_temperature_analysis")
    reset_dir(outdir)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    data = np.load(dataset_path, allow_pickle=True)
    jv_input = data["jv_input"]          # (N, num_points)
    V_grid   = data["V_grid"]            # (num_points,)
    weather  = data["weather"]           # (N, 2) = [module_temp, module_irr]

    # Use MEAN sample for stability
    base_jv = np.mean(jv_input, axis=0).astype(np.float32)
    base_weather = np.mean(weather, axis=0).astype(np.float32)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    ckpt = torch.load(model_path, map_location="cpu")
    jv_dim = ckpt["jv_dim"]
    weather_dim = ckpt["weather_dim"]
    latent_dim = ckpt["latent_dim"]

    model = JVAutoencoder(jv_dim, weather_dim, latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -----------------------------------------------------
    # Sweep temperature
    # -----------------------------------------------------
    all_curves = []
    temps = temp_range

    for T in temps:

        # Copy base inputs
        jv_in = base_jv.copy()
        w = base_weather.copy()

        # Replace module temperature
        w[0] = T  # weather vector = [temp_mod, irr_mod]

        # Convert to tensors
        jv_tensor = torch.tensor(jv_in, dtype=torch.float32).unsqueeze(0)
        w_tensor  = torch.tensor(w, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            jv_recon, _ = model(jv_tensor, w_tensor)

        jv_recon = jv_recon.numpy().flatten()
        all_curves.append(jv_recon)

    all_curves = np.array(all_curves)   # shape = (len(temps), num_points)

    # -----------------------------------------------------
    # Knee tracking
    # -----------------------------------------------------
    knee_data = []

    for i, T in enumerate(temps):
        J = all_curves[i]
        J_smooth = savgol_filter(J, window_length=11, polyorder=3)

        # First and second derivatives
        dJdV = np.gradient(J_smooth, V_grid)
        d2JdV2 = np.gradient(dJdV, V_grid)

        # Knee = maximum curvature magnitude
        # knee_idx = np.argmax(np.abs(d2JdV2))

        # Vknee = V_grid[knee_idx]
        # Jknee = J[knee_idx]
        Voc_idx = np.argmin(np.abs(J))
        Voc = V_grid[Voc_idx]

        mask = (V_grid > 0.7 * Voc) & (V_grid < 0.9 * Voc) #& (J > 0.3*np.max(J))

        # --- Find knee only inside combined mask ---
        local_idx = np.argmax(np.abs(d2JdV2[mask]))
        global_idx = np.where(mask)[0][local_idx]

        Vknee = V_grid[global_idx]
        Jknee = J[global_idx]

        knee_data.append([T, Vknee, Jknee])

    knee_df = pd.DataFrame(
        knee_data,
        columns=["Temperature", "Vknee", "Jknee"]
    )

    knee_df.to_csv(
        outdir / "knee_positions.csv",
        index=False
    )

    # -----------------------------------------------------
    # Save curves
    # -----------------------------------------------------
    df = pd.DataFrame(all_curves, columns=[f"J_{i}" for i in range(len(V_grid))])
    df.insert(0, "Temperature", temps)
    df.to_csv(outdir / "JV_vs_T.csv", index=False)

    # -----------------------------------------------------
    # 1. Smoothed JV(T) Family Plot
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.plasma
    norm = plt.Normalize(temps.min(), temps.max())

    # for i, T in enumerate(temps):
    #     ax.plot(
    #         V_grid, all_curves[i],
    #         color=cmap(norm(T)),
    #         linewidth=2.5
    #     )
    for i, T in enumerate(temps):
        J = all_curves[i]

        ax.plot(
            V_grid,
            J,
            color=cmap(norm(T)),
            linewidth=2.5
        )

        Vknee = knee_df.iloc[i]["Vknee"]
        Jknee = knee_df.iloc[i]["Jknee"]

        ax.scatter(
            Vknee,
            Jknee,
            color="#00FF00",
            edgecolor='black',
            s=30,
            zorder=10,
            label="Detected knee" if i == 0 else None
        )

    ax.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar

    fig.colorbar(sm, ax=ax, label="Temperature (°C)")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (normalized)")
    ax.set_title("JV Curves vs Temperature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "JV_vs_T_smooth.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------
    # 2. dJ/dT
    # -----------------------------------------------------
    dJdT = np.gradient(all_curves, temps, axis=0)
    # -----------------------------------------------------
    # Compute scalar sensitivity scores per voltage
    # -----------------------------------------------------
    voltages = [20, 40, 60, 80]
    sensitivity_scores = []

    for V_target in voltages:
        idx = np.argmin(np.abs(V_grid - V_target))
        curve = dJdT[:, idx]

        # Normalize and take mean absolute sensitivity
        curve_norm = curve / np.max(np.abs(curve))
        score = np.mean(np.abs(curve_norm))

        sensitivity_scores.append(score)

    plt.figure(figsize=(8, 6))

    for V_target in [20, 40, 60, 80]:
        idx = np.argmin(np.abs(V_grid - V_target))
        curve = dJdT[:, idx]

        # Normalize to 0–1
        curve_norm = curve / np.max(np.abs(curve))

        plt.plot(
            temps,
            curve_norm,
            linewidth=2,
            label=f"V={V_target} V"
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Normalized d|J|/dT")
    plt.title("Normalized Temperature Sensitivity at Selected Voltages")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "dJdT_slices_normalized.png", dpi=200)
    plt.close()

    # -----------------------------------------------------
    # 3. ΔJ(V) Between Adjacent Temperatures
    # -----------------------------------------------------
    plt.figure(figsize=(8,6))

    for i in range(len(temps)-1):
        dJ = all_curves[i+1] - all_curves[i]
        plt.plot(V_grid, dJ, linewidth=2, label=f"{temps[i]}→{temps[i+1]}°C")

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Voltage (V)")
    plt.ylabel("ΔJ (normalized)")
    plt.title("Change in JV Curve Between Adjacent Temperatures")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "delta_J_between_temperatures.png", dpi=200)
    plt.close()

    # -----------------------------------------------------
    # Bar chart of sensitivity scores
    # -----------------------------------------------------
    plt.figure(figsize=(8, 4))
    labels = [f"V={v} V" for v in voltages]

    plt.barh(labels, sensitivity_scores, color="tab:blue")
    plt.xlabel("Sensitivity Score (normalized)")
    plt.title("Temperature Sensitivity Scores per Voltage")
    plt.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / "temperature_sensitivity_scores.png", dpi=200)
    plt.close()

    print(f"\n[INFO] Saved AE temperature analysis to {outdir}")

