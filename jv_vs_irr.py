import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
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
# Full JV Irradiance Analysis for Autoencoder
# ---------------------------------------------------------
def analyze_irradiance_AE(
    dataset_path,
    model_path,
    tech,
    irr_values=np.arange(200, 1200, 200)
):

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------
    outdir = Path(f"outputs/{tech}/jv_irradiance_analysis")
    reset_dir(outdir)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    data = np.load(dataset_path, allow_pickle=True)
    jv_input = data["jv_input"]          # (N, num_points)
    V_grid   = data["V_grid"]            # (num_points,)
    weather  = data["weather"]           # (N, 2) = [temp, irr]

    # Use mean sample for stability
    base_jv = np.mean(jv_input, axis=0).astype(np.float32)
    base_weather = np.mean(weather, axis=0).astype(np.float32)

    # -----------------------------------------------------
    # Load AE model
    # -----------------------------------------------------
    ckpt = torch.load(model_path, map_location="cpu")
    jv_dim = ckpt["jv_dim"]
    weather_dim = ckpt["weather_dim"]
    latent_dim = ckpt["latent_dim"]

    model = JVAutoencoder(jv_dim, weather_dim, latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -----------------------------------------------------
    # Sweep irradiance
    # -----------------------------------------------------
    all_curves = []

    for G in irr_values:

        jv_in = base_jv.copy()
        w = base_weather.copy()

        # Replace irradiance
        w[1] = G

        jv_tensor = torch.tensor(jv_in, dtype=torch.float32).unsqueeze(0)
        w_tensor  = torch.tensor(w, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            jv_recon, _ = model(jv_tensor, w_tensor)

        all_curves.append(jv_recon.numpy().flatten())

    all_curves = np.array(all_curves)  # shape = (len(irr_values), num_points)

    # -----------------------------------------------------
    # Knee detection for irradiance sweep
    # -----------------------------------------------------
    knee_data = []

    for i, G in enumerate(irr_values):
        J = all_curves[i]
        J_smooth = savgol_filter(J, window_length=11, polyorder=3)

        # First and second derivatives
        dJdV = np.gradient(J_smooth, V_grid)
        d2JdV2 = np.gradient(dJdV, V_grid)

        # Knee = maximum curvature magnitude
        # knee_idx = np.argmax(np.abs(d2JdV2))
        # --- Compute Voc ---
        Voc_idx = np.argmin(np.abs(J))
        Voc = V_grid[Voc_idx]

        mask = (V_grid > 0.6 * Voc) & (V_grid < 0.9 * Voc)

        # --- Find knee only inside combined mask ---
        local_idx = np.argmax(np.abs(d2JdV2[mask]))
        global_idx = np.where(mask)[0][local_idx]

        Vknee = V_grid[global_idx]
        Jknee = J[global_idx]

        knee_data.append([G, Vknee, Jknee])

    knee_df = pd.DataFrame(
        knee_data,
        columns=["Irradiance", "Vknee", "Jknee"]
    )

    knee_df.to_csv(outdir / "knee_positions_irradiance.csv", index=False)

    # -----------------------------------------------------
    # Save curves
    # -----------------------------------------------------
    df = pd.DataFrame(all_curves, columns=[f"J_{i}" for i in range(len(V_grid))])
    df.insert(0, "Irradiance", irr_values)
    df.to_csv(outdir / "JV_vs_irradiance.csv", index=False)

    # -----------------------------------------------------
    # JV vs Irradiance with Knee Points
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.plasma
    norm = plt.Normalize(irr_values.min(), irr_values.max())

    for i, G in enumerate(irr_values):
        J = all_curves[i]

        # Plot JV curve
        ax.plot(
            V_grid,
            J,
            color=cmap(norm(G)),
            linewidth=2.5
        )

        # Plot knee point
        Vknee = knee_df.iloc[i]["Vknee"]
        Jknee = knee_df.iloc[i]["Jknee"]

        ax.scatter(
            Vknee,
            Jknee,
            color="#00FF00",  # bright green
            edgecolor="black",
            s=30,
            zorder=10,
            label="Detected knee" if i == 0 else None
        )

    ax.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Irradiance (W/m²)")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (normalized)")
    ax.set_title("JV Curves vs Irradiance with Knee Points")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "JV_vs_irradiance_with_knees.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------
    # 2. dJ/dG vs irr
    # -----------------------------------------------------
    # Compute derivative w.r.t irradiance
    dJdG = np.gradient(all_curves, irr_values, axis=0)
    # -----------------------------------------------------
    # Compute scalar sensitivity scores per voltage
    # -----------------------------------------------------
    voltages = [20, 40, 60, 80]
    sensitivity_scores = []

    for V_target in voltages:
        idx = np.argmin(np.abs(V_grid - V_target))
        curve = dJdG[:, idx]

        # Normalize and take mean absolute sensitivity
        curve_norm = curve / np.max(np.abs(curve))
        score = np.mean(np.abs(curve_norm))

        sensitivity_scores.append(score)

    plt.figure(figsize=(8, 6))

    for V_target in [20, 40, 60, 80]:
        idx = np.argmin(np.abs(V_grid - V_target))
        curve = dJdG[:, idx]

        # Normalize to 0–1
        curve_norm = curve / np.max(np.abs(curve))

        plt.plot(
            irr_values,
            curve_norm,
            linewidth=2,
            label=f"V={V_target} V"
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Irradiance (W/m²)")
    plt.ylabel("Normalized d|J|/dG")
    plt.title("Normalized Irradiance Sensitivity at Selected Voltages")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "dJdG_slices_normalized.png", dpi=200)
    plt.close()

    # -----------------------------------------------------
    # 3. ΔJ(V) Between Adjacent Irradiances
    # -----------------------------------------------------
    plt.figure(figsize=(8,6))

    for i in range(len(irr_values)-1):
        dJ = all_curves[i+1] - all_curves[i]
        plt.plot(
            V_grid,
            dJ,
            linewidth=2,
            label=f"{irr_values[i]}→{irr_values[i+1]} W/m²"
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Voltage (V)")
    plt.ylabel("ΔJ (normalized)")
    plt.title("Change in JV Curve Between Adjacent Irradiances")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "delta_J_between_irradiances.png", dpi=200)
    plt.close()

    # -----------------------------------------------------
    # Bar chart of sensitivity scores
    # -----------------------------------------------------
    plt.figure(figsize=(8, 4))
    labels = [f"V={v} V" for v in voltages]

    plt.barh(labels, sensitivity_scores, color="tab:blue")
    plt.xlabel("Sensitivity Score (normalized)")
    plt.title("Irradiance Sensitivity Scores per Voltage")
    plt.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / "irradiance_sensitivity_scores.png", dpi=200)
    plt.close()

    print(f"\n[INFO] Saved AE irradiance analysis to {outdir}")
