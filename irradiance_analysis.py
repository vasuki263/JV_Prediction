import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

from model_autoencoder import PhysicsInformedJVModel  # adjust import


# ---------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Irradiance coefficient
# ---------------------------------------------------------
def irr_coeff(x, y):
    slope = np.polyfit(x, y, 1)[0]
    beta = slope * 1000 / np.mean(y)
    return beta, slope

def plot_param_log(
    irr,
    values,
    name,
    ylabel,
    outpath
):
    """
    Plot parameter vs log10(irradiance), but compute beta using linear irradiance.
    """

    # Compute beta using linear G (unchanged)
    beta, slope = irr_coeff(irr, values)

    # Log-transform irradiance
    irr_log = np.log10(irr)

    # Fit line in log-space (for visualization only)
    z = np.polyfit(irr_log, values, 1)
    fit = np.poly1d(z)

    xfit = np.linspace(irr_log.min(), irr_log.max(), 200)

    plt.figure(figsize=(7,5))

    plt.plot(
        irr_log,
        values,
        "o-",
        linewidth=2,
        markersize=6,
        label=name
    )

    plt.plot(
        xfit,
        fit(xfit),
        linewidth=3,
        label=f"β={beta:.3f}% per kW/m²"
    )

    plt.xlabel("log10(Irradiance)")
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return beta, slope

# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_param(irr, values, name, ylabel, outpath):

    beta, slope = irr_coeff(irr, values)

    z = np.polyfit(irr, values, 1)
    fit = np.poly1d(z)

    xfit = np.linspace(irr.min(), irr.max(), 200)

    plt.figure(figsize=(7,5))
    plt.plot(irr, values, "o-", linewidth=2, markersize=6, label=name)
    plt.plot(xfit, fit(xfit), linewidth=3, label=f"β={beta:.3f}% per kW/m²")

    plt.xlabel("Irradiance (W/m²)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return beta, slope


def plot_sensitivity_bar(betas, outpath):
    params = list(betas.keys())
    beta_values = [abs(betas[p][0]) for p in params]

    plt.figure(figsize=(7,5))
    plt.bar(params, beta_values, color="tab:blue")
    plt.ylabel("Sensitivity |β| (% per kW/m²)")
    plt.title("Irradiance Sensitivity per Parameter")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def extract_voc(V, I):
    V = np.asarray(V, float)
    I = np.asarray(I, float)

    # If all currents are negative → Voc near highest V
    if np.all(I < 0):
        k = np.argmin(np.abs(I))
        return V[k]

    # If all currents are positive → Voc near lowest V
    if np.all(I > 0):
        k = np.argmin(np.abs(I))
        return V[k]

    # Look for zero crossing between points
    for k in range(len(I) - 1):
        if (I[k] <= 0 and I[k+1] >= 0) or (I[k] >= 0 and I[k+1] <= 0):
            return np.interp(0.0, [I[k], I[k+1]], [V[k], V[k+1]])

    # Fallback: nearest-to-zero current
    k = np.argmin(np.abs(I))
    return V[k]

# ---------------------------------------------------------
# JV metric extraction
# ---------------------------------------------------------
def extract_jv_metrics(V, I):
    V = np.asarray(V, float)
    I = np.asarray(I, float)

    mask = np.isfinite(V) & np.isfinite(I)
    V = V[mask]
    I = I[mask]

    if len(V) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    idx = np.argsort(V)
    V = V[idx]
    I = I[idx]

    # Jsc
    Jsc = np.interp(0.0, V, I)

    # Voc
    Voc = extract_voc(V, I)

    # Use only V >= 0 region
    mask_vpos = V >= 0
    V_pos = V[mask_vpos]
    I_pos = I[mask_vpos]

    if len(V_pos) == 0:
        return Voc, Jsc, np.nan, np.nan, np.nan, np.nan

    # Power (absolute, since current is often negative)
    P = V_pos * I_pos
    P_abs = np.abs(P)

    k = np.argmax(P_abs)

    Vmpp = V_pos[k]
    Jmpp = I_pos[k]
    Pmpp = P_abs[k]

    # FF = Pmpp / (np.abs(Voc) * np.abs(Jsc) + 1e-12)
    denom = Voc * Jsc

    if (
            np.isnan(Voc)
            or np.isnan(Jsc)
            or abs(denom) < 1e-8
    ):
        FF = np.nan
    else:
        FF = Pmpp / denom

    return Voc, Jsc, Vmpp, Jmpp, Pmpp, FF


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def analyze_irradiance_physics_model(
    dataset_path,
    model_path,
    tech,
):

    output_dir = Path(f"outputs/{tech}/irradiance_analysis_physics")
    reset_dir(output_dir)

    # =====================================================
    # Load dataset
    # =====================================================
    data = np.load(dataset_path, allow_pickle=True)

    jv_input = data["jv_input"]          # (N, jv_dim)
    weather = data["weather"]            # (N, 2) → [temp, irr]
    V_grid = data["V_grid"]              # (jv_dim,)
    diode_stats = data["diode_params_stats"].item()

    diode_mean = diode_stats["mean"]
    diode_std = diode_stats["std"]

    jv_dim = jv_input.shape[1]
    weather_dim = weather.shape[1]

    # =====================================================
    # Load model
    # =====================================================
    ckpt = torch.load(model_path, map_location="cpu")

    model = PhysicsInformedJVModel(
        jv_dim=jv_input.shape[1],  # should be 200
        weather_dim=weather.shape[1],  # should be 2
        voltage_grid=V_grid,
        latent_dim=32,  # MUST match training
        hidden_dims=(256, 128),  # MUST match training
        diode_mean=diode_mean,
        diode_std=diode_std,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # =====================================================
    # Reference sample
    # =====================================================
    jv_sample = np.mean(jv_input, axis=0)
    weather_sample = np.mean(weather, axis=0)

    # irradiance index = 1 (weather = [temp, irr])
    irr_idx = 1

    # =====================================================
    # Irradiance sweep
    # =====================================================
    irr_values = np.arange(200, 1200, 50)

    vocs, jscs, ffs, pmpps = [], [], [], []
    Rs_list = []
    Rsh_list = []
    Rsh_norm_list = []

    for G in irr_values:

        jv_vec = jv_sample.copy()
        weather_vec = weather_sample.copy()

        # Direct irradiance (no scaler in dataset)
        weather_vec[irr_idx] = np.float32(G)

        jv_tensor = torch.tensor(jv_vec, dtype=torch.float32).unsqueeze(0)
        weather_tensor = torch.tensor(weather_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_tensor, weather_tensor)

        I_curve = I_pred.squeeze(0).cpu().numpy()
        # ---------------------------------------------
        # Extract diode parameters (denormalized)
        # ---------------------------------------------
        diode_raw = diode_pred_norm.squeeze(0).cpu().numpy() * diode_std + diode_mean

        Iph = diode_raw[0]
        Io = diode_raw[1]
        n = 0.8 + 2.2 * (1 / (1 + np.exp(-diode_raw[2])))  # sigmoid
        Rs = diode_raw[3]  # sigmoid
        Rsh = diode_raw[4]  # raw
        if Rsh < 0:
            Rsh = -Rsh

        Rs_list.append(Rs)
        Rsh_list.append(Rsh)

        Voc, Jsc, Vmpp, Jmpp, Pmpp, FF = extract_jv_metrics(V_grid, I_curve)
        # ---------------------------------------------
        # Normalized shunt resistance (dimensionless)
        # ---------------------------------------------
        Rsh_norm = Rsh / (Voc / (Jsc + 1e-12))
        Rsh_norm_list.append(Rsh_norm)

        vocs.append(Voc)
        jscs.append(Jsc)
        ffs.append(FF)
        pmpps.append(Pmpp)

    # =====================================================
    # Save predictions
    # =====================================================
    results = pd.DataFrame({
        "irradiance": irr_values,
        "Voc": vocs,
        "Jsc": jscs,
        "FF": ffs,
        "Pmpp": pmpps,
        "Rs": Rs_list,
        "Rsh": Rsh_list,
    })

    results.to_csv(output_dir / "irradiance_sweep_physics.csv", index=False)

    # =====================================================
    # Plots
    # =====================================================
    betas = {}

    betas["Voc"] = plot_param(irr_values, np.array(vocs), "Voc", "Voc (V)",
                              output_dir / "Voc_vs_irr_physics.png")

    betas["Jsc"] = plot_param(irr_values, np.array(jscs), "Jsc", "Jsc (A)",
                              output_dir / "Jsc_vs_irr_physics.png")

    betas["FF"] = plot_param(irr_values, np.array(ffs), "FF", "Fill Factor",
                             output_dir / "FF_vs_irr_physics.png")

    betas["Pmpp"] = plot_param(irr_values, np.array(pmpps), "Pmpp", "Pmpp (W)",
                               output_dir / "Pmpp_vs_irr_physics.png")
    betas["Rs"] = plot_param(
        irr_values,
        np.array(Rs_list),
        "Rs",
        "Series Resistance (Ω)",
        output_dir / "Rs_vs_irr_physics.png"
    )

    betas["Rsh"] = plot_param(
        irr_values,
        np.array(Rsh_list),
        "Rsh",
        "Shunt Resistance (Ω)",
        output_dir / "Rsh_vs_irr_physics.png"
    )

    plot_sensitivity_bar(betas, output_dir / "irradiance_sensitivity_bar_physics.png")

    # =====================================================
    # Save coefficients
    # =====================================================
    pd.DataFrame(betas, index=["beta_%_per_Wm2", "slope"]).T.to_csv(
        output_dir / "irradiance_coefficients_physics.csv"
    )

    print("\nIrradiance coefficients (physics-informed model):")
    print(pd.DataFrame(betas, index=["beta_%_per_Wm2", "slope"]).T)


if __name__ == "__main__":
    analyze_irradiance_physics_model(
        dataset_path="data/jv_autoencoder_dataset_perovskite.npz",
        model_path="models/jv_autoencoder_perovskite.pth",
        tech="Perovskite_5&6",
    )

    analyze_irradiance_physics_model(
        "data/jv_autoencoder_dataset_silicon.npz",
        "models/jv_autoencoder_silicon.pth",
        tech="SunPower_5-1",
    )
