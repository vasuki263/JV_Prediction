import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

from model_autoencoder import PhysicsInformedJVModel
from eval_AE import extract_jv_physics   # adjust import


# ---------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Temperature coefficient
# ---------------------------------------------------------
def temp_coeff(x, y):
    slope = np.polyfit(x, y, 1)[0]
    gamma = slope / np.mean(y) * 100
    return gamma, slope


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_param(temps, values, name, ylabel, outpath):
    gamma, slope = temp_coeff(temps, values)

    z = np.polyfit(temps, values, 1)
    fit = np.poly1d(z)

    xfit = np.linspace(temps.min(), temps.max(), 200)

    plt.figure(figsize=(7,5))
    plt.plot(temps, values, "o-", linewidth=2, markersize=6, label=name)
    plt.plot(xfit, fit(xfit), linewidth=3, label=f"γ={gamma:.3f}%/°C")

    plt.xlabel("Module Temperature (°C)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return gamma, slope


def plot_sensitivity_bar(betas, outpath):
    params = list(betas.keys())
    beta_values = [abs(betas[p][0]) for p in params]

    plt.figure(figsize=(7,5))
    plt.bar(params, beta_values, color="tab:blue")
    plt.ylabel("Sensitivity (γ %/°C)")
    plt.title("Temperature Sensitivity per Parameter")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def analyze_temperature(dataset_path, model_path, tech):

    output_dir = Path(f"outputs/{tech}/temperature_analysis_new/new")
    reset_dir(output_dir)

    # =====================================================
    # Load dataset
    # =====================================================
    data = np.load(dataset_path, allow_pickle=True)

    jv_in = data["jv_input"].astype(np.float32)
    weather = data["weather"].astype(np.float32)
    V_grid = data["V_grid"].astype(np.float32)

    diode_stats = data["diode_params_stats"].item()

    # =====================================================
    # Load model
    # =====================================================
    ckpt = torch.load(model_path, map_location="cpu")

    model = PhysicsInformedJVModel(
        jv_dim=jv_in.shape[1],
        weather_dim=weather.shape[1],
        voltage_grid=V_grid,
        latent_dim=ckpt["latent_dim"],
        diode_mean=ckpt["diode_mean"],
        diode_std=ckpt["diode_std"],
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # =====================================================
    # Pick representative sample near 30°C
    # =====================================================
    temps_raw = weather[:, 0]
    idx = np.argmin(np.abs(temps_raw - 30))
    sample_jv = jv_in[idx].copy()
    sample_weather = weather[idx].copy()

    # =====================================================
    # Temperature sweep
    # =====================================================
    temps = np.arange(20, 75, 5)

    vocs, jscs, vmpps, jmpps, pmpps, ffs = [], [], [], [], [], []
    rss, rshs = [], []

    for T in temps:
        w = sample_weather.copy()
        w[0] = T  # temperature is weather[0]

        jv_t = torch.tensor(sample_jv, dtype=torch.float32, device=device).unsqueeze(0)
        w_t = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

        I_pred = I_pred.squeeze(0).cpu().numpy()

        # Extract JV physics
        Voc, Jsc, Vmpp, Jmpp, Pmpp, FF = extract_jv_physics(V_grid, I_pred)

        # Unnormalize diode parameters
        diode_params = diode_pred_norm.squeeze(0).cpu().numpy()
        diode_params = diode_params * diode_stats["std"] + diode_stats["mean"]

        Iph, Io, n, Rs, Rsh = diode_params
        if Rsh < 0:
            Rsh = -Rsh

        vocs.append(Voc)
        jscs.append(Jsc)
        vmpps.append(Vmpp)
        jmpps.append(Jmpp)
        pmpps.append(Pmpp)
        ffs.append(FF)

        rss.append(Rs)
        rshs.append(Rsh)

    # =====================================================
    # Save predictions
    # =====================================================
    results = pd.DataFrame({
        "temperature": temps,
        "Voc": vocs,
        "Jsc": jscs,
        "Vmpp": vmpps,
        "Jmpp": jmpps,
        "Pmpp": pmpps,
        "FF": ffs,
        "Rs": rss,
        "Rsh": rshs
    })

    results.to_csv(output_dir / "temperature_sweep.csv", index=False)

    # =====================================================
    # Plots
    # =====================================================
    gammas = {}

    gammas["Voc"] = plot_param(temps, np.array(vocs), "Voc", "Voc (V)", output_dir / "Voc_vs_T.png")
    gammas["Jsc"] = plot_param(temps, np.array(jscs), "Jsc", "Jsc (A)", output_dir / "Jsc_vs_T.png")
    gammas["Vmpp"] = plot_param(temps, np.array(vmpps), "Vmpp", "Vmpp (V)", output_dir / "Vmpp_vs_T.png")
    gammas["Jmpp"] = plot_param(temps, np.array(jmpps), "Jmpp", "Jmpp (A)", output_dir / "Jmpp_vs_T.png")
    gammas["Pmpp"] = plot_param(temps, np.array(pmpps), "Pmpp", "Pmpp (W)", output_dir / "Pmpp_vs_T.png")
    gammas["FF"] = plot_param(temps, np.array(ffs), "FF", "Fill Factor", output_dir / "FF_vs_T.png")

    gammas["Rs"] = plot_param(temps, np.array(rss), "Rs", "Series Resistance (Ω)", output_dir / "Rs_vs_T.png")
    gammas["Rsh"] = plot_param(temps, np.array(rshs), "Rsh", "Shunt Resistance (Ω)", output_dir / "Rsh_vs_T.png")

    # Sensitivity bar chart
    plot_sensitivity_bar(gammas, output_dir / "temp_sensitivity_bar.png")

    # Save coefficients
    pd.DataFrame(gammas, index=["gamma_%_per_C", "slope"]).T.to_csv(
        output_dir / "temperature_coefficients.csv"
    )

    print("\nTemperature coefficients:")
    print(pd.DataFrame(gammas, index=["gamma_%_per_C", "slope"]).T)

if __name__ == "__main__":
    # analyze_temperature("data/jv_autoencoder_dataset_perovskite.npz", "models/jv_autoencoder_perovskite.pth", tech="Perovskite_5&6")
    analyze_temperature(
        "data/jv_autoencoder_dataset_silicon.npz",
        "models/jv_autoencoder_silicon.pth",
        tech="SunPower_5-1",
    )