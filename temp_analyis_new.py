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
# Plot helper (with error bars)
# ---------------------------------------------------------
def plot_param(temps, mean_vals, std_vals, name, ylabel, outpath):
    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)

    gamma, slope = temp_coeff(temps, mean_vals)

    z = np.polyfit(temps, mean_vals, 1)
    fit = np.poly1d(z)
    xfit = np.linspace(temps.min(), temps.max(), 200)

    plt.figure(figsize=(7,5))

    # Error bars
    plt.errorbar(
        temps,
        mean_vals,
        yerr=std_vals,
        fmt="o",
        markersize=6,
        capsize=4,
        color="orange",
        label=name
    )

    # Regression line
    plt.plot(xfit, fit(xfit), "r--", linewidth=2, label=f"γ={gamma:.3f}%/°C")

    # Optional reference temperature
    plt.axvline(25, color="green", linestyle="--", alpha=0.6)

    plt.xlabel("Module Temperature (°C)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
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

    output_dir = Path(f"outputs/{tech}/temperature_analysis/diff_rep")
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
    # FAST Temperature sweep using K random samples
    # =====================================================
    temps = np.arange(20, 75, 5)
    K = 40  # number of samples per temperature

    N = len(jv_in)
    rng = np.random.default_rng(42)

    vocs_mean, vocs_std = [], []
    jscs_mean, jscs_std = [], []
    vmpps_mean, vmpps_std = [], []
    jmpps_mean, jmpps_std = [], []
    pmpps_mean, pmpps_std = [], []
    ffs_mean, ffs_std = [], []
    rss_mean, rss_std = [], []
    rshs_mean, rshs_std = [], []

    for T in temps:

        idxs = rng.choice(N, size=K, replace=False)

        Voc_list, Jsc_list, Vmpp_list, Jmpp_list, Pmpp_list, FF_list = [], [], [], [], [], []
        Rs_list, Rsh_list = [], []

        for i in idxs:

            w = weather[i].copy()
            w[0] = T

            jv_t = torch.tensor(jv_in[i], dtype=torch.float32, device=device).unsqueeze(0)
            w_t  = torch.tensor(w,        dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

            I_pred_np = I_pred.squeeze(0).cpu().numpy()

            Voc, Jsc, Vmpp, Jmpp, Pmpp, FF = extract_jv_physics(V_grid, I_pred_np)

            diode_params = diode_pred_norm.squeeze(0).cpu().numpy()
            diode_params = diode_params * diode_stats["std"] + diode_stats["mean"]
            Iph, Io, n, Rs, Rsh = diode_params

            Voc_list.append(Voc)
            Jsc_list.append(Jsc)
            Vmpp_list.append(Vmpp)
            Jmpp_list.append(Jmpp)
            Pmpp_list.append(Pmpp)
            FF_list.append(FF)
            Rs_list.append(Rs)
            Rsh_list.append(Rsh)

        vocs_mean.append(np.nanmean(Voc_list))
        vocs_std.append(np.nanstd(Voc_list))

        jscs_mean.append(np.nanmean(Jsc_list))
        jscs_std.append(np.nanstd(Jsc_list))

        vmpps_mean.append(np.nanmean(Vmpp_list))
        vmpps_std.append(np.nanstd(Vmpp_list))

        jmpps_mean.append(np.nanmean(Jmpp_list))
        jmpps_std.append(np.nanstd(Jmpp_list))

        pmpps_mean.append(np.nanmean(Pmpp_list))
        pmpps_std.append(np.nanstd(Pmpp_list))

        ffs_mean.append(np.nanmean(FF_list))
        ffs_std.append(np.nanstd(FF_list))

        rss_mean.append(np.nanmean(Rs_list))
        rss_std.append(np.nanstd(Rs_list))

        rshs_mean.append(np.nanmean(Rsh_list))
        rshs_std.append(np.nanstd(Rsh_list))

    # =====================================================
    # Save predictions
    # =====================================================
    results = pd.DataFrame({
        "temperature": temps,
        "Voc_mean": vocs_mean,
        "Voc_std": vocs_std,
        "Jsc_mean": jscs_mean,
        "Jsc_std": jscs_std,
        "Vmpp_mean": vmpps_mean,
        "Vmpp_std": vmpps_std,
        "Jmpp_mean": jmpps_mean,
        "Jmpp_std": jmpps_std,
        "Pmpp_mean": pmpps_mean,
        "Pmpp_std": pmpps_std,
        "FF_mean": ffs_mean,
        "FF_std": ffs_std,
        "Rs_mean": rss_mean,
        "Rs_std": rss_std,
        "Rsh_mean": rshs_mean,
        "Rsh_std": rshs_std,
    })

    results.to_csv(output_dir / "temperature_sweep_mean_std.csv", index=False)

    # =====================================================
    # Plots
    # =====================================================
    gammas = {}

    gammas["Voc"] = plot_param(temps, vocs_mean, vocs_std, "Voc", "Voc (V)", output_dir / "Voc_vs_T.png")
    gammas["Jsc"] = plot_param(temps, jscs_mean, jscs_std, "Jsc", "Jsc (A)", output_dir / "Jsc_vs_T.png")
    gammas["Vmpp"] = plot_param(temps, vmpps_mean, vmpps_std, "Vmpp", "Vmpp (V)", output_dir / "Vmpp_vs_T.png")
    gammas["Jmpp"] = plot_param(temps, jmpps_mean, jmpps_std, "Jmpp", "Jmpp (A)", output_dir / "Jmpp_vs_T.png")
    gammas["Pmpp"] = plot_param(temps, pmpps_mean, pmpps_std, "Pmpp", "Pmpp (W)", output_dir / "Pmpp_vs_T.png")
    gammas["FF"] = plot_param(temps, ffs_mean, ffs_std, "FF", "Fill Factor", output_dir / "FF_vs_T.png")

    gammas["Rs"] = plot_param(temps, rss_mean, rss_std, "Rs", "Series Resistance (Ω)", output_dir / "Rs_vs_T.png")
    gammas["Rsh"] = plot_param(temps, rshs_mean, rshs_std, "Rsh", "Shunt Resistance (Ω)", output_dir / "Rsh_vs_T.png")

    plot_sensitivity_bar(gammas, output_dir / "temp_sensitivity_bar.png")

    pd.DataFrame(gammas, index=["gamma_%_per_C", "slope"]).T.to_csv(
        output_dir / "temperature_coefficients.csv"
    )

    print("\nTemperature coefficients:")
    print(pd.DataFrame(gammas, index=["gamma_%_per_C", "slope"]).T)


if __name__ == "__main__":
    analyze_temperature(
        "data/jv_autoencoder_dataset_perovskite.npz",
        "models/jv_autoencoder_perovskite.pth",
        tech="Perovskite_5&6",
    )
    analyze_temperature(
        "data/jv_autoencoder_dataset_silicon.npz",
        "models/jv_autoencoder_silicon.pth",
        tech="SunPower_5-1",
    )
