import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

from model_autoencoder import PhysicsInformedJVModel
from eval_AE import extract_jv_physics


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
    gamma = slope / np.mean(y) * 100
    return gamma, slope


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_param(x, values, name, ylabel, xlabel, outpath):

    gamma, slope = irr_coeff(x, values)

    z = np.polyfit(x, values, 1)
    fit = np.poly1d(z)

    xx = np.linspace(x.min(), x.max(), 200)

    plt.figure(figsize=(7,5))
    plt.plot(x, values, "o-", linewidth=2, markersize=6, label=name)
    plt.plot(xx, fit(xx), linewidth=3,
             label=f"γ={gamma:.3f}%/(W/m²)")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return gamma, slope


def plot_sensitivity_bar(coeffs, outpath):

    params = list(coeffs.keys())
    values = [abs(coeffs[p][0]) for p in params]

    plt.figure(figsize=(7,5))
    plt.bar(params, values)
    plt.ylabel("Sensitivity")
    plt.title("Irradiance Sensitivity")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def analyze_irradiance(dataset_path,
                       model_path,
                       tech,
                       irradiance_index=1):

    output_dir = Path(f"outputs/{tech}/irradiance_analysis")
    reset_dir(output_dir)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    data = np.load(dataset_path, allow_pickle=True)

    jv_in = data["jv_input"].astype(np.float32)
    weather = data["weather"].astype(np.float32)
    V_grid = data["V_grid"].astype(np.float32)

    diode_stats = data["diode_params_stats"].item()
    # True (unnormalized) diode parameters from dataset
    diode_true = data["diode_params"]

    Iph_true = diode_true[:, 0]
    Io_true = diode_true[:, 1]
    n_true = diode_true[:, 2]
    Rs_true = diode_true[:, 3]
    Rsh_true = diode_true[:, 4]

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Representative sample
    # -----------------------------------------------------
    idx = np.argmin(np.abs(weather[:, irradiance_index] - 800))

    sample_jv = jv_in[idx].copy()
    sample_weather = weather[idx].copy()

    # -----------------------------------------------------
    # Irradiance sweep
    # -----------------------------------------------------
    irradiances = np.arange(200, 1100, 100)

    vocs = []
    jscs = []
    vmpps = []
    jmpps = []
    pmpps = []
    ffs = []

    rss = []
    rshs = []

    for G in irradiances:

        w = sample_weather.copy()
        w[irradiance_index] = G

        jv_t = torch.tensor(sample_jv,
                            dtype=torch.float32,
                            device=device).unsqueeze(0)

        w_t = torch.tensor(w,
                           dtype=torch.float32,
                           device=device).unsqueeze(0)

        with torch.no_grad():
            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

        I_pred = I_pred.squeeze().cpu().numpy()

        Voc, Jsc, Vmpp, Jmpp, Pmpp, FF = extract_jv_physics(
            V_grid,
            I_pred
        )

        diode = diode_pred_norm.squeeze().cpu().numpy()
        diode = diode * diode_stats["std"] + diode_stats["mean"]

        Iph, Io, n, Rs, Rsh = diode

        vocs.append(Voc)
        jscs.append(Jsc)
        vmpps.append(Vmpp)
        jmpps.append(Jmpp)
        pmpps.append(Pmpp)
        ffs.append(FF)

        rss.append(Rs)
        rshs.append(Rsh)

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------
    df = pd.DataFrame({

        "Irradiance": irradiances,

        "Voc": vocs,
        "Jsc": jscs,
        "Vmpp": vmpps,
        "Jmpp": jmpps,
        "Pmpp": pmpps,
        "FF": ffs,
        "Rs": rss,
        "Rsh": rshs

    })

    df.to_csv(output_dir / "irradiance_sweep.csv",
              index=False)

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    coeffs = {}

    coeffs["Voc"] = plot_param(
        irradiances,
        np.array(vocs),
        "Voc",
        "Voc (V)",
        "Irradiance (W/m²)",
        output_dir / "Voc_vs_G.png"
    )

    coeffs["Jsc"] = plot_param(
        irradiances,
        np.array(jscs),
        "Jsc",
        "Jsc (A)",
        "Irradiance (W/m²)",
        output_dir / "Jsc_vs_G.png"
    )

    coeffs["Vmpp"] = plot_param(
        irradiances,
        np.array(vmpps),
        "Vmpp",
        "Vmpp (V)",
        "Irradiance (W/m²)",
        output_dir / "Vmpp_vs_G.png"
    )

    coeffs["Jmpp"] = plot_param(
        irradiances,
        np.array(jmpps),
        "Jmpp",
        "Jmpp (A)",
        "Irradiance (W/m²)",
        output_dir / "Jmpp_vs_G.png"
    )

    coeffs["Pmpp"] = plot_param(
        irradiances,
        np.array(pmpps),
        "Pmpp",
        "Pmpp (W)",
        "Irradiance (W/m²)",
        output_dir / "Pmpp_vs_G.png"
    )

    coeffs["FF"] = plot_param(
        irradiances,
        np.array(ffs),
        "FF",
        "Fill Factor",
        "Irradiance (W/m²)",
        output_dir / "FF_vs_G.png"
    )

    coeffs["Rs"] = plot_param(
        irradiances,
        np.array(rss),
        "Rs",
        "Series Resistance (Ω)",
        "Irradiance (W/m²)",
        output_dir / "Rs_vs_G.png"
    )

    coeffs["Rsh"] = plot_param(
        irradiances,
        np.array(rshs),
        "Rsh",
        "Shunt Resistance (Ω)",
        "Irradiance (W/m²)",
        output_dir / "Rsh_vs_G.png"
    )

    plot_sensitivity_bar(
        coeffs,
        output_dir / "irradiance_sensitivity_bar.png"
    )

    pd.DataFrame(
        coeffs,
        index=["gamma", "slope"]
    ).T.to_csv(
        output_dir / "irradiance_coefficients.csv"
    )

    print(pd.DataFrame(
        coeffs,
        index=["gamma", "slope"]
    ).T)

    # -----------------------------------------------------
    # Plot TRUE diode parameters vs irradiance
    # (mean value in each irradiance bin)
    # -----------------------------------------------------

    irr = weather[:, irradiance_index]

    irradiance_bins = np.arange(200, 1100, 100)

    true_params = {
        "Iph": (Iph_true, "Photocurrent (A)"),
        "Io": (Io_true, "Saturation Current (A)"),
        "n": (n_true, "Ideality Factor"),
        "Rs": (Rs_true, "Series Resistance (Ω)"),
        "Rsh": (Rsh_true, "Shunt Resistance (Ω)")
    }

    for name, (values, ylabel) in true_params.items():

        means = []

        for G in irradiance_bins:
            mask = np.abs(irr - G) <= 50  # ±50 W/m² bin

            if np.sum(mask) > 0:
                means.append(np.mean(values[mask]))
            else:
                means.append(np.nan)

        means = np.array(means)

        gamma, slope = irr_coeff(irradiance_bins, means)

        z = np.polyfit(irradiance_bins[~np.isnan(means)],
                       means[~np.isnan(means)], 1)

        fit = np.poly1d(z)

        xx = np.linspace(
            irradiance_bins.min(),
            irradiance_bins.max(),
            200
        )

        plt.figure(figsize=(7, 5))

        plt.plot(
            irradiance_bins,
            means,
            "o-",
            linewidth=2,
            markersize=6,
            label="True"
        )

        plt.plot(
            xx,
            fit(xx),
            linewidth=3,
            label=f"γ={gamma:.3f}%/(W/m²)"
        )

        plt.xlabel("Irradiance (W/m²)")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{name}_true_vs_G.png",
            dpi=150
        )

        plt.close()


if __name__ == "__main__":

    analyze_irradiance(
        "data/jv_autoencoder_dataset_perovskite.npz",
        "models/jv_autoencoder_perovskite.pth",
        tech="Perovskite_5&6",
        irradiance_index=1     # <-- change if irradiance is another column
    )

    analyze_irradiance(
        "data/jv_autoencoder_dataset_silicon.npz",
        "models/jv_autoencoder_silicon.pth",
        tech="SunPower_5-1",
        irradiance_index=1
    )