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
def irradiance_coeff(x, y):
    slope = np.polyfit(x, y, 1)[0]
    gamma = slope / np.mean(y) * 100
    return gamma, slope


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_param(irr, mean_vals, std_vals, name, ylabel, outpath):

    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)

    gamma, slope = irradiance_coeff(irr, mean_vals)

    z = np.polyfit(irr, mean_vals, 1)
    fit = np.poly1d(z)

    xfit = np.linspace(irr.min(), irr.max(), 300)

    plt.figure(figsize=(7,5))

    plt.errorbar(
        irr,
        mean_vals,
        yerr=std_vals,
        fmt="o",
        color="orange",
        capsize=4,
        label=name
    )

    plt.plot(
        xfit,
        fit(xfit),
        "r--",
        linewidth=2,
        label=f"γ={gamma:.4f}%/(W/m²)"
    )

    ymin = max(0, np.min(mean_vals) - 0.05 * np.min(mean_vals))
    plt.ylim(bottom=ymin)
    ymax = np.max(mean_vals) + 0.05 * np.max(mean_vals)
    plt.ylim(top=ymax)

    plt.xlabel("Irradiance (W/m²)")
    plt.ylabel(ylabel)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath, dpi=150)
    plt.close()

    return gamma, slope


def plot_sensitivity_bar(betas, outpath):

    params = list(betas.keys())
    vals = [abs(betas[p][0]) for p in params]

    plt.figure(figsize=(7,5))
    plt.bar(params, vals)

    plt.ylabel("Sensitivity (%/(W/m²))")
    plt.title("Irradiance Sensitivity")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(outpath, dpi=200)
    plt.close()

def compute_sensitivity_scores(P_base, params_dict, outdir):

    SS_plus = {}
    SS_minus = {}
    NL = {}

    for pname, pvals in params_dict.items():

        pvals = np.array(pvals)

        # +10% and -10% perturbations
        p_plus = 1.10 * pvals
        p_minus = 0.90 * pvals

        # Recompute power depending on parameter
        if pname == "Vmpp":
            P_plus = p_plus * params_dict["Jmpp"]
            P_minus = p_minus * params_dict["Jmpp"]

        elif pname == "Jmpp":
            P_plus = params_dict["Vmpp"] * p_plus
            P_minus = params_dict["Vmpp"] * p_minus

        elif pname == "FF":
            Voc = params_dict["Voc"]
            Jsc = params_dict["Jsc"]
            P_plus = p_plus * Voc * Jsc
            P_minus = p_minus * Voc * Jsc

        else:
            # Linear approximation for Voc, Jsc, Rs, Rsh
            P_plus = P_base * (p_plus / pvals)
            P_minus = P_base * (p_minus / pvals)

        # ΔP
        dP_plus = P_plus - P_base
        dP_minus = P_minus - P_base

        # Sensitivity Score
        SS_plus[pname] = (np.mean(dP_plus) / np.mean(P_base)) * 100
        SS_minus[pname] = (np.mean(dP_minus) / np.mean(P_base)) * 100

        # Nonlinearity
        NL[pname] = abs(abs(SS_plus[pname]) - abs(SS_minus[pname]))

    # Save CSV
    df = pd.DataFrame({
        "SS_plus": SS_plus,
        "SS_minus": SS_minus,
        "Nonlinearity": NL
    })
    df.to_csv(outdir / "irradiance_sensitivity_scores.csv")

    # Plot sensitivity
    plt.figure(figsize=(10,6))
    plt.bar(df.index, df["SS_plus"], label="SS(+10%)", alpha=0.7)
    plt.bar(df.index, df["SS_minus"], label="SS(-10%)", alpha=0.7)
    plt.xticks(rotation=45)
    plt.ylabel("Sensitivity Score (%)")
    plt.title("Sensitivity Scores (+10% and -10%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "irradiance_sensitivity_scores_bar.png")
    plt.close()

    # Plot nonlinearity
    plt.figure(figsize=(10,6))
    plt.bar(df.index, df["Nonlinearity"], color="purple")
    plt.xticks(rotation=45)
    plt.ylabel("|SS(+)| - |SS(-)|")
    plt.title("Nonlinearity per Parameter")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "irradiance_nonlinearity_scores.png")
    plt.close()

    return df

# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def analyze_irradiance(dataset_path, model_path, tech):

    output_dir = Path(f"outputs/{tech}/irradiance_analysis/diff_rep3")
    reset_dir(output_dir)

    # =====================================================
    # Load data
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
    # Irradiance sweep
    # =====================================================
    irradiances = np.arange(200, 1100, 100)

    K = 50
    rng = np.random.default_rng(42)

    N = len(jv_in)

    vocs_mean, vocs_std = [], []
    jscs_mean, jscs_std = [], []
    vmpps_mean, vmpps_std = [], []
    jmpps_mean, jmpps_std = [], []
    pmpps_mean, pmpps_std = [], []
    ffs_mean, ffs_std = [], []
    Ios_mean, Ios_std = [], []
    Iphs_mean, Iphs_std = [], []
    n_mean, n_std = [], []
    rss_mean, rss_std = [], []
    rshs_mean, rshs_std = [], []

    for G in irradiances:

        idxs = rng.choice(N, size=K, replace=False)

        Voc_list = []
        Jsc_list = []
        Vmpp_list = []
        Jmpp_list = []
        Pmpp_list = []
        FF_list = []
        Rs_list = []
        Rsh_list = []
        Iph_list, Io_list, n_list = [], [], []

        for i in idxs:

            w = weather[i].copy()

            # weather = [Temperature, Irradiance]
            w[1] = G

            jv_t = torch.tensor(jv_in[i], dtype=torch.float32, device=device).unsqueeze(0)
            w_t = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():

                I_pred,I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

            I_pred_np = I_pred.squeeze().cpu().numpy()

            Voc, Jsc, Vmpp, Jmpp, Pmpp, FF = extract_jv_physics(
                V_grid,
                I_pred_np
            )

            diode = diode_pred_norm.squeeze().cpu().numpy()
            diode = diode * diode_stats["std"] + diode_stats["mean"]

            Iph, Io, n, Rs, Rsh = diode

            if Rsh < 0:
                Rsh = -Rsh

            Voc_list.append(Voc)
            Jsc_list.append(Jsc)
            Vmpp_list.append(Vmpp)
            Jmpp_list.append(Jmpp)
            Pmpp_list.append(Pmpp)
            FF_list.append(FF)
            Iph_list.append(Iph)
            Io_list.append(Io)
            n_list.append(n)
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

        Ios_mean.append(np.nanmean(Io_list))
        Ios_std.append(np.nanstd(Io_list))

        Iphs_mean.append(np.nanmean(Iph_list))
        Iphs_std.append(np.nanstd(Iph_list))

        n_mean.append(np.nanmean(n_list))
        n_std.append(np.nanstd(n_list))

        rss_mean.append(np.nanmean(Rs_list))
        rss_std.append(np.nanstd(Rs_list))

        rshs_mean.append(np.nanmean(Rsh_list))
        rshs_std.append(np.nanstd(Rsh_list))

    # =====================================================
    # Save CSV
    # =====================================================
    results = pd.DataFrame({

        "irradiance": irradiances,

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

        "Io_mean": Ios_mean,
        "Io_std": Ios_std,
        "Iph_mean": Iphs_mean,
        "Iph_std": Iphs_std,
        "n_mean": n_mean,
        "n_std": n_std,

        "Rs_mean": rss_mean,
        "Rs_std": rss_std,

        "Rsh_mean": rshs_mean,
        "Rsh_std": rshs_std,

    })

    results.to_csv(output_dir/"irradiance_sweep_mean_std.csv", index=False)

    # =====================================================
    # Plots
    # =====================================================
    # -----------------------------------------------------
    # Sensitivity Score Analysis
    # -----------------------------------------------------
    params_dict = {
        "Voc": vocs_mean,
        "Jsc": jscs_mean,
        "Vmpp": vmpps_mean,
        "Jmpp": jmpps_mean,
        "Pmpp": pmpps_mean,
        "FF": ffs_mean,
        "Rs": rss_mean,
        "Rsh": rshs_mean,
    }

    P_base = np.array(pmpps_mean)

    # compute_sensitivity_scores(P_base, params_dict, output_dir)

    gammas = {}

    gammas["Voc"] = plot_param(irradiances, vocs_mean, vocs_std,
                               "Voc","Voc (V)",output_dir/"Voc_vs_G.png")

    gammas["Jsc"] = plot_param(irradiances, jscs_mean, jscs_std,
                               "Jsc","Jsc (A)",output_dir/"Jsc_vs_G.png")

    gammas["Vmpp"] = plot_param(irradiances, vmpps_mean, vmpps_std,
                                "Vmpp","Vmpp (V)",output_dir/"Vmpp_vs_G.png")

    gammas["Jmpp"] = plot_param(irradiances, jmpps_mean, jmpps_std,
                                "Jmpp","Jmpp (A)",output_dir/"Jmpp_vs_G.png")

    gammas["Pmpp"] = plot_param(irradiances, pmpps_mean, pmpps_std,
                                "Pmpp","Pmpp (W)",output_dir/"Pmpp_vs_G.png")

    gammas["FF"] = plot_param(irradiances, ffs_mean, ffs_std,
                              "FF","Fill Factor",output_dir/"FF_vs_G.png")
    gammas["Io"] = plot_param(irradiances, Ios_mean, Ios_std, "Io", "Io (A)", output_dir / "Io_vs_T.png")
    gammas["Iph"] = plot_param(irradiances, Iphs_mean, Iphs_std, "Iph", "Iph (A)", output_dir / "Iph_vs_T.png")
    gammas["n"] = plot_param(irradiances, n_mean, n_std, "n", "", output_dir / "n_vs_T.png")

    gammas["Rs"] = plot_param(irradiances, rss_mean, rss_std,
                              "Rs","Series Resistance (Ω)",output_dir/"Rs_vs_G.png")

    gammas["Rsh"] = plot_param(irradiances, rshs_mean, rshs_std,
                               "Rsh","Shunt Resistance (Ω)",output_dir/"Rsh_vs_G.png")

    plot_sensitivity_bar(
        gammas,
        output_dir/"irradiance_sensitivity_bar.png"
    )

    pd.DataFrame(
        gammas,
        index=["gamma_%_per_Wm2", "slope"]
    ).T.to_csv(
        output_dir/"irradiance_coefficients.csv"
    )

    print(pd.DataFrame(gammas,
                       index=["gamma_%_per_Wm2","slope"]).T)


if __name__ == "__main__":

    # analyze_irradiance(
    #     "data/jv_autoencoder_dataset_perovskite.npz",
    #     "models/jv_autoencoder_perovskite.pth",
    #     tech="Perovskite_5&6",
    # )

    analyze_irradiance(
        "data/jv_autoencoder_dataset_silicon.npz",
        "models/jv_autoencoder_silicon.pth",
        tech="SunPower_5-1",
    )