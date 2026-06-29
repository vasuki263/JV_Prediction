import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil

from model_lstm import JV_LSTM   # adjust import


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

    gamma = (
            slope /
            np.mean(y)
            * 100
    )

    return gamma, slope


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_param(
    temps,
    values,
    name,
    ylabel,
    outpath
):

    gamma, slope = temp_coeff(temps, values)

    z = np.polyfit(temps, values, 1)
    fit = np.poly1d(z)

    xfit = np.linspace(
        temps.min(),
        temps.max(),
        200
    )

    plt.figure(figsize=(7,5))

    plt.plot(
        temps,
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
        label=f"γ={gamma:.3f}%/°C"
    )

    plt.xlabel("Module Temperature (°C)")
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return gamma, slope

def plot_sensitivity_bar(betas, outpath):
    """
    betas: dict like {"Voc": (beta, slope), "Jsc": (...), ...}
    """
    params = list(betas.keys())
    beta_values = [abs(betas[p][0]) for p in params]  # absolute sensitivity

    plt.figure(figsize=(7,5))
    plt.bar(params, beta_values, color="tab:blue")
    plt.ylabel("Sensitivity gamma_%_per_C)")
    plt.title("Irradiance Sensitivity per Parameter")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def analyze_temperature(
    dataset_path,
    model_path,
    tech
):

    output_dir = Path(f"outputs/{tech}/temperature_analysis")
    reset_dir(output_dir)

    # =====================================================
    # Load dataset
    # =====================================================

    data = np.load(
        dataset_path,
        allow_pickle=True
    )

    X = data["X"]

    scalers = data["scalers"].item()

    # =====================================================
    # Load model
    # =====================================================

    ckpt = torch.load(
        model_path,
        map_location="cpu"
    )

    model = JV_LSTM(
        input_size=10,
        hidden=64,
        layers=2
    )

    model.load_state_dict(
        ckpt["model"]
    )


    model.eval()

    # =====================================================
    # Reference sample
    # =====================================================

    # sample = X[0].copy()
    # sample = np.mean(X, axis=0)
    idx = np.argmin(np.abs(
        scalers["temperature"].inverse_transform(
            X[:, 0, 3].reshape(-1, 1)
        ).flatten() - 30
    ))

    sample = X[idx].copy()

    # =====================================================
    # Temperature sweep
    # =====================================================

    temps = np.arange(20, 75, 5)

    vocs = []
    jscs = []
    ffs = []
    pmpps = []

    for T in temps:

        x = sample.copy()

        T_scaled = scalers["temperature"].transform(
            [[T]]
        )[0,0]

        # feature index 3 = module temperature
        x[:,3] = T_scaled

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            pred = model(x_tensor)

        pred = pred.numpy()

        pred_phys = scalers["y"].inverse_transform(
            pred
        )
        # print(f"T={T}", pred_phys[0])

        Voc, Jsc, Pmpp = pred_phys[0]
        FF = Pmpp / (Voc * Jsc + 1e-12)

        vocs.append(Voc)
        jscs.append(Jsc)
        ffs.append(FF)
        pmpps.append(Pmpp)

    # =====================================================
    # Save predictions
    # =====================================================

    results = pd.DataFrame({
        "temperature": temps,
        "Voc": vocs,
        "Jsc": jscs,
        "FF": ffs,
        "Pmpp": pmpps
    })

    results.to_csv(
        output_dir / "temperature_sweep.csv",
        index=False
    )

    # =====================================================
    # Plots
    # =====================================================

    gammas = {}

    gammas["Voc"] = plot_param(
        temps,
        np.array(vocs),
        "Voc",
        "Voc (V)",
        output_dir / "Voc_vs_T.png"
    )

    gammas["Jsc"] = plot_param(
        temps,
        np.array(jscs),
        "Jsc",
        "Jsc (A)",
        output_dir / "Jsc_vs_T.png"
    )

    gammas["FF"] = plot_param(
        temps,
        np.array(ffs),
        "FF",
        "Fill Factor",
        output_dir / "FF_vs_T.png"
    )

    gammas["Pmpp"] = plot_param(
        temps,
        np.array(pmpps),
        "Pmpp",
        "Pmpp (W)",
        output_dir / "Pmpp_vs_T.png"
    )

    # -----------------------------------------------------
    # Sensitivity bar chart
    # -----------------------------------------------------
    plot_sensitivity_bar(
        gammas,
        output_dir / "temp_sensitivity_bar.png"
    )

    # =====================================================
    # Save coefficients
    # =====================================================

    pd.DataFrame(
        gammas,
        index=["gamma_%_per_C", "slope"]
    ).T.to_csv(
        output_dir / "temperature_coefficients.csv"
    )

    print("\nTemperature coefficients:")
    print(
        pd.DataFrame(
            gammas,
            index=["gamma_%_per_C", "slope"]
        ).T
    )