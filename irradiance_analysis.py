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
# Irradiance coefficient (same logic as temp_coeff)
# ---------------------------------------------------------
def irr_coeff(x, y):
    slope = np.polyfit(x, y, 1)[0]
    beta = slope * 1000 / np.mean(y)
    # beta = slope / np.mean(y) * 100
    return beta, slope


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------
def plot_param(
    irr,
    values,
    name,
    ylabel,
    outpath
):

    beta, slope = irr_coeff(irr, values)

    z = np.polyfit(irr, values, 1)
    fit = np.poly1d(z)

    xfit = np.linspace(
        irr.min(),
        irr.max(),
        200
    )

    plt.figure(figsize=(7,5))

    plt.plot(
        irr,
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

    plt.xlabel("Irradiance (W/m²)")
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return beta, slope


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def analyze_irradiance(
    dataset_path,
    model_path,
    tech
):

    output_dir = Path(f"outputs/{tech}/irradiance_analysis")
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

    sample = np.mean(X, axis=0)

    # =====================================================
    # Irradiance sweep
    # =====================================================

    # irr_values = np.linspace(
    #     scalers["irradiance"].data_min_[0],
    #     scalers["irradiance"].data_max_[0],
    #     20
    # )
    irr_values = np.arange(200, 1200, 50)  # W/m²

    vocs = []
    jscs = []
    ffs = []
    pmpps = []

    for G in irr_values:

        x = sample.copy()

        G_scaled = scalers["irradiance"].transform(
            [[G]]
        )[0,0]


        x[:,4] = G_scaled

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            pred = model(x_tensor)

        pred = pred.numpy()

        pred_phys = scalers["y"].inverse_transform(pred)

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
        "irradiance": irr_values,
        "Voc": vocs,
        "Jsc": jscs,
        "FF": ffs,
        "Pmpp": pmpps
    })

    results.to_csv(
        output_dir / "irradiance_sweep.csv",
        index=False
    )

    # =====================================================
    # Plots
    # =====================================================

    betas = {}

    betas["Voc"] = plot_param(
        irr_values,
        np.array(vocs),
        "Voc",
        "Voc (V)",
        output_dir / "Voc_vs_irr.png"
    )

    betas["Jsc"] = plot_param(
        irr_values,
        np.array(jscs),
        "Jsc",
        "Jsc (A)",
        output_dir / "Jsc_vs_irr.png"
    )

    betas["FF"] = plot_param(
        irr_values,
        np.array(ffs),
        "FF",
        "Fill Factor",
        output_dir / "FF_vs_irr.png"
    )

    betas["Pmpp"] = plot_param(
        irr_values,
        np.array(pmpps),
        "Pmpp",
        "Pmpp (W)",
        output_dir / "Pmpp_vs_irr.png"
    )

    # =====================================================
    # Save coefficients
    # =====================================================

    pd.DataFrame(
        betas,
        index=["beta_%_per_Wm2", "slope"]
    ).T.to_csv(
        output_dir / "irradiance_coefficients.csv"
    )

    print("\nIrradiance coefficients:")
    print(
        pd.DataFrame(
            betas,
            index=["beta_%_per_Wm2", "slope"]
        ).T
    )
