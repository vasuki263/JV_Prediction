# evaluate_lstm.py

import os
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_lstm import JV_LSTM
from build_dataset import build_dataset, nearest_timestamp


# ---------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# JV reconstruction model
# ---------------------------------------------------------
def reconstruct_jv_curve(Voc, Jsc, Pmpp, num_points=100):
    Voc = max(Voc, 1e-6)
    Jsc = max(Jsc, 1e-6)

    V = np.linspace(0, Voc, num_points)
    FF = np.clip(Pmpp / (Voc * Jsc + 1e-12), 0.01, 0.85)

    Vmpp = FF * Voc
    Impp = FF * Jsc

    ratio1 = np.clip(1 - Impp / (Jsc + 1e-9), 1e-6, 1)
    ratio2 = np.clip(Vmpp / (Voc + 1e-9), 1e-6, 0.999999)

    a = np.log(ratio1) / np.log(ratio2)
    a = np.clip(a, 1.2, 10)

    J = Jsc * (1 - (V / (Voc + 1e-9))**a)
    return V, J


# ---------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------
def plot_true_vs_pred(true, pred, name, base_plot_dir):
    out_dir = os.path.join(base_plot_dir, "true_vs_pred")
    ensure_dir(out_dir)

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, edgecolor="k")

    lim_min = min(true.min(), pred.min())
    lim_max = max(true.max(), pred.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=2)

    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs Predicted {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
    plt.close()


def plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx, base_plot_dir):
    out_dir = os.path.join(base_plot_dir, "jv_curves")
    ensure_dir(out_dir)

    plt.figure(figsize=(6, 5))
    plt.plot(V_true, J_true, label="True JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current Density (A)")
    plt.title(f"Reconstructed JV #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"reconstructed_{idx}.png"), dpi=150)
    plt.close()


def plot_raw_vs_pred(V_raw, I_raw, V_pred, J_pred, idx, base_plot_dir):
    out_dir = os.path.join(base_plot_dir, "jv_curves")
    ensure_dir(out_dir)

    plt.figure(figsize=(6, 5))
    plt.plot(V_raw, I_raw, label="Raw JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current Density (A)")
    plt.title(f"Raw vs Predicted JV #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"raw_vs_pred_{idx}.png"), dpi=150)
    plt.close()


def plot_dependency(x, y, xlabel, ylabel, base_plot_dir, subdir="rs_rsh_effects"):
    out_dir = os.path.join(base_plot_dir, subdir)
    ensure_dir(out_dir)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.6, edgecolor="k")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{ylabel}_vs_{xlabel}.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------
def evaluate(tech, weather_csv, model_path="jv_model.pth"):

    device = tech
    base_plot_dir = f"outputs/{device}/plots"
    base_metric_dir = f"outputs/{device}/metrics"

    reset_dir(base_plot_dir)
    reset_dir(base_metric_dir)

    # Load dataset
    _, _, test_loader, scalers, raw_test = build_dataset(
        tech,
        clean_weather_path=weather_csv
    )

    # Load model
    ckpt = torch.load(model_path, map_location="cpu")
    model = JV_LSTM(input_size=10)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for X, Y in test_loader:
            preds.append(model(X).cpu().numpy())
            trues.append(Y.cpu().numpy())

    preds = scalers["y"].inverse_transform(np.vstack(preds))
    trues = scalers["y"].inverse_transform(np.vstack(trues))

    # Errors
    err_Voc = np.abs(trues[:, 0] - preds[:, 0])
    err_Jsc = np.abs(trues[:, 1] - preds[:, 1])
    err_Pmpp = np.abs(trues[:, 2] - preds[:, 2])

    FF_true = trues[:, 2] / (trues[:, 0] * trues[:, 1] + 1e-12)
    FF_pred = preds[:, 2] / (preds[:, 0] * preds[:, 1] + 1e-12)

    # Save predictions
    df = pd.DataFrame({
        "Voc_true": trues[:, 0], "Voc_pred": preds[:, 0],
        "Jsc_true": trues[:, 1], "Jsc_pred": preds[:, 1],
        "FF_true": FF_true,
        "FF_pred": FF_pred,
        "Pmpp_true": trues[:, 2], "Pmpp_pred": preds[:, 2],
        "err_Voc": err_Voc, "err_Jsc": err_Jsc, "err_Pmpp": err_Pmpp
    })

    df.to_csv(os.path.join(base_metric_dir, "predictions_vs_true.csv"), index=False)

    # True vs Predicted plots
    plot_true_vs_pred(trues[:, 0], preds[:, 0], "Voc", base_plot_dir)
    plot_true_vs_pred(trues[:, 1], preds[:, 1], "Jsc", base_plot_dir)
    plot_true_vs_pred(trues[:, 2], preds[:, 2], "Pmpp", base_plot_dir)
    plot_true_vs_pred(FF_true, FF_pred, "FF", base_plot_dir)

    # JV curve plots
    raw_V = [entry["voltage_raw"] for entry in raw_test]
    raw_I = [entry["current_raw"] for entry in raw_test]

    for idx in range(min(5, len(trues))):
        V_true, J_true = reconstruct_jv_curve(*trues[idx])
        V_pred, J_pred = reconstruct_jv_curve(*preds[idx])

        plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx, base_plot_dir)
        plot_raw_vs_pred(raw_V[idx], raw_I[idx], V_pred, J_pred, idx, base_plot_dir)

    # ---------------------------------------------------------
    # RS / RSH effects
    # ---------------------------------------------------------
    Rs_vals  = np.array([e["Rs"]  for e in raw_test])
    Rsh_vals = np.array([e["Rsh"] for e in raw_test])

    Voc_true = trues[:, 0]
    Jsc_true = trues[:, 1]
    Pmpp_true = trues[:, 2]
    FF_true = Pmpp_true / (Voc_true * Jsc_true + 1e-9)

    for x, name in [(Rs_vals, "Rs"), (Rsh_vals, "Rsh")]:
        plot_dependency(x, Voc_true, name, "Voc", base_plot_dir)
        plot_dependency(x, Jsc_true, name, "Jsc", base_plot_dir)
        plot_dependency(x, FF_true, name, "FF", base_plot_dir)
        plot_dependency(x, Pmpp_true, name, "Pmpp", base_plot_dir)

    # ---------------------------------------------------------
    # Two‑diode parameter effects
    # ---------------------------------------------------------
    n1_vals  = np.array([e["n1"]  for e in raw_test])
    n2_vals  = np.array([e["n2"]  for e in raw_test])
    J01_vals = np.array([e["J01"] for e in raw_test])
    J02_vals = np.array([e["J02"] for e in raw_test])
    Jph_vals = np.array([e["Jph"] for e in raw_test])

    plot_dependency(n1_vals, Voc_true, "n1", "Voc", base_plot_dir, subdir="two_diode")
    plot_dependency(n2_vals, Voc_true, "n2", "Voc", base_plot_dir, subdir="two_diode")
    plot_dependency(J02_vals, FF_true, "J02", "FF", base_plot_dir, subdir="two_diode")
    plot_dependency(J01_vals, Voc_true, "J01", "Voc", base_plot_dir, subdir="two_diode")
    plot_dependency(Jph_vals, Jsc_true, "Jph", "Jsc", base_plot_dir, subdir="two_diode")

    print("Evaluation complete.")
    return preds, trues
