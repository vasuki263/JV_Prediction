# evaluate_lstm.py — FINAL VERSION (WEATHER‑AWARE, FULL JV PLOTTING)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_lstm import JV_LSTM
from build_dataset import build_dataset, nearest_timestamp
import os
import shutil


# ---------------------------------------------------------
# DIRECTORY HELPERS
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# DEVICE TYPE DETECTION
# ---------------------------------------------------------
def get_device_type(directory):
    name = directory.lower()
    if "psk" in name or "perov" in name:
        return "Perovskite"
    if "si" in name or "silicon" in name:
        return "Silicon"
    return "Unknown"


# ---------------------------------------------------------
# JV RECONSTRUCTION MODEL
# ---------------------------------------------------------
def reconstruct_jv_curve(Voc, Jsc, Pmpp, num_points=100):
    Voc = max(Voc, 1e-6)
    Jsc = max(Jsc, 1e-6)

    V = np.linspace(0, Voc, num_points)
    FF = np.clip(Pmpp / (Voc * Jsc + 1e-9), 0.01, 0.85)

    Vmpp = FF * Voc
    Impp = FF * Jsc

    ratio1 = np.clip(1 - Impp / (Jsc + 1e-9), 1e-6, 1)
    ratio2 = np.clip(Vmpp / (Voc + 1e-9), 1e-6, 0.999999)

    a = np.log(ratio1) / np.log(ratio2)
    a = np.clip(a, 1.2, 10)

    J = Jsc * (1 - (V / (Voc + 1e-9))**a)
    return V, J


# ---------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------
def plot_true_vs_pred(true, pred, name, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/true_vs_pred")

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, edgecolor='k')
    lim_min = min(true.min(), pred.min())
    lim_max = max(true.max(), pred.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs Predicted {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/true_vs_pred/{name}.png", dpi=150)
    plt.close()


def plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/jv_curves")

    plt.figure(figsize=(6, 5))
    plt.plot(V_true, J_true, label="True JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current Density (A)")
    plt.title(f"Reconstructed JV #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/jv_curves/reconstructed_{idx}.png", dpi=150)
    plt.close()


def plot_raw_vs_pred(V_raw, I_raw, V_pred, J_pred, idx, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/jv_curves")

    plt.figure(figsize=(6, 5))
    plt.plot(V_raw, I_raw, label="Raw JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current Density (A)")
    plt.title(f"Raw vs Predicted JV #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/jv_curves/raw_vs_pred_{idx}.png", dpi=150)
    plt.close()


def plot_error_vs_weather(errors, weather_values, name, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/weather_plots")

    plt.figure(figsize=(6,5))
    plt.scatter(weather_values, errors, alpha=0.6, edgecolor='k')
    plt.xlabel(name)
    plt.ylabel("Absolute Error")
    plt.title(f"Error vs {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/weather_plots/error_vs_{name}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate(directory,
             weather_csv,
             model_path="jv_model.pth"):

    device = get_device_type(directory)

    base_plot_dir = f"outputs/{device}/plots"
    base_metric_dir = f"outputs/{device}/metrics"

    reset_dir(base_plot_dir)
    reset_dir(base_metric_dir)

    # Load dataset
    _, _, test_loader, scalers, raw_test = build_dataset(
        directory,
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
            out = model(X)
            preds.append(out.cpu().numpy())
            trues.append(Y.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    preds = scalers["targets"].inverse_transform(preds)
    trues = scalers["targets"].inverse_transform(trues)

    # Compute errors
    err_Voc = np.abs(trues[:,0] - preds[:,0])
    err_Jsc = np.abs(trues[:,1] - preds[:,1])
    err_Pmpp = np.abs(trues[:,2] - preds[:,2])

    # Save predictions
    df = pd.DataFrame({
        "Voc_true": trues[:, 0],
        "Voc_pred": preds[:, 0],
        "Jsc_true": trues[:, 1],
        "Jsc_pred": preds[:, 1],
        "Pmpp_true": trues[:, 2],
        "Pmpp_pred": preds[:, 2],
        "err_Voc": err_Voc,
        "err_Jsc": err_Jsc,
        "err_Pmpp": err_Pmpp
    })
    df.to_csv(f"{base_metric_dir}/predictions_vs_true.csv", index=False)

    # ---------------------------------------------------------
    # TRUE VS PREDICTED PLOTS
    # ---------------------------------------------------------
    plot_true_vs_pred(trues[:, 0], preds[:, 0], "Voc", base_plot_dir)
    plot_true_vs_pred(trues[:, 1], preds[:, 1], "Jsc", base_plot_dir)
    plot_true_vs_pred(trues[:, 2], preds[:, 2], "Pmpp", base_plot_dir)

    FF_true = trues[:, 2] / (trues[:, 0] * trues[:, 1] + 1e-9)
    FF_pred = preds[:, 2] / (preds[:, 0] * preds[:, 1] + 1e-9)
    plot_true_vs_pred(FF_true, FF_pred, "FF", base_plot_dir)

    # ---------------------------------------------------------
    # JV CURVE PLOTS
    # ---------------------------------------------------------
    raw_V = [entry["voltage_raw"] for entry in raw_test]
    raw_I = [entry["current_raw"] for entry in raw_test]

    for idx in range(min(5, len(trues))):
        V_true, J_true = reconstruct_jv_curve(*trues[idx])
        V_pred, J_pred = reconstruct_jv_curve(*preds[idx])

        plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx, base_plot_dir)
        plot_raw_vs_pred(raw_V[idx], raw_I[idx], V_pred, J_pred, idx, base_plot_dir)

    # ---------------------------------------------------------
    # WEATHER ANALYSIS
    # ---------------------------------------------------------
    weather = pd.read_csv(weather_csv, parse_dates=["timestamp"])
    weather = weather.set_index("timestamp").sort_index()

    weather_vals = []
    for entry in raw_test:
        ts = pd.Timestamp(entry["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        w = nearest_timestamp(weather, ts)
        weather_vals.append([
            w["temp_C"], w["humidity"], w["cloud_cover"],
            w["precip_mm"], w["global_radiation"]
        ])

    weather_vals = np.array(weather_vals)

    # Weather‑error plots
    plot_error_vs_weather(err_Pmpp, weather_vals[:,4], "Global Radiation", base_plot_dir)
    plot_error_vs_weather(err_Pmpp, weather_vals[:,0], "Ambient Temperature", base_plot_dir)
    plot_error_vs_weather(err_Pmpp, weather_vals[:,1], "Humidity", base_plot_dir)
    plot_error_vs_weather(err_Pmpp, weather_vals[:,2], "Cloud Cover", base_plot_dir)
    plot_error_vs_weather(err_Pmpp, weather_vals[:,3], "Precipitation", base_plot_dir)

    # Weather‑error correlations
    weather_metrics = {
        "corr_errPmpp_temp": np.corrcoef(err_Pmpp, weather_vals[:,0])[0,1],
        "corr_errPmpp_humidity": np.corrcoef(err_Pmpp, weather_vals[:,1])[0,1],
        "corr_errPmpp_cloud": np.corrcoef(err_Pmpp, weather_vals[:,2])[0,1],
        "corr_errPmpp_precip": np.corrcoef(err_Pmpp, weather_vals[:,3])[0,1],
        "corr_errPmpp_rad": np.corrcoef(err_Pmpp, weather_vals[:,4])[0,1],
    }

    pd.DataFrame([weather_metrics]).to_csv(
        f"{base_metric_dir}/weather_error_correlations.csv",
        index=False
    )

    print("Evaluation complete.")
    return preds, trues
