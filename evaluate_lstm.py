# evaluate_lstm.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_lstm import JV_LSTM
from build_dataset import build_dataset
from hysteresis_analysis import analyze_hysteresis, correlate_hysteresis_with_weather
import os

# ---------------------------------------------------------
# DIRECTORY HELPERS
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

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
    V = np.linspace(0, Voc, num_points)

    FF = Pmpp / (Voc * Jsc)
    FF = np.clip(FF, 0.01, 0.85)

    Vmpp = FF * Voc
    Impp = FF * Jsc

    ratio1 = np.clip(1 - Impp / Jsc, 1e-6, 1)
    ratio2 = np.clip(Vmpp / Voc, 1e-6, 0.999999)

    a = np.log(ratio1) / np.log(ratio2)
    a = np.clip(a, 1.2, 10)

    J = Jsc * (1 - (V / Voc)**a)
    return V, J


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    return mae, mse, rmse, r2


# ---------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------
def plot_true_vs_pred(true, pred, name, base_plot_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, edgecolor='k')
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', linewidth=2)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs Predicted {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    ensure_dir(f"{base_plot_dir}/true_vs_pred")
    plt.savefig(f"{base_plot_dir}/true_vs_pred/plot_true_vs_pred_{name}.png", dpi=150)
    plt.close()


def plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx, base_plot_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(V_true, J_true, label="True Reconstructed JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Reconstructed JV Comparison #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ensure_dir(f"{base_plot_dir}/jv_curves")
    plt.savefig(f"{base_plot_dir}/jv_curves/reconstructed_jv_comparison_{idx}.png", dpi=150)
    plt.close()


def plot_raw_vs_pred(V_raw, I_raw, V_pred, J_pred, idx, base_plot_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(V_raw, I_raw, label="Raw JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Raw JV vs Predicted JV #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ensure_dir(f"{base_plot_dir}/jv_curves")
    plt.savefig(f"{base_plot_dir}/jv_curves/raw_vs_pred_jv_{idx}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate(directory):

    # Detect device type
    device = get_device_type(directory)
    base_plot_dir = f"outputs/{device}/plots"
    base_metric_dir = f"outputs/{device}/metrics"

    ensure_dir(base_plot_dir)
    ensure_dir(base_metric_dir)

    # Load dataset
    _, _, test_loader, scalers, raw_test = build_dataset(directory)

    ckpt = torch.load("jv_model.pth")
    model = JV_LSTM()
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for X, Y in test_loader:
            preds.append(model(X).numpy())
            trues.append(Y.numpy())

    preds = scalers["targets"].inverse_transform(np.vstack(preds))
    trues = scalers["targets"].inverse_transform(np.vstack(trues))

    # Compute FF
    FF_true = trues[:, 2] / (trues[:, 0] * trues[:, 1])
    FF_pred = preds[:, 2] / (preds[:, 0] * preds[:, 1])

    # ---------------------------------------------------------
    # SAVE METRICS CSV
    # ---------------------------------------------------------
    df = pd.DataFrame({
        "Voc_true": trues[:, 0],
        "Voc_pred": preds[:, 0],
        "Jsc_true": trues[:, 1],
        "Jsc_pred": preds[:, 1],
        "Pmpp_true": trues[:, 2],
        "Pmpp_pred": preds[:, 2],
        "FF_true": FF_true,
        "FF_pred": FF_pred
    })

    df.to_csv(f"{base_metric_dir}/predictions_vs_true.csv", index=False)
    print("\nSaved predictions_vs_true.csv")

    # ---------------------------------------------------------
    # TRUE VS PREDICTED PLOTS
    # ---------------------------------------------------------
    plot_true_vs_pred(trues[:, 0], preds[:, 0], "Voc", base_plot_dir)
    plot_true_vs_pred(trues[:, 1], preds[:, 1], "Jsc", base_plot_dir)
    plot_true_vs_pred(trues[:, 2], preds[:, 2], "Pmpp", base_plot_dir)
    plot_true_vs_pred(FF_true, FF_pred, "FF", base_plot_dir)

    print("Saved true vs predicted plots.")

    # ---------------------------------------------------------
    # JV CURVE PLOTS
    # ---------------------------------------------------------
    print("\nGenerating JV curve plots...")

    raw_V = [entry["voltage_raw"] for entry in raw_test]
    raw_I = [entry["current_raw"] for entry in raw_test]

    num_examples = min(5, len(trues))

    for idx in range(num_examples):
        Voc_t, Jsc_t, Pmpp_t = trues[idx]
        Voc_p, Jsc_p, Pmpp_p = preds[idx]

        V_true, J_true = reconstruct_jv_curve(Voc_t, Jsc_t, Pmpp_t)
        V_pred, J_pred = reconstruct_jv_curve(Voc_p, Jsc_p, Pmpp_p)

        V_raw = raw_V[idx]
        I_raw = raw_I[idx]

        plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx, base_plot_dir)
        plot_raw_vs_pred(V_raw, I_raw, V_pred, J_pred, idx, base_plot_dir)

    print("Saved reconstructed and raw JV curve plots.")

    # ---------------------------------------------------------
    # HYSTERESIS ANALYSIS
    # ---------------------------------------------------------
    print("\nRunning hysteresis analysis...")

    hyst_results = analyze_hysteresis(raw_test, preds, device)
    pd.DataFrame(hyst_results).to_csv(f"{base_metric_dir}/hysteresis_metrics.csv", index=False)
    print("Saved hysteresis_metrics.csv")

    # ---------------------------------------------------------
    # WEATHER CORRELATION
    # ---------------------------------------------------------
    weather = pd.read_csv("weather_jv_cleaned.csv", parse_dates=["timestamp"])
    weather = weather.set_index("timestamp").sort_index()

    corr = correlate_hysteresis_with_weather(hyst_results, weather, device)
    print("\nWeather correlation matrix:")
    print(corr)


if __name__ == "__main__":
    evaluate("C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Si_raw")
