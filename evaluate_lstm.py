# evaluate_lstm.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_lstm import JV_LSTM
from build_dataset import build_dataset


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
def plot_true_vs_pred(true, pred, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, edgecolor='k')
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', linewidth=2)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs Predicted {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plot_true_vs_pred_{name}.png", dpi=150)
    plt.close()


def plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx):
    plt.figure(figsize=(6, 5))
    plt.plot(V_true, J_true, label="True Reconstructed JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Reconstructed JV Comparison #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reconstructed_jv_comparison_{idx}.png", dpi=150)
    plt.close()


def plot_raw_vs_pred(V_raw, I_raw, V_pred, J_pred, idx):
    plt.figure(figsize=(6, 5))
    plt.plot(V_raw, I_raw, label="Raw JV", linewidth=2)
    plt.plot(V_pred, J_pred, "--", label="Predicted JV", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Raw JV vs Predicted JV #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"raw_vs_pred_jv_{idx}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate(directory):

    # IMPORTANT: use test_loader + raw curves
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
    # PRINT METRICS
    # ---------------------------------------------------------
    names = ["Voc", "Jsc", "Pmpp"]
    for i, name in enumerate(names):
        mae, mse, rmse, r2 = metrics(trues[:, i], preds[:, i])
        print(f"\n{name}:")
        print(f" MAE  = {mae:.4f}")
        print(f" MSE  = {mse:.4f}")
        print(f" RMSE = {rmse:.4f}")
        print(f" R²   = {r2:.4f}")

    mae, mse, rmse, r2 = metrics(FF_true, FF_pred)
    print("\nFF:")
    print(f" MAE  = {mae:.4f}")
    print(f" MSE  = {mse:.4f}")
    print(f" RMSE = {rmse:.4f}")
    print(f" R²   = {r2:.4f}")

    # ---------------------------------------------------------
    # SAVE CSV
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
    df.to_csv("predictions_vs_true.csv", index=False)
    print("\nSaved predictions_vs_true.csv")

    # ---------------------------------------------------------
    # TRUE VS PREDICTED PLOTS
    # ---------------------------------------------------------
    plot_true_vs_pred(trues[:, 0], preds[:, 0], "Voc")
    plot_true_vs_pred(trues[:, 1], preds[:, 1], "Jsc")
    plot_true_vs_pred(trues[:, 2], preds[:, 2], "Pmpp")
    plot_true_vs_pred(FF_true, FF_pred, "FF")

    print("Saved true vs predicted plots.")

    # ---------------------------------------------------------
    # JV CURVE PLOTS (RAW + RECONSTRUCTED)
    # ---------------------------------------------------------
    print("\nGenerating JV curve plots...")

    raw_V = [v for v, i in raw_test]
    raw_I = [i for v, i in raw_test]

    num_examples = min(5, len(trues))

    for idx in range(num_examples):
        Voc_t, Jsc_t, Pmpp_t = trues[idx]
        Voc_p, Jsc_p, Pmpp_p = preds[idx]

        # Reconstructed true JV
        V_true, J_true = reconstruct_jv_curve(Voc_t, Jsc_t, Pmpp_t)
        # Predicted JV
        V_pred, J_pred = reconstruct_jv_curve(Voc_p, Jsc_p, Pmpp_p)

        # Raw JV
        V_raw = raw_V[idx]
        I_raw = raw_I[idx]

        # Save both plots
        plot_reconstructed_jv(V_true, J_true, V_pred, J_pred, idx)
        plot_raw_vs_pred(V_raw, I_raw, V_pred, J_pred, idx)

    print("Saved reconstructed and raw JV curve plots.")


if __name__ == "__main__":
    evaluate("C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw")
