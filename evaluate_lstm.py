# evaluate_lstm.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_lstm import JV_LSTM
from build_dataset import build_dataset


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true-y_pred))
    mse = np.mean((y_true-y_pred)**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true-np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot
    return mae, mse, rmse, r2


def plot_true_vs_pred(true, pred, name):
    plt.figure(figsize=(6,6))
    plt.scatter(true, pred, alpha=0.6, edgecolor='k')
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', linewidth=2)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs Predicted {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plot_true_vs_pred_{name}.png", dpi=150)
    plt.close()


def evaluate(directory):
    loader, scalers = build_dataset(directory)
    ckpt = torch.load("jv_model.pth")
    model = JV_LSTM()
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for X, Y in loader:
            p = model(X).numpy()
            preds.append(p)
            trues.append(Y.numpy())

    preds = scalers["targets"].inverse_transform(np.vstack(preds))
    trues = scalers["targets"].inverse_transform(np.vstack(trues))

    # Compute FF analytically
    FF_true = trues[:,2] / (trues[:,0] * trues[:,1])
    FF_pred = preds[:,2] / (preds[:,0] * preds[:,1])

    # Print metrics
    names = ["Voc", "Jsc", "Pmpp"]
    for i, name in enumerate(names):
        mae, mse, rmse, r2 = metrics(trues[:,i], preds[:,i])
        print(f"\n{name}:")
        print(f" MAE  = {mae:.4f}")
        print(f" MSE  = {mse:.4f}")
        print(f" RMSE = {rmse:.4f}")
        print(f" R²   = {r2:.4f}")

    mae, mse, rmse, r2 = metrics(FF_true, FF_pred)
    print("\nFF (computed):")
    print(f" MAE  = {mae:.4f}")
    print(f" MSE  = {mse:.4f}")
    print(f" RMSE = {rmse:.4f}")
    print(f" R²   = {r2:.4f}")

    # ---------------------------------------------------------
    # SAVE CSV WITH TRUE VS PREDICTED VALUES
    # ---------------------------------------------------------
    df = pd.DataFrame({
        "Voc_true": trues[:,0],
        "Voc_pred": preds[:,0],
        "Jsc_true": trues[:,1],
        "Jsc_pred": preds[:,1],
        "Pmpp_true": trues[:,2],
        "Pmpp_pred": preds[:,2],
        "FF_true": FF_true,
        "FF_pred": FF_pred
    })

    df.to_csv("predictions_vs_true.csv", index=False)
    print("\n📁 CSV saved: predictions_vs_true.csv")

    # ---------------------------------------------------------
    # GENERATE TRUE VS PREDICTED PLOTS
    # ---------------------------------------------------------
    plot_true_vs_pred(trues[:,0], preds[:,0], "Voc")
    plot_true_vs_pred(trues[:,1], preds[:,1], "Jsc")
    plot_true_vs_pred(trues[:,2], preds[:,2], "Pmpp")
    plot_true_vs_pred(FF_true, FF_pred, "FF")

    print("📊 Plots saved: plot_true_vs_pred_*.png")


if __name__ == "__main__":
    evaluate("C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw")
