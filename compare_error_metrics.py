import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Load per-sample metrics CSV and compute mean values
# ---------------------------------------------------------
def load_and_aggregate_metrics(csv_path):
    """
    Loads a CSV of the form:
    MSE,MAE,RMSE,R2
    0.001,0.02,0.03,0.97
    0.0008,0.018,0.028,0.99
    ...

    Returns a dict:
    {
        "MSE": mean_MSE,
        "MAE": mean_MAE,
        "RMSE": mean_RMSE,
        "R²": mean_R2
    }
    """
    df = pd.read_csv(csv_path)

    metrics = {
        "MSE":  df["MSE"].mean(),
        "MAE":  df["MAE"].mean(),
        "RMSE": df["RMSE"].mean(),
        "R²":   df["R2"].mean()
    }

    return metrics


# ---------------------------------------------------------
# Normalize Silicon + Perovskite values to 0–1
# ---------------------------------------------------------
def normalize_pairwise(sil_metrics, per_metrics):
    """
    Normalizes each metric across Silicon + Perovskite:
    norm(x) = x / max(sil, per)
    """
    norm_sil = {}
    norm_per = {}

    for key in ["MSE", "MAE", "RMSE"]:   # R² excluded
        max_val = max(sil_metrics[key], per_metrics[key])

        if max_val == 0:
            norm_sil[key] = 0
            norm_per[key] = 0
        else:
            norm_sil[key] = sil_metrics[key] / max_val
            norm_per[key] = per_metrics[key] / max_val

    return norm_sil, norm_per


# ---------------------------------------------------------
# Plot normalized comparison bar chart
# ---------------------------------------------------------
def plot_comparison_bars(sil_metrics, per_metrics, outpath):

    metrics = list(sil_metrics.keys())
    metrics = [m for m in sil_metrics.keys() if m not in ["R²", "R2"]]

    sil_values = [sil_metrics[m] for m in metrics]
    per_values = [per_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.3

    plt.figure(figsize=(10,5))

    bars1 = plt.bar(
        x - width/2,
        sil_values,
        width,
        label="Silicon",
        color="tab:blue"
    )

    bars2 = plt.bar(
        x + width/2,
        per_values,
        width,
        label="Perovskite",
        color="tab:orange"
    )

    # Annotate Silicon bars
    for bar in bars1:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # Annotate Perovskite bars
    for bar in bars2:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(x, metrics)
    plt.title("Silicon vs Perovskite — Error Metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------------------------------------
# Plot R² separately (raw values)
# ---------------------------------------------------------
def plot_r2_comparison(sil_metrics, per_metrics, outpath):
    r2_sil = sil_metrics["R²"]
    r2_per = per_metrics["R²"]

    labels = ["Silicon", "Perovskite"]
    values = [r2_sil, r2_per]

    plt.figure(figsize=(6, 5))

    bars = plt.bar(
        labels,
        values,
        width=0.2,
        color=["tab:blue", "tab:orange"]
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.4f}",      # 4 decimal places
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.ylabel("R² Score")
    plt.ylim(0, 1.05)   # leave a little room for the labels
    plt.title("R² Comparison — Silicon vs Perovskite")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
def main():

    # Paths to your metric CSVs
    silicon_csv = Path("outputs/SunPower_5-1/ae/metrics/basic_errors.csv")
    perovskite_csv = Path("outputs/Perovskite_5&6/ae/metrics/basic_errors.csv")

    # Load and aggregate metrics
    sil_raw = load_and_aggregate_metrics(silicon_csv)
    per_raw = load_and_aggregate_metrics(perovskite_csv)

    # Normalize MSE, MAE, RMSE
    sil_norm, per_norm = normalize_pairwise(sil_raw, per_raw)

    # Output directory
    outdir = Path("outputs/comparison")
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot normalized metrics
    plot_comparison_bars(
        sil_raw,
        per_raw,
        outdir / "silicon_vs_perovskite_errors.png"
    )

    # Plot R² separately
    plot_r2_comparison(
        sil_raw,
        per_raw,
        outdir / "silicon_vs_perovskite_r2.png"
    )

    print(f"[INFO] Saved comparison charts to {outdir}")


if __name__ == "__main__":
    main()
