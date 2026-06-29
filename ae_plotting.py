import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_error_histogram(mse, mae, rmse, r2, out_dir: Path):
    plt.figure(figsize=(7, 5))
    plt.hist(mse, bins=40, alpha=0.6, label="MSE")
    plt.hist(mae, bins=40, alpha=0.6, label="MAE")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.title("AE Reconstruction Error Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "error_histogram.png", dpi=150)
    plt.close()


# ---------------------------------------------------------
# Voltage‑resolved error
# ---------------------------------------------------------
def plot_voltage_error(V_grid, preds, targets, out_dir: Path):
    abs_err = np.abs(preds - targets)
    mean_err = abs_err.mean(axis=0)

    plt.figure(figsize=(7, 5))
    plt.plot(V_grid, mean_err, linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Mean Absolute Error (A)")
    plt.title("Voltage‑Resolved Reconstruction Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_voltage.png", dpi=150)
    plt.close()


def plot_pred_vs_actual(preds, targets, out_dir: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets.flatten(), preds.flatten(), s=3, alpha=0.4)

    lim_min = min(targets.min(), preds.min())
    lim_max = max(targets.max(), preds.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=2)

    plt.xlabel("Actual Current (A)")
    plt.ylabel("Predicted Current (A)")
    plt.title("Predicted vs Actual JV Points")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_actual_scatter.png", dpi=150)
    plt.close()


def plot_reconstructed_curves(V_grid, jv_in, jv_tgt, preds, out_dir):
    """
    If jv_in or jv_tgt is None, only plot smooth predicted curves.
    """

    import matplotlib.pyplot as plt

    for idx in range(len(preds)):
        plt.figure(figsize=(6,4))

        # Plot measured curves only if provided
        if jv_in is not None:
            plt.plot(V_grid, jv_in[idx], label="Measured JV (raw)", alpha=0.5)

        if jv_tgt is not None:
            plt.plot(V_grid, jv_tgt[idx], label="Target JV (raw)", alpha=0.5)

        # Always plot predicted smooth curve
        plt.plot(V_grid, preds[idx], label="Predicted JV (smooth)", linewidth=2)

        plt.xlabel("Voltage (V)")
        plt.ylabel("Current density (A/m²)")
        plt.legend()
        plt.grid(True)

        plt.savefig(out_dir / f"curve_{idx}.png")
        plt.close()



# ---------------------------------------------------------
# Physics‑parameter accuracy
# ---------------------------------------------------------
def plot_physics_error(Voc_t, Voc_p, Jsc_t, Jsc_p, Pmpp_t, Pmpp_p, FF_t, FF_p, out_dir: Path):
    metrics = [
        ("Voc", Voc_t, Voc_p),
        ("Jsc", Jsc_t, Jsc_p),
        ("Pmpp", Pmpp_t, Pmpp_p),
        ("FF", FF_t, FF_p),
    ]

    for name, t, p in metrics:
        plt.figure(figsize=(6, 6))
        plt.scatter(t, p, alpha=0.6)

        lim_min = min(t.min(), p.min())
        lim_max = max(t.max(), p.max())
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--")

        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"True vs Predicted {name}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"Errors_{name}_true_vs_pred.png", dpi=150)
        plt.close()


def plot_error_heatmap(preds, targets, out_dir: Path):
    err = np.abs(preds - targets)

    plt.figure(figsize=(8, 6))
    plt.imshow(err, aspect="auto", cmap="viridis")
    plt.colorbar(label="Absolute Error (A)")
    plt.xlabel("Voltage Index")
    plt.ylabel("Sample Index")
    plt.title("Error Heatmap (Samples × Voltage)")
    plt.tight_layout()
    plt.savefig(out_dir / "error_heatmap.png", dpi=150)
    plt.close()


def plot_error_vs_physics_params(errors, params, out_dir: Path):
    for key, values in params.items():
        plt.figure(figsize=(6, 5))
        plt.scatter(values, errors, alpha=0.6)

        plt.xlabel(key)
        plt.ylabel("AE Error (MSE)")
        plt.title(f"AE Error vs {key}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"error_vs_{key}.png", dpi=150)
        plt.close()

def plot_param(true_vals, pred_vals, name, save_dir):
    plt.figure(figsize=(6,6))
    plt.scatter(true_vals, pred_vals, alpha=0.6)
    plt.plot([min(true_vals), max(true_vals)],
             [min(true_vals), max(true_vals)],
             'r--', label="Ideal")
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs Predicted {name}")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_dir / f"{name}_true_vs_pred.png")
    plt.close()