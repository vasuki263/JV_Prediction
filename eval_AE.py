import os
import shutil
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_autoencoder import PhysicsInformedJVModel


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Robust Voc extraction
# ---------------------------------------------------------
def extract_voc(V, I):
    V = np.asarray(V, float)
    I = np.asarray(I, float)

    # If all currents are negative → Voc near highest V
    if np.all(I < 0):
        k = np.argmin(np.abs(I))
        return V[k]

    # If all currents are positive → Voc near lowest V
    if np.all(I > 0):
        k = np.argmin(np.abs(I))
        return V[k]

    # Look for zero crossing between points
    for k in range(len(I) - 1):
        if (I[k] <= 0 and I[k+1] >= 0) or (I[k] >= 0 and I[k+1] <= 0):
            return np.interp(0.0, [I[k], I[k+1]], [V[k], V[k+1]])

    # Fallback: nearest-to-zero current
    k = np.argmin(np.abs(I))
    return V[k]


# ---------------------------------------------------------
# JV physics extraction
# ---------------------------------------------------------
def extract_jv_physics(V, I):
    V = np.asarray(V, float)
    I = np.asarray(I, float)

    mask = np.isfinite(V) & np.isfinite(I)
    V = V[mask]
    I = I[mask]

    if len(V) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    idx = np.argsort(V)
    V = V[idx]
    I = I[idx]

    # Jsc
    Jsc = np.interp(0.0, V, I)

    # Voc
    Voc = extract_voc(V, I)

    # Use only V >= 0 region
    mask_vpos = V >= 0
    V_pos = V[mask_vpos]
    I_pos = I[mask_vpos]

    if len(V_pos) == 0:
        return Voc, Jsc, np.nan, np.nan, np.nan, np.nan

    # Power (absolute, since current is often negative)
    P = V_pos * I_pos
    P_abs = np.abs(P)

    k = np.argmax(P_abs)

    Vmpp = V_pos[k]
    Jmpp = I_pos[k]
    Pmpp = P_abs[k]

    # FF = Pmpp / (np.abs(Voc) * np.abs(Jsc) + 1e-12)
    denom = Voc * Jsc

    if (
            np.isnan(Voc)
            or np.isnan(Jsc)
            or abs(denom) < 1e-8
    ):
        FF = np.nan
    else:
        FF = Pmpp / denom

    return Voc, Jsc, Vmpp, Jmpp, Pmpp, FF

# ---------------------------------------------------------
# Unnormalize diode parameters
# ---------------------------------------------------------
def unnormalize_params(params_norm, stats):
    return params_norm * stats["std"] + stats["mean"]


# ---------------------------------------------------------
# Error metrics
# ---------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_error_metrics(true, pred):
    """
    Computes regression metrics while ignoring NaNs.
    """

    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(true) & np.isfinite(pred)

    true = true[mask]
    pred = pred[mask]

    if len(true) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "NRMSE": np.nan,
            "R2": np.nan,
            "PearsonR": np.nan,
        }

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    if len(true) > 1:
        r2 = r2_score(true, pred)
        pearson = np.corrcoef(true, pred)[0, 1]
    else:
        r2 = np.nan
        pearson = np.nan

    mape = np.mean(
        np.abs((true - pred) / (true + 1e-8))
    ) * 100

    nrmse = rmse / (true.max() - true.min() + 1e-12)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "NRMSE": nrmse,
        "R2": r2,
        "PearsonR": pearson,
    }


# ---------------------------------------------------------
# Plot electrical parameters
# ---------------------------------------------------------
def plot_electrical_params(df, params_true, params_pred, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "Voc": ("Voc_true", "Voc_pred"),
        "Jsc": ("Jsc_true", "Jsc_pred"),
        "Vmpp": ("Vmpp_true", "Vmpp_pred"),
        "Impp": ("Jmpp_true", "Jmpp_pred"),
        "FF": ("FF_true", "FF_pred"),
        "Rs": (params_true[:, 3], params_pred[:, 3]),
        "Rsh": (params_true[:, 4], params_pred[:, 4]),
    }

    for name, cols in metrics.items():

        if isinstance(cols[0], str):
            true = df[cols[0]].values
            pred = df[cols[1]].values
        else:
            true = cols[0]
            pred = cols[1]

        err = pred - true

        # Scatter
        plt.figure(figsize=(6, 5))
        plt.scatter(true, pred, alpha=0.6, edgecolor="k")
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        plt.plot(lims, lims, "r--", label="Ideal")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{name}: True vs Predicted")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_scatter.png"), dpi=200)
        plt.close()

        # Error histogram
        plt.figure(figsize=(6, 5))
        plt.hist(err, bins=40, alpha=0.7, color="steelblue")
        plt.axvline(0, color="r", linestyle="--")
        plt.xlabel(f"{name} Error (pred - true)")
        plt.ylabel("Count")
        plt.title(f"{name} Error Distribution")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_error_hist.png"), dpi=200)
        plt.close()


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------
def evaluate_diode_model(model_path, dataset_path, tech, num_curves=10):

    base_dir = f"outputs/{tech}/ae/physics"
    curve_dir = os.path.join(base_dir, "curves")
    metrics_dir = os.path.join(base_dir, "metrics")

    reset_dir(base_dir)
    reset_dir(curve_dir)
    reset_dir(metrics_dir)

    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)

    jv_in = data["jv_input"].astype(np.float32)
    jv_tgt = data["jv_target"].astype(np.float32)
    weather = data["weather"].astype(np.float32)
    V_grid = data["V_grid"].astype(np.float32)

    diode_true_norm = data["diode_params_norm"].astype(np.float32)
    diode_stats = data["diode_params_stats"].item()

    # Load model
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

    preds = []
    params_preds = []

    curve_rmse = []
    curve_mae = []

    # Inference
    with torch.no_grad():
        for jv, w in zip(jv_in, weather):

            jv_t = torch.tensor(jv, dtype=torch.float32, device=device).unsqueeze(0)
            w_t = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)

            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

            preds.append(I_pred.squeeze(0).cpu().numpy())

            params_pred = unnormalize_params(
                diode_pred_norm.squeeze(0).cpu().numpy(),
                diode_stats
            )
            params_preds.append(params_pred)

    preds = np.array(preds, dtype=np.float32)
    params_preds = np.array(params_preds, dtype=np.float32)

    diode_true = unnormalize_params(diode_true_norm, diode_stats)

    # Compute physics metrics
    rows = []
    for idx in range(len(preds)):

        J_true = jv_tgt[idx]
        J_pred = preds[idx]

        curve_rmse.append(
            np.sqrt(np.mean((J_true - J_pred) ** 2))
        )

        curve_mae.append(
            np.mean(np.abs(J_true - J_pred))
        )

        Voc_t, Jsc_t, Vmpp_t, Jmpp_t, Pmpp_t, FF_t = extract_jv_physics(V_grid, J_true)
        Voc_p, Jsc_p, Vmpp_p, Jmpp_p, Pmpp_p, FF_p = extract_jv_physics(V_grid, J_pred)

        Iph_t, Io_t, n_t, Rs_t, Rsh_t = diode_true[idx]
        Iph_p, Io_p, n_p, Rs_p, Rsh_p = params_preds[idx]
        if Rsh_p < 0:
            Rsh_p = -Rsh_p

        rows.append({
            "idx": idx,
            "Voc_true": Voc_t,
            "Voc_pred": Voc_p,
            "Jsc_true": Jsc_t,
            "Jsc_pred": Jsc_p,
            "Vmpp_true": Vmpp_t,
            "Vmpp_pred": Vmpp_p,
            "Jmpp_true": Jmpp_t,
            "Jmpp_pred": Jmpp_p,
            "Pmpp_true": Pmpp_t,
            "Pmpp_pred": Pmpp_p,
            "FF_true": FF_t,
            "FF_pred": FF_p,
            "Rs_true": Rs_t,
            "Rs_pred": Rs_p,
            "Rsh_true": Rsh_t,
            "Rsh_pred": Rsh_p,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(metrics_dir, "jv_predictions_with_diode_params.csv"), index=False)

    # Save curves
    np.savez(
        os.path.join(metrics_dir, "curves.npz"),
        V=V_grid,
        J_true=jv_tgt,
        J_pred=preds,
        diode_true=diode_true,
        diode_pred=params_preds,
    )

    # Plot JV curves
    plot_dir = os.path.join(curve_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    N = min(num_curves, len(preds))
    for idx in range(N):
        plt.figure(figsize=(6, 5))
        plt.plot(V_grid, jv_tgt[idx], label="True JV", linewidth=2)
        plt.plot(V_grid, preds[idx], "--", label="Pred JV", linewidth=2)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"JV Curve {idx}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"jv_{idx:04d}.png"), dpi=200)
        plt.close()

    # ---------------------------------------------------------
    # Plot electrical parameter comparisons
    # ---------------------------------------------------------
    plot_electrical_params(
        df=df,
        params_true=diode_true,
        params_pred=params_preds,
        out_dir=os.path.join(base_dir, "electrical_param_plots")
    )

    # ---------------------------------------------------------
    # Compute error metrics for all parameters
    # ---------------------------------------------------------
    error_metrics = {

    "Voc":
        compute_error_metrics(
            df["Voc_true"],
            df["Voc_pred"]
        ),

    "Jsc":
        compute_error_metrics(
            df["Jsc_true"],
            df["Jsc_pred"]
        ),

    "Vmpp":
        compute_error_metrics(
            df["Vmpp_true"],
            df["Vmpp_pred"]
        ),

    "Impp":
        compute_error_metrics(
            df["Jmpp_true"],
            df["Jmpp_pred"]
        ),

    "FF":
        compute_error_metrics(
            df["FF_true"],
            df["FF_pred"]
        ),

    "Rs":
        compute_error_metrics(
            diode_true[:,3],
            params_preds[:,3]
        ),

    "Rsh":
        compute_error_metrics(
            diode_true[:,4],
            params_preds[:,4]
        ),

    "Curve RMSE":
        {
            "MAE": np.mean(curve_mae),
            "RMSE": np.mean(curve_rmse),
            "MAPE": np.nan,
            "NRMSE": np.nan,
            "R2": np.nan,
            "PearsonR": np.nan,
        }
}

    metrics_out = []

    for name, m in error_metrics.items():
        metrics_out.append({

            "Parameter": name,

            "MAE": m["MAE"],

            "RMSE": m["RMSE"],

            "MAPE (%)": m["MAPE"],

            "NRMSE": m["NRMSE"],

            "R2": m["R2"],

            "PearsonR": m["PearsonR"]

        })

    metrics_df = pd.DataFrame(metrics_out)
    metrics_df.to_csv(os.path.join(metrics_dir, "error_metrics.csv"), index=False)

    print("\n=== Error Metrics ===")
    print(metrics_df)

    print(f"Evaluation complete → {base_dir}")

    return preds, df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate JV Autoencoder")

    parser.add_argument("--tech", type=str, required=True,
                        help="Technology name (e.g., SunPower_5-1 or Perovskite_5&6)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to AE model .pth file")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to AE dataset .npz file")
    parser.add_argument("--num-curves", type=int, default=10,
                        help="Number of JV curves to plot")

    args = parser.parse_args()

    preds, df = evaluate_diode_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        tech=args.tech,
        num_curves=args.num_curves
    )