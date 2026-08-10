import os
import shutil
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from model_autoencoder import PhysicsInformedJVModel


# =========================================================
# Utility
# =========================================================
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# =========================================================
# Robust Voc extraction
# =========================================================
def extract_voc(V, I):

    V = np.asarray(V, float)
    I = np.asarray(I, float)

    for k in range(len(I) - 1):

        if (I[k] <= 0 and I[k + 1] >= 0) or (
            I[k] >= 0 and I[k + 1] <= 0
        ):

            return np.interp(
                0.0,
                [I[k], I[k + 1]],
                [V[k], V[k + 1]],
            )

    return V[np.argmin(np.abs(I))]


# =========================================================
# JV physics extraction
# =========================================================
def extract_jv_physics(V, I):

    V = np.asarray(V, float)
    I = np.asarray(I, float)

    idx = np.argsort(V)

    V = V[idx]
    I = I[idx]

    Jsc = np.interp(0.0, V, I)
    Voc = extract_voc(V, I)

    mask = V >= 0

    Vp = V[mask]
    Ip = I[mask]

    P = np.abs(Vp * Ip)

    k = np.argmax(P)

    Vmpp = Vp[k]
    Jmpp = Ip[k]
    Pmpp = P[k]

    FF = Pmpp / (np.abs(Voc) * np.abs(Jsc) + 1e-12)

    return Voc, Jsc, Vmpp, Jmpp, Pmpp, FF


# =========================================================
# Single diode model (NumPy version)
# =========================================================
def single_diode_current(
    V,
    Iph,
    Io,
    n,
    Rs,
    Rsh,
    Vt=0.02585,
    Ns=1,
    n_iter=40,
):

    V = np.asarray(V, dtype=float)

    I = np.full_like(V, Iph, dtype=float)

    for _ in range(n_iter):

        exp_arg = (V + I * Rs) / (n * Ns * Vt)
        exp_arg = np.clip(exp_arg, -100, 80)

        exp_term = np.exp(exp_arg)

        f = (
            Iph
            - Io * (exp_term - 1.0)
            - (V + I * Rs) / Rsh
            - I
        )

        df = (
            -Io
            * exp_term
            * (Rs / (n * Ns * Vt))
            - Rs / Rsh
            - 1.0
        )

        I -= f / (df + 1e-12)

    return I


# =========================================================
# Residual function for optimization
# =========================================================
def diode_residuals(params, V, I_measured):

    Iph, Io, n, Rs, Rsh = params

    # Model current
    I_model = single_diode_current(V, Iph, Io, n, Rs, Rsh)

    weights = np.ones_like(V)

    # Region for Iph (Jsc)
    weights[V < 0.05] = 12.0

    # Region for Rsh (low-voltage curvature)
    weights[(V >= 0.05) & (V < 0.15)] = 8.0

    # Region for n, Io (mid-voltage)
    weights[(V >= 0.15) & (V < 0.7 * np.max(V))] = 3.0

    # Region for Rs (near Voc)
    weights[V > 0.7 * np.max(V)] = 10.0

    res_main = weights * (I_model - I_measured)

    # --- Extract physics from measured curve ---
    try:
        Voc_meas, Jsc_meas, Vmpp_meas, Jmpp_meas, _, _ = extract_jv_physics(V, I_measured)
    except:
        Voc_meas, Jsc_meas, Vmpp_meas, Jmpp_meas = 0, 0, 0, 0

    # --- Extract physics from model curve ---
    try:
        Voc_mod, Jsc_mod, Vmpp_mod, Jmpp_mod, _, _ = extract_jv_physics(V, I_model)
    except:
        Voc_mod, Jsc_mod, Vmpp_mod, Jmpp_mod = 0, 0, 0, 0

    # --- Physics penalties ---
    pen_Jsc = 15.0 * (Jsc_mod - Jsc_meas)
    pen_Voc = 5.0 * (Voc_mod - Voc_meas)

    # Slope penalty near Voc (Rs sensitivity)
    # Strong slope penalty near Voc (Rs sensitivity)
    try:
        idx_voc = np.argmin(np.abs(V - Voc_meas))
        if 2 < idx_voc < len(V) - 3:
            slope_mod = (I_model[idx_voc + 1] - I_model[idx_voc - 1]) / (V[idx_voc + 1] - V[idx_voc - 1])
            slope_meas = (I_measured[idx_voc + 1] - I_measured[idx_voc - 1]) / (V[idx_voc + 1] - V[idx_voc - 1])
            pen_slope = 12.0 * (slope_mod - slope_meas)
        else:
            pen_slope = 0.0
    except:
        pen_slope = 0.0

    # Rsh physics penalty: enforce correct low-voltage slope
    try:
        idx0 = np.argmin(np.abs(V - 0.0))
        if 1 < idx0 < len(V) - 2:
            slope_mod_0 = (I_model[idx0 + 1] - I_model[idx0 - 1]) / (V[idx0 + 1] - V[idx0 - 1])
            slope_meas_0 = (I_measured[idx0 + 1] - I_measured[idx0 - 1]) / (V[idx0 + 1] - V[idx0 - 1])
            pen_Rsh = 6.0 * (slope_mod_0 - slope_meas_0)
        else:
            pen_Rsh = 0.0
    except:
        pen_Rsh = 0.0

    # --- Return combined residuals ---
    return np.concatenate([
        res_main,
        [pen_Jsc, pen_Voc, pen_slope, pen_Rsh]
    ])



# =========================================================
# Fit diode parameters from a JV curve
# =========================================================
def fit_diode_parameters(V, I):

    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    Voc, Jsc, Vmpp, Jmpp, Pmpp, FF = extract_jv_physics(V, I)

    # Iph_guess = max(Jsc, np.max(I), 1e-3)
    Iph_guess = Jsc * 1.01
    # Better Rs guess using slope near Voc
    idx_voc = np.argmin(np.abs(V - Voc))
    if 2 < idx_voc < len(V) - 3:
        slope_meas = (I[idx_voc + 1] - I[idx_voc - 1]) / (V[idx_voc + 1] - V[idx_voc - 1])
        Rs_guess = np.clip(-1.0 / slope_meas, 0.01, 3.0)
    else:
        Rs_guess = 0.5

    Rsh_guess = np.clip(abs(V[1] / (Iph_guess - I[1] + 1e-9)), 200, 5000)

    x0 = np.array(
        [
            Iph_guess,
            1e-9,
            1.2,
            Rs_guess,
            Rsh_guess,
        ]
    )

    lower = np.array(
        [
            0.0,
            1e-15,
            0.8,
            0.0,
            10.0,
        ]
    )

    upper = np.array(
        [
            1.3 * Jsc, #100.0
            1e-4,
            3.0,
            6.0, #20.0,
            5000,
        ]
    )

    try:

        result = least_squares(
            diode_residuals,
            x0,
            bounds=(lower, upper),
            args=(V, I),
            method="trf",
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        return result.x

    except Exception as e:

        print(f"Fit failed: {e}")

        return np.array([np.nan] * 5)


# =========================================================
# Error metrics
# =========================================================
def compute_error_metrics(true, pred):

    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(true) & np.isfinite(pred)

    true = true[mask]
    pred = pred[mask]

    if len(true) == 0:

        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

    mae = mean_absolute_error(true, pred)

    rmse = np.sqrt(mean_squared_error(true, pred))

    r2 = (
        r2_score(true, pred)
        if len(true) > 1
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

def unnormalize_params(params_norm, stats):
    return params_norm * stats["std"] + stats["mean"]

# =========================================================
# Main evaluation
# =========================================================
def evaluate_diode_model(
    model_path,
    dataset_path,
    out_dir="outputs_fit",
):

    reset_dir(out_dir)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    data = np.load(dataset_path, allow_pickle=True)

    jv_in = data["jv_input"].astype(np.float32)

    jv_tgt = data["jv_target"].astype(np.float32)

    weather = data["weather"].astype(np.float32)

    V_grid = data["V_grid"].astype(np.float32)

    diode_true = data["diode_params"].astype(np.float32)
    diode_stats = data["diode_params_stats"].item()

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    # -----------------------------------------------------
    # Predict JV curves
    # -----------------------------------------------------
    preds = []
    params_preds = []

    with torch.no_grad():

        for jv, w in zip(jv_in, weather):

            jv_t = torch.tensor(
                jv,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            w_t = torch.tensor(
                w,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

            preds.append(
                I_phys.squeeze(0)
                .cpu()
                .numpy()
            )
            params_pred = unnormalize_params(
                diode_pred_norm.squeeze(0).cpu().numpy(),
                diode_stats
            )
            params_preds.append(params_pred)

    preds = np.asarray(preds, dtype=np.float32)
    params_preds = np.array(params_preds, dtype=np.float32)

    # -----------------------------------------------------
    # Extract diode parameters from predicted JV
    # -----------------------------------------------------
    print("Fitting diode parameters from predicted JV curves...")

    fitted_params = []

    for i, curve in enumerate(preds):

        params = fit_diode_parameters(
            V_grid,
            curve,
        )

        fitted_params.append(params)

        if (i + 1) % 50 == 0:

            print(
                f"Processed {i + 1}/{len(preds)}"
            )

    fitted_params = np.asarray(
        fitted_params,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Save extracted parameters
    # -----------------------------------------------------
    df = pd.DataFrame(
        {
            "Iph_true": diode_true[:, 0],
            "Iph_fit": fitted_params[:, 0],
            "Io_true": diode_true[:, 1],
            "Io_fit": fitted_params[:, 1],
            "n_true": diode_true[:, 2],
            "n_fit": fitted_params[:, 2],
            "Rs_true": diode_true[:, 3],
            "Rs_fit": fitted_params[:, 3],
            "Rsh_true": diode_true[:, 4],
            "Rsh_fit": fitted_params[:, 4],
        }
    )

    df.to_csv(
        os.path.join(
            out_dir,
            "fitted_diode_parameters.csv",
        ),
        index=False,
    )

    # -----------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------
    metrics = {}

    names = [
        "Iph",
        "Io",
        "n",
        "Rs",
        "Rsh",
    ]

    for i, name in enumerate(names):

        metrics[name] = compute_error_metrics(
            diode_true[:, i],
            fitted_params[:, i],
        )

    metrics_df = pd.DataFrame(metrics).T

    metrics_df.to_csv(
        os.path.join(
            out_dir,
            "diode_fit_metrics.csv",
        )
    )

    print("\n=== Fitted Parameter Metrics ===")

    print(metrics_df)

    # -----------------------------------------------------
    # Plot comparisons
    # -----------------------------------------------------
    plot_dir = os.path.join(out_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)

    for i, name in enumerate(names):

        true = diode_true[:, i]

        pred = fitted_params[:, i]

        mask = np.isfinite(true) & np.isfinite(pred)

        true = true[mask]

        pred = pred[mask]

        plt.figure(figsize=(5, 5))

        plt.scatter(
            true,
            pred,
            alpha=0.6,
            edgecolor="k",
        )

        lims = [
            min(true.min(), pred.min()),
            max(true.max(), pred.max()),
        ]

        plt.plot(lims, lims, "r--")

        plt.xlabel(f"True {name}")

        plt.ylabel(f"Fitted {name}")

        plt.title(f"{name}: True vs Fitted")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                plot_dir,
                f"{name}_fit_scatter.png",
            ),
            dpi=200,
        )

        plt.close()

    print(f"\nResults saved to: {out_dir}")

    return fitted_params


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract diode parameters from predicted JV curves"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs_fit",
    )

    args = parser.parse_args()

    evaluate_diode_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        out_dir=args.out_dir,
    )
