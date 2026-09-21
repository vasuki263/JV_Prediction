import os
import shutil

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from jv_analysis import extract_jv_physics
from matplotlib.lines import Line2D
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from model_autoencoder import (
    PhysicsInformedJVModel, SingleDiodeSolver
)


# ============================================================
# Directory utilities
# ============================================================

def reset_dir(path):

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(
        path,
        exist_ok=True
    )

# ========================================================
# Smooth JV curves for visualization ONLY
# ========================================================
def smooth_jv_curve(current, sigma=8.0):
    """
    Smooth JV curve for plotting only.

    The original current data is NOT modified.
    This function should not be used before electrical
    parameter extraction or metric calculation.
    """
    current = np.asarray(current, dtype=float)

    if len(current) < 5:
        return current

    return gaussian_filter1d(
        current,
        sigma=sigma,
        mode="nearest"
    )

# ============================================================
# Regression metrics
# ============================================================

def compute_error_metrics(true, pred, param_name=None):

    true = np.asarray(
        true,
        dtype=float
    )

    pred = np.asarray(
        pred,
        dtype=float
    )

    mask = (
        np.isfinite(true)
        & np.isfinite(pred)
    )

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

    mae = mean_absolute_error(
        true,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            true,
            pred
        )
    )

    # --------------------------------------------------------
    # MAPE
    #
    # Avoid division by values close to zero.
    # This is especially important for Voc/Io/FF.
    # --------------------------------------------------------

    scale = np.maximum(
        np.abs(true),
        1e-6
    )

    mape = np.mean(
        np.abs(
            (true - pred)
            / scale
        )
    ) * 100.0

    # --------------------------------------------------------
    # NRMSE
    # --------------------------------------------------------

    data_range = (
        np.max(true)
        - np.min(true)
    )

    if data_range > 1e-12:

        nrmse = (
            rmse
            / data_range
        )

    else:

        nrmse = np.nan

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------
    # --------------------------------------------------------
    # R2 (scale all parameters except Voc, Rs, Rsh)
    # --------------------------------------------------------

    if len(true) > 1:

        # Scale unless parameter has meaningful physical units
        if param_name not in ["Voc", "Rs", "Rsh"]:
            t_min = np.min(true)
            t_max = np.max(true)

            if t_max - t_min > 1e-12:
                true_scaled = (true - t_min) / (t_max - t_min)
                pred_scaled = (pred - t_min) / (t_max - t_min)
            else:
                true_scaled = true.copy()
                pred_scaled = pred.copy()
        else:
            true_scaled = true
            pred_scaled = pred

        try:
            r2 = r2_score(true_scaled, pred_scaled)
        except Exception:
            r2 = np.nan

        try:
            if np.std(true_scaled) > 1e-12 and np.std(pred_scaled) > 1e-12:
                pearson = np.corrcoef(true_scaled, pred_scaled)[0, 1]
            else:
                pearson = np.nan
        except Exception:
            pearson = np.nan

    else:
        r2 = np.nan
        pearson = np.nan

    return {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "NRMSE": nrmse,

        "R2": r2,

        "PearsonR": pearson,
    }


# ============================================================
# Plot parameter comparison
# ============================================================

def plot_parameter_comparison(
    name,
    true,
    pred,
    out_dir
):

    mask = (
        np.isfinite(true)
        & np.isfinite(pred)
    )

    true = true[mask]
    pred = pred[mask]

    if len(true) == 0:
        return

    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.scatter(
        true,
        pred,
        alpha=0.6,
        edgecolor="k"
    )

    lo = min(
        true.min(),
        pred.min()
    )

    hi = max(
        true.max(),
        pred.max()
    )

    plt.plot(
        [lo, hi],
        [lo, hi],
        "r--",
        label="Ideal"
    )

    plt.xlabel(
        f"Fitted {name}"
    )

    plt.ylabel(
        f"Predicted {name}"
    )

    plt.title(
        f"{name}: Fitted vs Predicted"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            f"{name}_scatter.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Error histogram
    # --------------------------------------------------------

    error = (
        pred
        - true
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.hist(
        error,
        bins=40,
        alpha=0.7
    )

    plt.axvline(
        0,
        color="r",
        linestyle="--"
    )

    plt.xlabel(
        f"{name} Error "
        "(predicted - fitted)"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        f"{name} Error Distribution"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            f"{name}_error_hist.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# Main evaluation
# ============================================================

def evaluate_diode_model(

    model_path,

    dataset_path,

    tech,

    num_curves=15,
):

    base_dir = (
        f"outputs/{tech}/SEPIV3/test_test"
    )

    curve_dir = os.path.join(
        base_dir,
        "curves"
    )

    metrics_dir = os.path.join(
        base_dir,
        "metrics"
    )

    parameter_dir = os.path.join(
        base_dir,
        "diode_parameters"
    )

    reset_dir(
        base_dir
    )

    reset_dir(
        curve_dir
    )

    reset_dir(
        metrics_dir
    )

    reset_dir(
        parameter_dir
    )

    # ========================================================
    # Load dataset
    # ========================================================

    data = np.load(
        dataset_path,
        allow_pickle=True
    )

    jv_true = (
        data["jv_target"]
        .astype(np.float32)
    )

    weather = (
        data["weather"]
        .astype(np.float32)
    )

    V_grid = (
        data["V_grid"]
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Fitted diode parameters
    # --------------------------------------------------------

    if "diode_params" in data:

        diode_true = (
            data["diode_params"]
            .astype(np.float32)
        )

    else:

        raise KeyError(
            "Dataset must contain "
            "'diode_params'."
        )

    # ========================================================
    # Load checkpoint
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device, weights_only=False
    )

    # ========================================================
    # Construct model
    # ========================================================

    model = PhysicsInformedJVModel(

        jv_dim=checkpoint["jv_dim"],

        weather_dim=checkpoint["weather_dim"],

        voltage_grid=checkpoint["V_grid"],

        latent_dim=checkpoint["latent_dim"],

        diode_mean=checkpoint[
            "diode_mean"
        ],

        diode_std=checkpoint[
            "diode_std"
        ],

    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # Prediction containers
    # ========================================================

    final_preds = []

    physics_preds = []

    residual_preds = []

    diode_preds_norm = []

    # ========================================================
    # Batch inference
    # ========================================================

    weather_tensor = torch.tensor(
        weather,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        (
            final_pred,
            physics_pred,
            residual_pred,
            z,
            diode_pred_norm,
        ) = model(
            weather_tensor
        )

    final_preds = (
        final_pred
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    physics_preds = (
        physics_pred
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    residual_preds = (
        residual_pred
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    diode_preds_norm = (
        diode_pred_norm
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # ========================================================
    # Unnormalize diode predictions
    # ========================================================

    diode_mean = np.asarray(
        checkpoint[
            "diode_mean"
        ],
        dtype=np.float32
    )

    diode_std = np.asarray(
        checkpoint[
            "diode_std"
        ],
        dtype=np.float32
    )

    diode_preds = (
        diode_preds_norm
        * diode_std
        + diode_mean
    )
    # ========================================================
    # TRUE JV RESIDUAL
    #
    # This is the quantity the neural residual should learn.
    #
    # measured JV
    #       -
    # predicted single-diode JV
    # ========================================================

    true_jv_residual = (
        jv_true
        - physics_preds
    )

    # ========================================================
    # Curve-level metrics
    # ========================================================

    final_curve_rmse = np.sqrt(
        np.mean(
            (
                jv_true
                - final_preds
            ) ** 2,
            axis=1
        )
    )

    final_curve_mae = np.mean(
        np.abs(
            jv_true
            - final_preds
        ),
        axis=1
    )

    physics_curve_rmse = np.sqrt(
        np.mean(
            (
                jv_true
                - physics_preds
            ) ** 2,
            axis=1
        )
    )

    physics_curve_mae = np.mean(
        np.abs(
            jv_true
            - physics_preds
        ),
        axis=1
    )

    residual_curve_rmse = np.sqrt(
        np.mean(
            (
                true_jv_residual
                - residual_preds
            ) ** 2,
            axis=1
        )
    )

    residual_curve_mae = np.mean(
        np.abs(
            true_jv_residual
            - residual_preds
        ),
        axis=1
    )

    # ========================================================
    # Electrical parameters from JV curves ONLY
    # ========================================================

    rows = []

    for idx in range(
        len(jv_true)
    ):

        # ----------------------------------------------------
        # Measured JV
        # ----------------------------------------------------

        (
            Voc_true,
            Jsc_true,
            Vmpp_true,
            Jmpp_true,
            Pmpp_true,
            FF_true,
        ) = extract_jv_physics(
            V_grid,
            jv_true[idx], area_m2=1
        )

        # ----------------------------------------------------
        # Final predicted JV
        # ----------------------------------------------------

        (
            Voc_pred,
            Jsc_pred,
            Vmpp_pred,
            Jmpp_pred,
            Pmpp_pred,
            FF_pred,
        ) = extract_jv_physics(
            V_grid,
            final_preds[idx], area_m2=1
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        row = {

            "idx": idx,

            "Temperature": weather[idx, 0],
            "Irradiance": weather[idx, 1],

            # JV
            "Final_JV_RMSE":
                final_curve_rmse[idx],

            "Final_JV_MAE":
                final_curve_mae[idx],

            "Physics_JV_RMSE":
                physics_curve_rmse[idx],

            "Physics_JV_MAE":
                physics_curve_mae[idx],

            "Residual_RMSE":
                residual_curve_rmse[idx],

            "Residual_MAE":
                residual_curve_mae[idx],

            # Electrical parameters
            "Voc_true":
                Voc_true,

            "Voc_pred":
                Voc_pred,

            "Jsc_true":
                Jsc_true,

            "Jsc_pred":
                Jsc_pred,

            "Vmpp_true":
                Vmpp_true,

            "Vmpp_pred":
                Vmpp_pred,

            "Jmpp_true":
                Jmpp_true,

            "Jmpp_pred":
                Jmpp_pred,

            "Pmpp_true":
                Pmpp_true,

            "Pmpp_pred":
                Pmpp_pred,

            "FF_true":
                FF_true,

            "FF_pred":
                FF_pred,

            # Fitted diode parameters
            "Iph_fitted":
                diode_true[idx, 0],

            "Iph_pred":
                diode_preds[idx, 0],

            "Io_fitted":
                diode_true[idx, 1],

            "Io_pred":
                diode_preds[idx, 1],

            "n_fitted":
                diode_true[idx, 2],

            "n_pred":
                diode_preds[idx, 2],

            "Rs_fitted":
                diode_true[idx, 3],

            "Rs_pred":
                diode_preds[idx, 3],

            "Rsh_fitted":
                diode_true[idx, 4],

            "Rsh_pred":
                diode_preds[idx, 4],
        }

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # Save detailed results
    # ========================================================

    df.to_csv(
        os.path.join(
            metrics_dir,
            "predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # Save curves
    # ========================================================

    np.savez(
        os.path.join(
            metrics_dir,
            "curves.npz"
        ),

        V=V_grid,

        J_true=jv_true,

        J_pred=final_preds,

        J_physics=physics_preds,

        J_residual_true=true_jv_residual,

        J_residual_pred=residual_preds,

        diode_true=diode_true,

        diode_pred=diode_preds,
    )

    # ========================================================
    # JV plots: TRUE vs PREDICTED only
    # ========================================================

    plot_dir = os.path.join(
        curve_dir,
        "plots"
    )

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    N = min(
        num_curves,
        len(final_preds)
    )

    for idx in range(N):
        # ----------------------------------------------------
        # Original curves
        # ----------------------------------------------------

        true_curve = jv_true[idx]
        pred_curve = final_preds[idx]
        physics_curve = physics_preds[idx]
        # ----------------------------------------------------
        # Smooth ONLY for visualization
        # ----------------------------------------------------

        true_smooth = smooth_jv_curve(
            true_curve,
            sigma=25.0
        )

        pred_smooth = smooth_jv_curve(
            pred_curve,
            sigma=25.0
        )

        physics_smooth = smooth_jv_curve(
            physics_curve,
            sigma=25.0
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            V_grid,
            true_smooth,
            label="True JV",
            linewidth=2.5
        )

        plt.plot(
            V_grid,
            physics_smooth,
            "-.",
            label="Physics JV",
            linewidth=2.5
        )

        plt.plot(
            V_grid,
            pred_smooth,
            "--",
            label="Predicted JV",
            linewidth=2.5
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        plt.xlabel(
            "Voltage (V)",
            fontsize=12
        )

        plt.ylabel(
            "Current Density(A/m$^2$)",
            fontsize=12
        )

        plt.title(
            f"JV Curve {idx}",
            fontsize=13
        )

        plt.grid(
            alpha=0.25
        )

        plt.legend(
            frameon=True
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                plot_dir,
                f"jv_{idx:04d}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ========================================================
    # Residual plots
    # ========================================================

    residual_plot_dir = os.path.join(
        curve_dir,
        "residuals"
    )

    os.makedirs(
        residual_plot_dir,
        exist_ok=True
    )

    for idx in range(N):

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            V_grid,
            true_jv_residual[idx],
            label="True JV residual",
            linewidth=2
        )
        pred_residual_smooth = smooth_jv_curve(residual_preds[idx], sigma=8.0)
        plt.plot(
            V_grid,
            residual_preds[idx],
            "--",
            label="Predicted JV residual",
            linewidth=2
        )

        plt.xlabel(
            "Voltage (V)"
        )

        plt.ylabel(
            "Current residual (A)"
        )

        plt.title(
            f"JV Residual {idx}"
        )

        plt.grid(
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                residual_plot_dir,
                f"residual_{idx:04d}.png"
            ),
            dpi=200
        )

        plt.close()


    error_metrics = {

        "Final JV Curve":
            compute_error_metrics(
                jv_true.flatten(),
                final_preds.flatten()
            ),

        "Single-Diode Physics":
            compute_error_metrics(
                jv_true.flatten(),
                physics_preds.flatten()
            ),

        "Diode JV Residual":
            compute_error_metrics(
                true_jv_residual.flatten(),
                residual_preds.flatten()
            ),

        "Voc":
            compute_error_metrics(df["Voc_true"], df["Voc_pred"], param_name="Voc"),

        "Jsc":
            compute_error_metrics(df["Jsc_true"], df["Jsc_pred"], param_name="Jsc"),

        "Vmpp":
            compute_error_metrics(df["Vmpp_true"], df["Vmpp_pred"], param_name="Vmpp"),

        "Jmpp":
            compute_error_metrics(df["Jmpp_true"], df["Jmpp_pred"], param_name="Jmpp"),

        "Pmpp":
            compute_error_metrics(df["Pmpp_true"], df["Pmpp_pred"], param_name="Pmpp"),

        "FF":
            compute_error_metrics(df["FF_true"], df["FF_pred"], param_name="FF"),

        # ----------------------------------------------------
        # Diode parameter supervision
        #
        # These compare directly against fitted parameters.
        # ----------------------------------------------------

        "Iph":
            compute_error_metrics(diode_true[:,0], diode_preds[:,0], param_name="Iph"),

        "Io":
            compute_error_metrics(diode_true[:,1], diode_preds[:,1], param_name="Io"),

        "n":
            compute_error_metrics(diode_true[:,2], diode_preds[:,2], param_name="n"),

        "Rs":
            compute_error_metrics(diode_true[:,3], diode_preds[:,3], param_name="Rs"),

        "Rsh":
            compute_error_metrics(diode_true[:, 4], diode_preds[:, 4], param_name="Rsh")
    }

    # ========================================================
    # Mean curve metrics
    # ========================================================

    error_metrics[
        "Final JV RMSE mean"
    ] = {

        "MAE":
            np.mean(
                final_curve_mae
            ),

        "RMSE":
            np.mean(
                final_curve_rmse
            ),

        "MAPE":
            np.nan,

        "NRMSE":
            np.nan,

        "R2":
            np.nan,

        "PearsonR":
            np.nan,
    }

    error_metrics[
        "Physics JV RMSE mean"
    ] = {

        "MAE":
            np.mean(
                physics_curve_mae
            ),

        "RMSE":
            np.mean(
                physics_curve_rmse
            ),

        "MAPE":
            np.nan,

        "NRMSE":
            np.nan,

        "R2":
            np.nan,

        "PearsonR":
            np.nan,
    }

    error_metrics[
        "Residual RMSE mean"
    ] = {

        "MAE":
            np.mean(
                residual_curve_mae
            ),

        "RMSE":
            np.mean(
                residual_curve_rmse
            ),

        "MAPE":
            np.nan,

        "NRMSE":
            np.nan,

        "R2":
            np.nan,

        "PearsonR":
            np.nan,
    }

    # ========================================================
    # Convert to DataFrame
    # ========================================================

    metrics_rows = []

    for name, metrics in (
        error_metrics.items()
    ):

        metrics_rows.append({

            "Parameter":
                name,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "MAPE":
                metrics["MAPE"],

            "NRMSE":
                metrics["NRMSE"],

            "R2":
                metrics["R2"],

            "PearsonR":
                metrics["PearsonR"],
        })

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        os.path.join(
            metrics_dir,
            "error_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # Diode parameter plots
    # ========================================================

    diode_names = [
        "Iph",
        "Io",
        "n",
        "Rs",
        "Rsh",
    ]

    for k, name in enumerate(
        diode_names
    ):

        plot_parameter_comparison(

            name=name,

            true=diode_true[:, k],

            pred=diode_preds[:, k],

            out_dir=parameter_dir,
        )

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n=== Error Metrics ==="
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print(
        "\n=== Mean Absolute JV Residual Error ==="
    )

    print(
        np.mean(
            residual_curve_mae
        )
    )

    print(
        f"\nEvaluation complete → "
        f"{base_dir}"
    )

    return (
        final_preds,
        physics_preds,
        residual_preds,
        diode_preds,
        df,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Physics-Informed JV Model"
        )
    )

    parser.add_argument(
        "--tech",
        type=str,
        required=True,
        help=(
            "Technology name"
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Path to trained model"
        )
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help=(
            "Path to dataset"
        )
    )

    parser.add_argument(
        "--num-curves",
        type=int,
        default=35,
        help=(
            "Number of curves to plot"
        )
    )

    args = parser.parse_args()

    evaluate_diode_model(

        model_path=args.model_path,

        dataset_path=args.dataset_path,

        tech=args.tech,

        num_curves=args.num_curves,
    )