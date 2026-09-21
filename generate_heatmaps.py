import os
import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from model_autoencoder import PhysicsInformedJVModel
from jv_analysis import extract_jv_physics


# ============================================================
# Configuration
# ============================================================
# --------------------------------------------------------
# Fixed global ranges for all heatmaps
# --------------------------------------------------------
TEMPERATURE_MIN = 20.0      # or 5.0, or whatever you want
TEMPERATURE_MAX = 60.0     # e.g. 60 °C

IRRADIANCE_MIN = 200.0       # or 50.0, etc.
IRRADIANCE_MAX = 1200.0    # e.g. 1200 W/m²

DIODE_PARAMS = [
    "Iph",
    "Io",
    "n",
    "Rs",
    "Rsh",
]

ELECTRICAL_PARAMS = [
    "Voc",
    "Jsc",
    "Vmpp",
    "Jmpp",
    "Pmpp",
    "FF",
]

ALL_PARAMS = (
    DIODE_PARAMS
    + ELECTRICAL_PARAMS
)


# ============================================================
# Units
# ============================================================

def get_units(parameter):

    units = {
        "Voc": "V",
        "Jsc": "A/m²",
        "Vmpp": "V",
        "Jmpp": "A/m²",
        "Pmpp": "W/m²",
        "FF": "",
        "Iph": "A",
        "Io": "A",
        "n": "",
        "Rs": "Ω",
        "Rsh": "Ω",
    }

    unit = units.get(
        parameter,
        ""
    )

    if unit:
        return f"{parameter} [{unit}]"

    return parameter


# ============================================================
# Load model
# ============================================================

def load_model(
    model_path,
    dataset,
):

    V_grid = dataset[
        "V_grid"
    ].astype(np.float32)

    weather = dataset[
        "weather"
    ].astype(np.float32)

    ckpt = torch.load(
        model_path,
        map_location="cpu", weights_only=False
    )

    model = PhysicsInformedJVModel(
        jv_dim=dataset[
            "jv_target"
        ].shape[1],

        weather_dim=weather.shape[1],

        voltage_grid=V_grid,

        latent_dim=ckpt[
            "latent_dim"
        ],

        diode_mean=ckpt[
            "diode_mean"
        ],

        diode_std=ckpt[
            "diode_std"
        ],
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    model.eval()

    return model, V_grid

def calculate_binned_metrics(df, parameter, n_bins=8, min_samples=5):
    """
    Calculate agreement between true and predicted parameter values
    after binning by temperature and irradiance.

    Metrics are calculated only for bins containing at least
    min_samples observations.
    """

    # Create common temperature and irradiance bins
    T_edges = np.linspace(
        df["temperature"].min(),
        df["temperature"].max(),
        n_bins + 1
    )

    G_edges = np.linspace(
        df["irradiance"].min(),
        df["irradiance"].max(),
        n_bins + 1
    )

    df = df.copy()

    df["T_bin"] = pd.cut(
        df["temperature"],
        bins=T_edges,
        include_lowest=True
    )

    df["G_bin"] = pd.cut(
        df["irradiance"],
        bins=G_edges,
        include_lowest=True
    )

    # Calculate mean true/predicted value and sample count
    grouped = (
        df.groupby(["T_bin", "G_bin"], observed=False)
        .agg(
            true_mean=(f"{parameter}_true", "mean"),
            pred_mean=(f"{parameter}_pred", "mean"),
            count=(f"{parameter}_true", "count")
        )
        .reset_index()
    )

    # Keep only sufficiently populated bins
    valid = grouped["count"] >= min_samples
    grouped_valid = grouped[valid].copy()

    y_true = grouped_valid["true_mean"].to_numpy()
    y_pred = grouped_valid["pred_mean"].to_numpy()

    if len(y_true) < 2:
        return {
            "parameter": parameter,
            "n_bins": n_bins,
            "min_samples": min_samples,
            "valid_bins": len(y_true),
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan
        }

    # Errors
    errors = y_pred - y_true

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot > 1e-12:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    return {
        "parameter": parameter,
        "n_bins": n_bins,
        "min_samples": min_samples,
        "valid_bins": len(y_true),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }
# ============================================================
# Generate predictions
# ============================================================

def generate_predictions(
    model,
    dataset,
    device,
):

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    weather = dataset[
        "weather"
    ].astype(np.float32)

    jv_true = dataset[
        "jv_target"
    ].astype(np.float32)

    V_grid = dataset[
        "V_grid"
    ].astype(np.float32)

    # --------------------------------------------------------
    # True diode parameters
    # --------------------------------------------------------

    if "diode_params" in dataset:

        diode_true = dataset[
            "diode_params"
        ].astype(np.float32)

    elif "diode_params_norm" in dataset:

        diode_norm = dataset[
            "diode_params_norm"
        ].astype(np.float32)

        if "diode_params_stats" in dataset:

            stats = dataset[
                "diode_params_stats"
            ].item()

            diode_true = (
                diode_norm
                * stats["std"]
                + stats["mean"]
            )

        else:

            raise ValueError(
                "Found diode_params_norm "
                "but no diode_params_stats."
            )

    else:

        raise ValueError(
            "Dataset must contain either "
            "'diode_params' or "
            "'diode_params_norm'."
        )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    weather_t = torch.tensor(
        weather,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        (
            I_pred,
            I_phys,
            delta_I,
            z,
            diode_pred_norm,
        ) = model(weather_t)

    # --------------------------------------------------------
    # Final J-V prediction
    #
    # I_pred = I_phys + delta_I
    # --------------------------------------------------------

    I_pred = (
        I_pred
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Convert normalized diode parameters
    # --------------------------------------------------------

    diode_pred_norm = (
        diode_pred_norm
        .detach()
        .cpu()
        .numpy()
    )

    diode_mean = (
        model.diode_mean
        .detach()
        .cpu()
        .numpy()
    )

    diode_std = (
        model.diode_std
        .detach()
        .cpu()
        .numpy()
    )

    diode_pred = (
        diode_pred_norm
        * diode_std
        + diode_mean
    )

    # --------------------------------------------------------
    # Enforce physical constraint: Rsh must be positive
    # --------------------------------------------------------
    RSH_INDEX = DIODE_PARAMS.index("Rsh")
    diode_pred[:, RSH_INDEX] = np.abs(diode_pred[:, RSH_INDEX])

    # --------------------------------------------------------
    # Extract electrical parameters
    # from TRUE and FINAL predicted J-V curves
    # --------------------------------------------------------

    electrical_true = []

    electrical_pred = []

    for i in range(
        len(jv_true)
    ):

        true_values = extract_jv_physics(
            V_grid,
            jv_true[i],
            area_m2=0.7
        )

        pred_values = extract_jv_physics(
            V_grid,
            I_pred[i],
            area_m2=0.7
        )

        electrical_true.append(
            true_values
        )

        electrical_pred.append(
            pred_values
        )

    electrical_true = np.asarray(
        electrical_true,
        dtype=np.float32
    )

    electrical_pred = np.asarray(
        electrical_pred,
        dtype=np.float32
    )

    return (
        weather,
        diode_true,
        diode_pred,
        electrical_true,
        electrical_pred,
    )


# ============================================================
# Build dataframe
# ============================================================

def build_dataframe(
    weather,
    diode_true,
    diode_pred,
    electrical_true,
    electrical_pred,
):

    rows = []

    for i in range(
        len(weather)
    ):

        # ----------------------------------------------------
        # Same weather convention as your original code
        # ----------------------------------------------------

        temperature = weather[
            i, 0
        ]

        irradiance = weather[
            i, 1
        ]

        row = {

            "index": i,

            "temperature": temperature,

            "irradiance": irradiance,
        }

        # ----------------------------------------------------
        # Diode parameters
        # ----------------------------------------------------

        for j, name in enumerate(
            DIODE_PARAMS
        ):

            row[
                f"{name}_true"
            ] = diode_true[
                i, j
            ]

            row[
                f"{name}_pred"
            ] = diode_pred[
                i, j
            ]

            row[
                f"{name}_error"
            ] = (
                diode_pred[i, j]
                - diode_true[i, j]
            )

        # ----------------------------------------------------
        # Electrical parameters
        # ----------------------------------------------------

        for j, name in enumerate(
            ELECTRICAL_PARAMS
        ):

            row[
                f"{name}_true"
            ] = electrical_true[
                i, j
            ]

            row[
                f"{name}_pred"
            ] = electrical_pred[
                i, j
            ]

            row[
                f"{name}_error"
            ] = (
                electrical_pred[i, j]
                - electrical_true[i, j]
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# Create binned heatmap matrix
# ============================================================

def create_heatmap_matrix(
    df,
    parameter,
    value_type,
    temperature_bins,
    irradiance_bins,
):
    """
    Convert point-wise observations into a traditional
    temperature × irradiance matrix.

    Each cell contains the MEAN parameter value for all
    observations falling inside that temperature/irradiance bin.
    """

    column = (
        f"{parameter}_{value_type}"
    )

    data = df[
        [
            "temperature",
            "irradiance",
            column,
        ]
    ].copy()

    # --------------------------------------------------------
    # Remove invalid observations
    # --------------------------------------------------------

    data = data.dropna()

    # --------------------------------------------------------
    # Assign observations to bins
    # --------------------------------------------------------

    data["temperature_bin"] = pd.cut(
        data["temperature"],
        bins=temperature_bins,
        include_lowest=True
    )

    data["irradiance_bin"] = pd.cut(
        data["irradiance"],
        bins=irradiance_bins,
        include_lowest=True
    )

    # --------------------------------------------------------
    # Calculate mean value in each bin
    # --------------------------------------------------------

    matrix = data.pivot_table(
        index="irradiance_bin",
        columns="temperature_bin",
        values=column,
        aggfunc="mean",
        observed=False
    )

    # --------------------------------------------------------
    # Sort axes
    # --------------------------------------------------------

    matrix = matrix.sort_index(
        ascending=False
    )

    matrix = matrix.sort_index(
        axis=1
    )

    return matrix


# ============================================================
# Format heatmap labels
# ============================================================

def format_temperature_labels(
    matrix
):

    labels = []

    for interval in matrix.columns:

        midpoint = (
            interval.left
            + interval.right
        ) / 2

        labels.append(
            f"{midpoint:.1f}"
        )

    return labels


def format_irradiance_labels(
    matrix
):

    labels = []

    for interval in matrix.index:

        midpoint = (
            interval.left
            + interval.right
        ) / 2

        labels.append(
            f"{midpoint:.0f}"
        )

    return labels


# ============================================================
# Plot traditional heatmap
# ============================================================

def plot_heatmap(
    df,
    parameter,
    value_type,
    output_path,
    temperature_bins,
    irradiance_bins,
):
    """
    Plot a traditional matrix-style heatmap.

    Rows:
        Irradiance bins

    Columns:
        Temperature bins

    Cell:
        Mean parameter value
    """
    # if parameter == "Voc":
    #     # Use actual irradiance range from the dataframe
    #     G_min = df["irradiance"].min()
    #     G_max = df["irradiance"].max()
    #
    #     # Log-spaced irradiance bins ONLY for Voc
    #     irradiance_bins = np.logspace(
    #         np.log10(G_min),
    #         np.log10(G_max),
    #         len(irradiance_bins)
    #     )
    matrix = create_heatmap_matrix(
        df=df,
        parameter=parameter,
        value_type=value_type,
        temperature_bins=temperature_bins,
        irradiance_bins=irradiance_bins,
    )

    if matrix.empty:

        print(
            f"Skipping {parameter} "
            f"{value_type}: empty matrix."
        )

        return

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    temperature_labels = (
        format_temperature_labels(
            matrix
        )
    )

    irradiance_labels = (
        format_irradiance_labels(
            matrix
        )
    )

    matrix.columns = (
        temperature_labels
    )

    matrix.index = (
        irradiance_labels
    )

    # --------------------------------------------------------
    # Figure size
    # --------------------------------------------------------

    n_columns = len(
        matrix.columns
    )

    n_rows = len(
        matrix.index
    )

    width = max(
        8,
        n_columns * 0.65
    )

    height = max(
        6,
        n_rows * 0.55
    )

    fig, ax = plt.subplots(
        figsize=(width, height)
    )

    # ========================================================
    # True / Predicted
    # ========================================================

    if value_type in ["true", "pred"]:

        # ----------------------------------------------------
        # Special case: log scale for Voc only
        # ----------------------------------------------------


        sns.heatmap(
            matrix,
            ax=ax,
            cmap="magma",
            annot=True,
            fmt=".3g",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": get_units(parameter)},
        )


        # sns.heatmap(
        #     matrix,
        #     ax=ax,
        #     cmap="viridis",
        #     annot=True,
        #     fmt=".3g",
        #     linewidths=0.5,
        #     linecolor="white",
        #     cbar_kws={
        #         "label": get_units(
        #             parameter
        #         )
        #     },
        # )

        if value_type == "true":

            title = (
                f"{parameter} — True"
            )

        else:

            title = (
                f"{parameter} — Predicted"
            )

    # ========================================================
    # Error
    # ========================================================

    else:

        # ----------------------------------------------------
        # Symmetric scale around zero
        # ----------------------------------------------------

        finite_values = matrix.values[
            np.isfinite(
                matrix.values
            )
        ]

        if len(finite_values) > 0:

            vmax = np.max(
                np.abs(
                    finite_values
                )
            )

        else:

            vmax = 1.0

        if vmax == 0:

            vmax = 1.0

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".3g",
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 12},
            cbar_kws={
                "label": (
                    "Prediction error "
                    "(Predicted − True)"
                )
            },
        )

        title = (
            f"{parameter} — Prediction Error"
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")
    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Module Temperature (°C)"
    )

    ax.set_ylabel(
        "Irradiance (W/m²)"
    )

    ax.set_title(
        title
    )

    # --------------------------------------------------------
    # Tick rotation
    # --------------------------------------------------------

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Generate all heatmaps
# ============================================================

def generate_all_heatmaps(
    df,
    output_dir,
    temperature_bins,
    irradiance_bins,
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for parameter in ALL_PARAMS:

        parameter_dir = os.path.join(
            output_dir,
            parameter
        )

        os.makedirs(
            parameter_dir,
            exist_ok=True
        )

        print(
            f"\nGenerating heatmaps for "
            f"{parameter}..."
        )

        # ----------------------------------------------------
        # True
        # ----------------------------------------------------

        plot_heatmap(
            df=df,
            parameter=parameter,
            value_type="true",
            output_path=os.path.join(
                parameter_dir,
                "true_heatmap.png"
            ),
            temperature_bins=temperature_bins,
            irradiance_bins=irradiance_bins,
        )

        # ----------------------------------------------------
        # Predicted
        # ----------------------------------------------------

        plot_heatmap(
            df=df,
            parameter=parameter,
            value_type="pred",
            output_path=os.path.join(
                parameter_dir,
                "pred_heatmap.png"
            ),
            temperature_bins=temperature_bins,
            irradiance_bins=irradiance_bins,
        )

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        plot_heatmap(
            df=df,
            parameter=parameter,
            value_type="error",
            output_path=os.path.join(
                parameter_dir,
                "error_heatmap.png"
            ),
            temperature_bins=temperature_bins,
            irradiance_bins=irradiance_bins,
        )


# ============================================================
# Main analysis
# ============================================================

def run_heatmap_analysis(
    model_path,
    dataset_path,
    tech,
    output_root="outputs",
    n_temperature_bins=8,
    n_irradiance_bins=8,
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    base_dir = os.path.join(
        output_root,
        tech,
        "SEPIV3/temp_irr_heatmaps"
    )

    heatmap_dir = os.path.join(
        base_dir,
        "heatmaps"
    )

    data_dir = os.path.join(
        base_dir,
        "data"
    )

    os.makedirs(
        heatmap_dir,
        exist_ok=True
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    dataset = np.load(
        dataset_path,
        allow_pickle=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading model..."
    )

    model, V_grid = load_model(
        model_path,
        dataset
    )

    model.to(device)

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "Running inference..."
    )

    (
        weather,
        diode_true,
        diode_pred,
        electrical_true,
        electrical_pred,
    ) = generate_predictions(
        model,
        dataset,
        device
    )

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------

    df = build_dataframe(
        weather,
        diode_true,
        diode_pred,
        electrical_true,
        electrical_pred,
    )

    # --------------------------------------------------------
    # Save point-wise data
    # --------------------------------------------------------

    csv_path = os.path.join(
        data_dir,
        "heatmap_data.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nSaved point-wise data:"
        f"\n{csv_path}"
    )

    # ========================================================
    # Create bins
    # ========================================================

    temperature_min = (
        df["temperature"].min()
    )

    temperature_max = (
        df["temperature"].max()
    )

    irradiance_min = (
        df["irradiance"].min()
    )

    irradiance_max = (
        df["irradiance"].max()
    )

    # temperature_bins = np.linspace(
    #     temperature_min,
    #     temperature_max,
    #     n_temperature_bins + 1
    # )
    #
    # irradiance_bins = np.linspace(
    #     irradiance_min,
    #     irradiance_max,
    #     n_irradiance_bins + 1
    # )
    temperature_bins = np.linspace(
        TEMPERATURE_MIN,
        TEMPERATURE_MAX,
        n_temperature_bins + 1
    )

    irradiance_bins = np.linspace(
        IRRADIANCE_MIN,
        IRRADIANCE_MAX,
        n_irradiance_bins + 1
    )

    print(
        "\nTemperature bins:"
    )

    print(
        temperature_bins
    )

    print(
        "\nIrradiance bins:"
    )

    print(
        irradiance_bins
    )

    # --------------------------------------------------------
    # Save bin definitions
    # --------------------------------------------------------

    np.savez(
        os.path.join(
            data_dir,
            "heatmap_bins.npz"
        ),
        temperature_bins=temperature_bins,
        irradiance_bins=irradiance_bins,
    )

    heatmap_metrics = []

    for parameter in DIODE_PARAMS + ELECTRICAL_PARAMS:
        metrics = calculate_binned_metrics(
            df,
            parameter,
            n_bins=8,
            min_samples=5
        )

        heatmap_metrics.append(metrics)

    metrics_df = pd.DataFrame(heatmap_metrics)

    metrics_path = os.path.join(
        base_dir,
        "binned_heatmap_metrics.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)

    print("\nBinned heatmap metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved to: {metrics_path}")

    # ========================================================
    # Generate heatmaps
    # ========================================================

    generate_all_heatmaps(
        df=df,
        output_dir=heatmap_dir,
        temperature_bins=temperature_bins,
        irradiance_bins=irradiance_bins,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "Traditional heatmap analysis complete"
    )

    print(
        "=========================================="
    )

    print(
        f"\nOutput directory:"
        f"\n{base_dir}"
    )


# ============================================================
# Command line
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate traditional temperature-irradiance "
            "heatmaps for predicted and true "
            "diode/electrical parameters."
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained .pth model"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset .npz"
    )

    parser.add_argument(
        "--tech",
        type=str,
        required=True,
        help=(
            "Technology/device name, "
            "e.g. Perovskite_5&6"
        )
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root output directory"
    )

    parser.add_argument(
        "--n-temperature-bins",
        type=int,
        default=8,
        help=(
            "Number of temperature bins "
            "for the heatmap"
        )
    )

    parser.add_argument(
        "--n-irradiance-bins",
        type=int,
        default=8,
        help=(
            "Number of irradiance bins "
            "for the heatmap"
        )
    )

    args = parser.parse_args()

    run_heatmap_analysis(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        tech=args.tech,
        output_root=args.output_root,
        n_temperature_bins=(
            args.n_temperature_bins
        ),
        n_irradiance_bins=(
            args.n_irradiance_bins
        ),
    )