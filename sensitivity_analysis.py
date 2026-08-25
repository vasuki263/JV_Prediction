import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model_autoencoder import PhysicsInformedJVModel
from jv_analysis import extract_jv_physics


# ============================================================
# CONFIGURATION
# ============================================================

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

ALL_PARAMS = DIODE_PARAMS + ELECTRICAL_PARAMS

PERTURBATION = 0.10


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path, dataset, device):

    V_grid = dataset["V_grid"].astype(np.float32)
    weather = dataset["weather"].astype(np.float32)

    ckpt = torch.load(
        model_path,
        map_location=device
    )

    model = PhysicsInformedJVModel(
        jv_dim=dataset["jv_target"].shape[1],
        weather_dim=weather.shape[1],
        voltage_grid=V_grid,
        latent_dim=ckpt["latent_dim"],
        diode_mean=ckpt["diode_mean"],
        diode_std=ckpt["diode_std"],
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, V_grid


# ============================================================
# PREDICT PARAMETERS
# ============================================================

def predict_parameters(
    model,
    weather,
    V_grid,
    device,
    area_m2=0.7,
):

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
    # Convert predicted JV curves to NumPy
    # --------------------------------------------------------

    I_pred = (
        I_pred
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Denormalize diode parameters
    # --------------------------------------------------------

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
        diode_pred_norm.detach().cpu().numpy()
        * diode_std
        + diode_mean
    )

    # --------------------------------------------------------
    # Extract electrical parameters from predicted JV
    # --------------------------------------------------------

    electrical_pred = []

    for i in range(len(I_pred)):

        values = extract_jv_physics(
            V_grid,
            I_pred[i],
            area_m2=area_m2
        )

        electrical_pred.append(values)

    electrical_pred = np.asarray(
        electrical_pred,
        dtype=np.float32
    )

    return diode_pred, electrical_pred


# ============================================================
# COMBINE PARAMETERS
# ============================================================

def combine_parameters(
    diode_params,
    electrical_params,
):

    return np.concatenate(
        [
            diode_params,
            electrical_params,
        ],
        axis=1
    )


# ============================================================
# CREATE ±10% WEATHER CONDITIONS
# ============================================================

def create_perturbed_weather(
    weather,
    input_name,
):

    weather_minus = weather.copy()
    weather_plus = weather.copy()

    if input_name == "temperature":

        # -10% temperature
        weather_minus[:, 0] = (
            weather[:, 0] * 0.90
        )

        # +10% temperature
        weather_plus[:, 0] = (
            weather[:, 0] * 1.10
        )

    elif input_name == "irradiance":

        # -10% irradiance
        weather_minus[:, 1] = (
            weather[:, 1] * 0.90
        )

        # +10% irradiance
        weather_plus[:, 1] = (
            weather[:, 1] * 1.10
        )

        # Prevent negative irradiance
        weather_minus[:, 1] = np.maximum(
            weather_minus[:, 1],
            0.0
        )

    else:

        raise ValueError(
            "input_name must be "
            "'temperature' or 'irradiance'"
        )

    return (
        weather_minus,
        weather_plus
    )


# ============================================================
# RUN ±10% ANALYSIS
# ============================================================

def run_plus_minus_analysis(
    model,
    weather,
    V_grid,
    device,
    input_name,
    area_m2=0.7,
):

    print(
        f"\nRunning ±10% {input_name} analysis..."
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    diode_baseline, electrical_baseline = (
        predict_parameters(
            model=model,
            weather=weather,
            V_grid=V_grid,
            device=device,
            area_m2=area_m2,
        )
    )

    # --------------------------------------------------------
    # Create ±10% conditions
    # --------------------------------------------------------

    weather_minus, weather_plus = (
        create_perturbed_weather(
            weather,
            input_name
        )
    )

    # --------------------------------------------------------
    # -10%
    # --------------------------------------------------------

    diode_minus, electrical_minus = (
        predict_parameters(
            model=model,
            weather=weather_minus,
            V_grid=V_grid,
            device=device,
            area_m2=area_m2,
        )
    )

    # --------------------------------------------------------
    # +10%
    # --------------------------------------------------------

    diode_plus, electrical_plus = (
        predict_parameters(
            model=model,
            weather=weather_plus,
            V_grid=V_grid,
            device=device,
            area_m2=area_m2,
        )
    )

    # --------------------------------------------------------
    # Combine diode + electrical parameters
    # --------------------------------------------------------

    baseline = combine_parameters(
        diode_baseline,
        electrical_baseline
    )

    minus = combine_parameters(
        diode_minus,
        electrical_minus
    )

    plus = combine_parameters(
        diode_plus,
        electrical_plus
    )

    return {
        "baseline": baseline,
        "minus": minus,
        "plus": plus,
        "weather_baseline": weather,
        "weather_minus": weather_minus,
        "weather_plus": weather_plus,
    }


# ============================================================
# CALCULATE MEAN PARAMETER VALUES
# ============================================================

def calculate_summary(
    results,
):

    baseline = results["baseline"]
    minus = results["minus"]
    plus = results["plus"]

    rows = []

    for j, parameter in enumerate(ALL_PARAMS):

        baseline_mean = np.nanmean(
            baseline[:, j]
        )

        minus_mean = np.nanmean(
            minus[:, j]
        )

        plus_mean = np.nanmean(
            plus[:, j]
        )

        # ----------------------------------------------------
        # Percentage changes relative to baseline
        # ----------------------------------------------------

        if np.abs(baseline_mean) > 1e-12:

            minus_change = (
                100.0
                * (minus_mean - baseline_mean)
                / np.abs(baseline_mean)
            )

            plus_change = (
                100.0
                * (plus_mean - baseline_mean)
                / np.abs(baseline_mean)
            )

        else:

            minus_change = np.nan
            plus_change = np.nan

        rows.append({

            "Parameter": parameter,

            "Baseline": baseline_mean,

            "Minus_10_percent": minus_mean,

            "Plus_10_percent": plus_mean,

            "Change_minus_10_percent":
                minus_change,

            "Change_plus_10_percent":
                plus_change,
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOT PARAMETER CHANGE
# ============================================================

def plot_parameter_change(
    summary_df,
    input_name,
    output_path,
):

    parameters = summary_df[
        "Parameter"
    ].values

    minus_values = summary_df[
        "Change_minus_10_percent"
    ].values

    plus_values = summary_df[
        "Change_plus_10_percent"
    ].values

    x = np.arange(
        len(parameters)
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------

    bars_minus = ax.bar(
        x - width / 2,
        minus_values,
        width,
        label="-10%"
    )

    bars_plus = ax.bar(
        x + width / 2,
        plus_values,
        width,
        label="+10%"
    )

    # --------------------------------------------------------
    # Zero reference
    # --------------------------------------------------------

    ax.axhline(
        0,
        linewidth=1
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Parameter"
    )

    ax.set_ylabel(
        "Change relative to baseline (%)"
    )

    ax.set_title(
        f"Predicted Parameter Change for "
        f"±10% {input_name.capitalize()}"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        parameters,
        rotation=45,
        ha="right"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    # --------------------------------------------------------
    # Numerical labels
    # --------------------------------------------------------

    for bar in (
        list(bars_minus)
        + list(bars_plus)
    ):

        height = bar.get_height()

        if np.isfinite(height):

            if height >= 0:

                va = "bottom"
                offset = 0.5

            else:

                va = "top"
                offset = -0.5

            ax.annotate(
                f"{height:.2f}%",
                (
                    bar.get_x()
                    + bar.get_width() / 2,
                    height
                ),
                xytext=(
                    0,
                    offset
                ),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=7,
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE SAMPLE-LEVEL DATA
# ============================================================

def save_sample_level_data(
    results,
    output_path,
):

    baseline = results["baseline"]
    minus = results["minus"]
    plus = results["plus"]

    rows = []

    for i in range(
        len(baseline)
    ):

        row = {
            "sample": i
        }

        for j, parameter in enumerate(
            ALL_PARAMS
        ):

            baseline_value = (
                baseline[i, j]
            )

            minus_value = (
                minus[i, j]
            )

            plus_value = (
                plus[i, j]
            )

            row[
                f"{parameter}_baseline"
            ] = baseline_value

            row[
                f"{parameter}_minus_10_percent"
            ] = minus_value

            row[
                f"{parameter}_plus_10_percent"
            ] = plus_value

            # ------------------------------------------------
            # Percentage change
            # ------------------------------------------------

            if (
                np.isfinite(baseline_value)
                and
                np.abs(baseline_value) > 1e-12
            ):

                row[
                    f"{parameter}_change_minus_percent"
                ] = (
                    100.0
                    * (
                        minus_value
                        - baseline_value
                    )
                    / np.abs(baseline_value)
                )

                row[
                    f"{parameter}_change_plus_percent"
                ] = (
                    100.0
                    * (
                        plus_value
                        - baseline_value
                    )
                    / np.abs(baseline_value)
                )

            else:

                row[
                    f"{parameter}_change_minus_percent"
                ] = np.nan

                row[
                    f"{parameter}_change_plus_percent"
                ] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def run_analysis(
    model_path,
    dataset_path,
    tech,
    output_root="outputs",
    area_m2=0.7,
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
    # Output directories
    # --------------------------------------------------------

    base_dir = os.path.join(
        output_root,
        tech,

        "sensitivity_analysis"
    )

    data_dir = os.path.join(
        base_dir,
        "data"
    )

    plot_dir = os.path.join(
        base_dir,
        "plots"
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    os.makedirs(
        plot_dir,
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

    weather = dataset[
        "weather"
    ].astype(np.float32)

    V_grid = dataset[
        "V_grid"
    ].astype(np.float32)

    print(
        f"Number of samples: "
        f"{len(weather)}"
    )

    print(
        f"Temperature range: "
        f"{weather[:, 0].min():.3f} "
        f"to "
        f"{weather[:, 0].max():.3f} °C"
    )

    print(
        f"Irradiance range: "
        f"{weather[:, 1].min():.3f} "
        f"to "
        f"{weather[:, 1].max():.3f} W/m²"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading model..."
    )

    model, V_grid = load_model(
        model_path=model_path,
        dataset=dataset,
        device=device,
    )

    # ========================================================
    # TEMPERATURE ±10%
    # ========================================================

    temperature_results = (
        run_plus_minus_analysis(
            model=model,
            weather=weather,
            V_grid=V_grid,
            device=device,
            input_name="temperature",
            area_m2=area_m2,
        )
    )

    temperature_summary = (
        calculate_summary(
            temperature_results
        )
    )

    # Save summary
    temperature_summary.to_csv(
        os.path.join(
            data_dir,
            "temperature_plus_minus_10_percent.csv"
        ),
        index=False
    )

    # Save sample-level data
    save_sample_level_data(
        temperature_results,
        os.path.join(
            data_dir,
            "temperature_sample_level.csv"
        )
    )

    # Plot
    plot_parameter_change(
        temperature_summary,
        input_name="temperature",
        output_path=os.path.join(
            plot_dir,
            "temperature_parameter_change_plus_minus_10_percent.png"
        )
    )

    # ========================================================
    # IRRADIANCE ±10%
    # ========================================================

    irradiance_results = (
        run_plus_minus_analysis(
            model=model,
            weather=weather,
            V_grid=V_grid,
            device=device,
            input_name="irradiance",
            area_m2=area_m2,
        )
    )

    irradiance_summary = (
        calculate_summary(
            irradiance_results
        )
    )

    # Save summary
    irradiance_summary.to_csv(
        os.path.join(
            data_dir,
            "irradiance_plus_minus_10_percent.csv"
        ),
        index=False
    )

    # Save sample-level data
    save_sample_level_data(
        irradiance_results,
        os.path.join(
            data_dir,
            "irradiance_sample_level.csv"
        )
    )

    # Plot
    plot_parameter_change(
        irradiance_summary,
        input_name="irradiance",
        output_path=os.path.join(
            plot_dir,
            "irradiance_parameter_change_plus_minus_10_percent.png"
        )
    )

    # ========================================================
    # COMBINED SUMMARY
    # ========================================================

    combined = pd.concat(
        [
            temperature_summary.assign(
                Input="Temperature"
            ),
            irradiance_summary.assign(
                Input="Irradiance"
            ),
        ],
        ignore_index=True
    )

    combined.to_csv(
        os.path.join(
            data_dir,
            "combined_plus_minus_10_percent_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "TEMPERATURE ±10% SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        temperature_summary.to_string(
            index=False
        )
    )

    print(
        "\n=============================================="
    )

    print(
        "IRRADIANCE ±10% SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        irradiance_summary.to_string(
            index=False
        )
    )

    print(
        "\n=============================================="
    )

    print(
        "Analysis complete"
    )

    print(
        "=============================================="
    )

    print(
        f"\nResults saved to:"
    )

    print(
        base_dir
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted diode and electrical "
            "parameters under ±10% temperature and "
            "irradiance perturbations."
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset (.npz)"
    )

    parser.add_argument(
        "--tech",
        type=str,
        required=True,
        help="Technology/panel name"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root output directory"
    )

    parser.add_argument(
        "--area-m2",
        type=float,
        default=0.7,
        help=(
            "Module area in m² used when extracting "
            "electrical parameters"
        )
    )

    args = parser.parse_args()

    run_analysis(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        tech=args.tech,
        output_root=args.output_root,
        area_m2=args.area_m2,
    )