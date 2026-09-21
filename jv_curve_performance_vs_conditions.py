import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. INPUT FILES
# ============================================================

panel_files = {
    "Perovskite_1":
        "outputs/Perovskite_5&6/SEPIV1/ae/eval_outputs_0_2_diode_loss_v2/metrics/predictions.csv",

    "Perovskite_2":
        "outputs/Perovskite_5&6/SEPIV3/ae/eval_outputs_0_2_diode_loss_v2/metrics/predictions.csv",

    "Silicon":
        "outputs/SunPower_5-1/ae/eval_outputs/metrics/predictions.csv"
}


# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

output_dir = Path("outputs/jv_rmse_grouped_conditions")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. COLUMN NAMES
# ============================================================

TEMPERATURE_COL = "Temperature"
IRRADIANCE_COL = "Irradiance"
RMSE_COL = "Final_JV_RMSE"


# ============================================================
# 4. LOAD EACH PANEL SEPARATELY
# ============================================================

panel_data = {}

for panel_name, file_path in panel_files.items():

    print("\n" + "=" * 80)
    print(f"Loading {panel_name}")
    print("=" * 80)

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"WARNING: File not found:")
        print(file_path)
        continue

    df = pd.read_csv(file_path)

    # Remove whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"Rows loaded: {len(df)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        TEMPERATURE_COL,
        IRRADIANCE_COL,
        RMSE_COL
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("WARNING: Missing columns:")
        print(missing_columns)

        print("\nAvailable columns:")
        print(df.columns.tolist())

        continue

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    )

    # Remove negative RMSE values, if any
    df = df[df[RMSE_COL] >= 0]

    panel_data[panel_name] = df

    print(f"Valid rows: {len(df)}")

    print(
        f"Temperature range: "
        f"{df[TEMPERATURE_COL].min():.2f} – "
        f"{df[TEMPERATURE_COL].max():.2f} °C"
    )

    print(
        f"Irradiance range: "
        f"{df[IRRADIANCE_COL].min():.2f} – "
        f"{df[IRRADIANCE_COL].max():.2f} W/m²"
    )

    print(
        f"Overall J-V RMSE: "
        f"{df[RMSE_COL].mean():.5f}"
    )


# ============================================================
# 5. GROUPED RMSE FUNCTION
# ============================================================

def calculate_grouped_rmse(
    df,
    condition_col,
    bin_width
):
    """
    Calculate grouped J-V RMSE statistics.

    For each temperature or irradiance group:

        N
        Mean RMSE
        Median RMSE
        RMS of curve-level RMSE
        Standard deviation
        Minimum RMSE
        Maximum RMSE
    """

    data = df.copy()

    # --------------------------------------------------------
    # Create groups
    # --------------------------------------------------------

    data["Condition_Group"] = (
        np.floor(
            data[condition_col] / bin_width
        ) * bin_width
    )

    results = []

    # --------------------------------------------------------
    # Calculate statistics for each group
    # --------------------------------------------------------

    for group, group_df in data.groupby(
        "Condition_Group"
    ):

        rmse = group_df[RMSE_COL].to_numpy(
            dtype=float
        )

        if len(rmse) == 0:
            continue

        # Mean RMSE
        mean_rmse = np.mean(rmse)

        # Median RMSE
        median_rmse = np.median(rmse)

        # RMS of curve-level RMSEs
        rms_rmse = np.sqrt(
            np.mean(rmse ** 2)
        )

        # Standard deviation
        std_rmse = np.std(
            rmse,
            ddof=1
        ) if len(rmse) > 1 else np.nan

        results.append({
            "Condition_Group": group,
            "N": len(rmse),
            "Mean_RMSE": mean_rmse,
            "Median_RMSE": median_rmse,
            "RMS_RMSE": rms_rmse,
            "Std_RMSE": std_rmse,
            "Min_RMSE": np.min(rmse),
            "Max_RMSE": np.max(rmse)
        })

    return pd.DataFrame(results).sort_values(
        "Condition_Group"
    )


# ============================================================
# 6. TEMPERATURE-GROUPED RMSE
# ============================================================

temperature_results = {}

for panel_name, df in panel_data.items():

    print("\n" + "=" * 80)
    print(f"TEMPERATURE-GROUPED J-V RMSE")
    print(panel_name)
    print("=" * 80)

    result = calculate_grouped_rmse(
        df=df,
        condition_col=TEMPERATURE_COL,
        bin_width=5
    )

    temperature_results[panel_name] = result

    print(result.to_string(index=False))

    # Save
    output_file = (
        output_dir /
        f"{panel_name.replace(' ', '_')}_"
        f"J-V_RMSE_vs_Temperature.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved: {output_file}")


# ============================================================
# 7. IRRADIANCE-GROUPED RMSE
# ============================================================

irradiance_results = {}

for panel_name, df in panel_data.items():

    print("\n" + "=" * 80)
    print(f"IRRADIANCE-GROUPED J-V RMSE")
    print(panel_name)
    print("=" * 80)

    result = calculate_grouped_rmse(
        df=df,
        condition_col=IRRADIANCE_COL,
        bin_width=100
    )

    irradiance_results[panel_name] = result

    print(result.to_string(index=False))

    # Save
    output_file = (
        output_dir /
        f"{panel_name.replace(' ', '_')}_"
        f"J-V_RMSE_vs_Irradiance.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved: {output_file}")


# ============================================================
# 8. PLOT FUNCTION
# ============================================================

def plot_grouped_rmse(
    grouped_results,
    x_label,
    filename,
    title
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    markers = ["o", "s", "^"]

    for i, (panel_name, result) in enumerate(
        grouped_results.items()
    ):

        if result.empty:
            continue

        ax.plot(
            result["Condition_Group"],
            result["Mean_RMSE"],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=6,
            label=panel_name
        )

    ax.set_xlabel(
        x_label,
        fontsize=12
    )

    ax.set_ylabel(
        "Mean J-V RMSE",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=14
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# 9. TEMPERATURE PLOT
# ============================================================

plot_grouped_rmse(
    grouped_results=temperature_results,
    x_label="Module Temperature (°C)",
    filename="J-V_RMSE_vs_Temperature_Grouped.png",
    title="J-V Prediction RMSE vs. Module Temperature"
)


# ============================================================
# 10. IRRADIANCE PLOT
# ============================================================

plot_grouped_rmse(
    grouped_results=irradiance_results,
    x_label="Average Irradiance (W/m²)",
    filename="J-V_RMSE_vs_Irradiance_Grouped.png",
    title="J-V Prediction RMSE vs. Average Irradiance"
)


# ============================================================
# 11. OPTIONAL: PLOT RMS OF CURVE-LEVEL RMSE
# ============================================================

def plot_grouped_rms_rmse(
    grouped_results,
    x_label,
    filename,
    title
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    markers = ["o", "s", "^"]

    for i, (panel_name, result) in enumerate(
        grouped_results.items()
    ):

        if result.empty:
            continue

        ax.plot(
            result["Condition_Group"],
            result["RMS_RMSE"],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=6,
            label=panel_name
        )

    ax.set_xlabel(
        x_label,
        fontsize=12
    )

    ax.set_ylabel(
        "RMS of curve-level J-V RMSE",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=14
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


plot_grouped_rms_rmse(
    grouped_results=temperature_results,
    x_label="Module Temperature (°C)",
    filename="J-V_RMS_RMSE_vs_Temperature_Grouped.png",
    title="RMS J-V RMSE vs. Module Temperature"
)

plot_grouped_rms_rmse(
    grouped_results=irradiance_results,
    x_label="Average Irradiance (W/m²)",
    filename="J-V_RMS_RMSE_vs_Irradiance_Grouped.png",
    title="RMS J-V RMSE vs. Average Irradiance"
)


# ============================================================
# 12. FINISHED
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(output_dir.resolve())

print("\nGenerated:")
print("  1. Temperature-grouped RMSE CSVs")
print("  2. Irradiance-grouped RMSE CSVs")
print("  3. Mean J-V RMSE vs temperature")
print("  4. Mean J-V RMSE vs irradiance")
print("  5. RMS J-V RMSE vs temperature")
print("  6. RMS J-V RMSE vs irradiance")