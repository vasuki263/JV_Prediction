import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. INPUT FILES
# ============================================================

panel_files = {
    "Perovskite_5(SEPIV1)":
        "outputs/Perovskite_5&6/SEPIV1/ae/eval_outputs_0_2_diode_loss_v2/metrics/predictions.csv",

    "Perovskite_6(SEPIV3)":
        "outputs/Perovskite_5&6/SEPIV3/ae/eval_outputs_0_2_diode_loss_v2/metrics/predictions.csv",

    "SunPower_5_1(SEPIV3)":
        "outputs/SunPower_5-1/ae/eval_outputs/metrics/predictions.csv"
}

# panel_files = {
#     "Perovskite":
#         "outputs/Perovskite_5&6/ae/eval_outputs_0_2_diode_loss/metrics/predictions.csv",
#
#     "SunPower_5_1":
#         "outputs/SunPower_5-1/ae/eval_outputs/metrics/predictions.csv"
# }

output_dir = Path("outputs/evaluation_plots_V2/jv_rmse_vs_conditions")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DEFINE COLUMN NAMES
# ============================================================

TEMP_COL = "Temperature"
IRRAD_COL = "Irradiance"
RMSE_COL = "Final_JV_RMSE"


# ============================================================
# 3. LOAD DATA
# ============================================================

panel_data = {}

for panel_name, file_path in panel_files.items():

    df = pd.read_csv(file_path)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    # Check required columns
    required_columns = [TEMP_COL, IRRAD_COL, RMSE_COL]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print(f"\nWARNING: {panel_name}")
        print("Missing columns:", missing_columns)
        print("Available columns:")
        print(df.columns.tolist())
        continue

    # Convert required columns to numeric
    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=required_columns
    )

    panel_data[panel_name] = df

    print(
        f"{panel_name}: "
        f"{len(df)} valid samples loaded"
    )


# ============================================================
# 4. FUNCTION:
#    SCATTER + BINNED MEAN
# ============================================================

def plot_rmse_vs_variable(
    variable_col,
    xlabel,
    filename,
    n_bins=15
):

    fig, ax = plt.subplots(figsize=(9, 6))

    # Different marker/color automatically assigned
    markers = ["o", "s", "^"]

    for i, (panel_name, df) in enumerate(panel_data.items()):

        x = df[variable_col].values
        y = df[RMSE_COL].values

        # ----------------------------------------------------
        # RAW SCATTER
        # ----------------------------------------------------
        ax.scatter(
            x,
            y,
            alpha=0.25,
            s=20,
            marker=markers[i % len(markers)]
        )

        # ----------------------------------------------------
        # BINNED MEAN
        # ----------------------------------------------------
        bins = np.linspace(
            x.min(),
            x.max(),
            n_bins + 1
        )

        bin_centers = (
            bins[:-1] + bins[1:]
        ) / 2

        bin_means = []

        for j in range(n_bins):

            if j == n_bins - 1:
                mask = (
                    (x >= bins[j]) &
                    (x <= bins[j + 1])
                )
            else:
                mask = (
                    (x >= bins[j]) &
                    (x < bins[j + 1])
                )

            if np.any(mask):
                bin_means.append(
                    np.mean(y[mask])
                )
            else:
                bin_means.append(
                    np.nan
                )

        # ----------------------------------------------------
        # PLOT BINNED MEAN
        # ----------------------------------------------------
        ax.plot(
            bin_centers,
            bin_means,
            marker=markers[i % len(markers)],
            linewidth=2,
            label=f"{panel_name} (binned mean)"
        )

    ax.set_xlabel(
        xlabel,
        fontsize=12
    )

    ax.set_ylabel(
        "Final J-V RMSE (A)",
        fontsize=12
    )

    ax.set_title(
        f"Final J-V RMSE vs. {xlabel}",
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
# 5. J-V RMSE vs TEMPERATURE
# ============================================================

plot_rmse_vs_variable(
    variable_col=TEMP_COL,
    xlabel="Module Temperature (°C)",
    filename="JV_RMSE_vs_Temperature.png",
    n_bins=15
)


# ============================================================
# 6. J-V RMSE vs IRRADIANCE
# ============================================================

plot_rmse_vs_variable(
    variable_col=IRRAD_COL,
    xlabel="Average Irradiance (W/m²)",
    filename="JV_RMSE_vs_Irradiance.png",
    n_bins=15
)


print("\nPlots saved to:")
print(output_dir.resolve())