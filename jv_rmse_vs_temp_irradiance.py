import pandas as pd
import numpy as np
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

output_dir = Path("outputs/jv_rmse_vs_conditions")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DEFINE COLUMN NAMES
# ============================================================

TEMP_COL = "Temperature"
IRRAD_COL = "Irradiance"
RMSE_COL = "Final_JV_RMSE"
TRUE_CURRENT_COL = "Jsc_true"


# ============================================================
# 4. LOAD DATA
# ============================================================

panel_data = {}

for panel_name, file_path in panel_files.items():

    df = pd.read_csv(file_path)

    # Remove accidental spaces
    df.columns = df.columns.str.strip()

    required_columns = [
        TEMP_COL,
        IRRAD_COL,
        RMSE_COL,
        TRUE_CURRENT_COL
    ]

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

    # Convert to numeric
    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=required_columns
    )

    # --------------------------------------------------------
    # Relative J-V RMSE
    #
    # RMSE normalized by the true short-circuit current
    # --------------------------------------------------------

    df["Relative_JV_RMSE"] = (
            df[RMSE_COL]
            / np.abs(df[TRUE_CURRENT_COL])
            * 100
    )

    panel_data[panel_name] = df

    print(
        f"{panel_name}: "
        f"{len(df)} valid samples loaded"
    )


# ============================================================
# 5. FUNCTION:
#    SCATTER + COMMON BINNED MEAN
# ============================================================

def plot_rmse_vs_variable(
variable_col,
    xlabel,
    filename,
    y_col=RMSE_COL,
    ylabel="JV RMSE",
    n_bins=15
):

    fig, ax = plt.subplots(figsize=(9, 6))

    markers = ["o", "s", "^"]

    # --------------------------------------------------------
    # FIND GLOBAL RANGE ACROSS ALL PANELS
    # --------------------------------------------------------

    all_x = np.concatenate([
        df[variable_col].values
        for df in panel_data.values()
    ])

    global_min = np.min(all_x)
    global_max = np.max(all_x)

    # --------------------------------------------------------
    # CREATE COMMON BIN EDGES
    # --------------------------------------------------------

    bins = np.linspace(
        global_min,
        global_max,
        n_bins + 1
    )

    bin_centers = (
        bins[:-1] + bins[1:]
    ) / 2


    # ========================================================
    # PLOT EACH PANEL
    # ========================================================

    for i, (panel_name, df) in enumerate(panel_data.items()):

        x = df[variable_col].values
        # y = df[RMSE_COL].values
        y = df[y_col].values


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
        # BINNED MEAN USING COMMON BINS
        # ----------------------------------------------------

        bin_means = []

        for j in range(n_bins):

            # Last bin includes upper edge
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


            # Calculate mean RMSE
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
            label=panel_name
        )


    # ========================================================
    # FORMATTING
    # ========================================================

    ax.set_xlabel(
        xlabel,
        fontsize=12
    )

    # ax.set_ylabel(
    #     "JV RMSE",
    #     fontsize=12
    # )
    ax.set_ylabel(
        ylabel,
        fontsize=12
    )

    # ax.set_title(
    #     f"JV RMSE vs. {xlabel}",
    #     fontsize=14
    # )
    ax.set_title(
        f"{ylabel} vs. {xlabel}",
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
# 6. JV RMSE vs TEMPERATURE
# ============================================================

plot_rmse_vs_variable(
    variable_col=TEMP_COL,
    xlabel="Module Temperature (°C)",
    filename="JV_RMSE_vs_Temperature.png",
    n_bins=15
)


# ============================================================
# 7. JV RMSE vs IRRADIANCE
# ============================================================

plot_rmse_vs_variable(
    variable_col=IRRAD_COL,
    xlabel="Average Irradiance (W/m²)",
    filename="JV_RMSE_vs_Irradiance.png",
    n_bins=15
)

plot_rmse_vs_variable(
    variable_col=TEMP_COL,
    xlabel="Module Temperature (°C)",
    filename="Relative_JV_RMSE_vs_Temperature.png",
    y_col="Relative_JV_RMSE",
    ylabel="Relative JV RMSE (%)",
    n_bins=15
)

plot_rmse_vs_variable(
    variable_col=IRRAD_COL,
    xlabel="Average Irradiance (W/m²)",
    filename="Relative_JV_RMSE_vs_Irradiance.png",
    y_col="Relative_JV_RMSE",
    ylabel="Relative JV RMSE (%)",
    n_bins=15
)

print("\nPlots saved to:")
print(output_dir.resolve())