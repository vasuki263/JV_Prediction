import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ============================================================
# 1. DEFINE CSV FILES
# ============================================================

panel_files = {
    "Perovskite_1":
        "outputs/Perovskite_5&6/SEPIV1/ae/eval_outputs_0_2_diode_loss_v2/metrics/error_metrics - Copy.csv",

    "Perovskite_2":
        "outputs/Perovskite_5&6/SEPIV3/ae/eval_outputs_0_2_diode_loss_v2/metrics/error_metrics - Copy.csv",

    "Silicon":
        "outputs/SunPower_5-1/ae/eval_outputs/metrics/error_metrics - Copy.csv"
}


# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

output_dir = Path("outputs/evaluation_plots_min_params")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 3. DEFINE CONSISTENT PANEL COLORS
# ============================================================
#
# Two different shades of blue for the two perovskite panels
# and green for silicon.
#

PANEL_COLORS = {
    "Perovskite_1": "#9ECAE1",   # Light blue
    "Perovskite_2": "#3182BD",   # Dark blue
    "Silicon": "#74A057"         # Green
}


# ============================================================
# 4. LOAD CSV FILES
# ============================================================

panel_data = {}

for panel_name, file_path in panel_files.items():

    df = pd.read_csv(file_path)

    # Clean column names and parameter names
    df.columns = df.columns.str.strip()
    df["Parameter"] = df["Parameter"].astype(str).str.strip()

    panel_data[panel_name] = df


panel_names = list(panel_data.keys())


# ============================================================
# 5. SELECT PARAMETERS
# ============================================================

parameters = [
    "Voc",
    "Jsc",
    # "Vmpp",
    # "Jmpp",
    "Pmpp",
    "FF",
    # "Iph",
    # "Io",
    # "n",
    "Rs",
    "Rsh"
]


# ============================================================
# 6. DISPLAY LABELS
# ============================================================

display_labels = {
    "Voc": "Voc",
    "Jsc": "Jsc",
    "Pmpp": "Pmpp",
    "FF": "FF",
    "Iph": "Iph",
    "Io": r"$I_0$",
    "Rs": "Rs",
    "Rsh": "Rsh"
}


# ============================================================
# 7. FUNCTION TO PLOT A METRIC ACROSS ALL PARAMETERS
# ============================================================

def plot_parameter_metric(metric, ylabel, filename, ylim=None):

    x = np.arange(len(parameters))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, panel_name in enumerate(panel_names):

        df = panel_data[panel_name]

        values = []

        for parameter in parameters:

            row = df[df["Parameter"] == parameter]

            if row.empty:
                print(f"Warning: '{parameter}' not found in {panel_name}")
                values.append(np.nan)

            else:
                values.append(float(row[metric].iloc[0]))

        # Position bars side-by-side
        offset = (i - (len(panel_names) - 1) / 2) * width

        # Get color for current panel
        color = PANEL_COLORS[panel_name]

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=panel_name,
            color=color
        )

        # Numerical labels
        labels = [
            f"{value:.3f}" if not np.isnan(value) else ""
            for value in values
        ]

        ax.bar_label(
            bars,
            labels=labels,
            padding=3,
            fontsize=8
        )

    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{ylabel} Comparison Across Parameters",
        fontsize=14
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [display_labels[p] for p in parameters],
        rotation=45,
        ha="right"
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 8. PLOT R² FOR ALL PARAMETERS
# ============================================================

plot_parameter_metric(
    metric="R2",
    ylabel=r"$R^2$",
    filename="R2_Comparison_All_Parameters.png",
    ylim=None
)


# ============================================================
# 9. PLOT PEARSON R FOR ALL PARAMETERS
# ============================================================

plot_parameter_metric(
    metric="PearsonR",
    ylabel="Pearson r",
    filename="PearsonR_Comparison_All_Parameters.png",
    ylim=None
)


# ============================================================
# 10. PLOT NRMSE FOR ALL PARAMETERS
# ============================================================

plot_parameter_metric(
    metric="NRMSE",
    ylabel="NRMSE",
    filename="NRMSE_Comparison_All_Parameters.png",
    ylim=None
)


# ============================================================
# 11. FINAL JV RMSE MEAN COMPARISON
# ============================================================

final_jv_rmse = {}

for panel_name, df in panel_data.items():

    # Locate Final JV Curve row
    row = df[df["Parameter"] == "Final JV RMSE mean"]

    if row.empty:
        print(
            f"Warning: 'Final JV RMSE mean' not found in {panel_name}"
        )
        continue

    # Use RMSE value from Final JV Curve
    final_jv_rmse[panel_name] = float(
        row["MAE"].iloc[0]
    )


# ============================================================
# 12. R² FOR FINAL JV CURVE
# ============================================================

curve_r2 = {}

for panel_name, df in panel_data.items():

    row = df[df["Parameter"] == "Final JV Curve"]

    if row.empty:
        print(
            f"Warning: 'Final JV Curve' not found in {panel_name}"
        )
        continue

    curve_r2[panel_name] = float(
        row["R2"].iloc[0]
    )


# ============================================================
# 13. PEARSON R FOR FINAL JV CURVE
# ============================================================

curve_pearsonR = {}

for panel_name, df in panel_data.items():

    row = df[df["Parameter"] == "Final JV Curve"]

    if row.empty:
        print(
            f"Warning: 'Final JV Curve' not found in {panel_name}"
        )
        continue

    curve_pearsonR[panel_name] = float(
        row["PearsonR"].iloc[0]
    )


# ============================================================
# 14. PLOT FINAL JV RMSE MEAN
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

rmse_panels = list(final_jv_rmse.keys())
rmse_values = list(final_jv_rmse.values())

# Use consistent panel colors
colors = [
    PANEL_COLORS[panel]
    for panel in rmse_panels
]

bars = ax.bar(
    rmse_panels,
    rmse_values,
    color=colors
)

ax.set_xlabel("Panel", fontsize=12)
ax.set_ylabel("JV Curve RMSE", fontsize=12)
ax.set_title("JV Curve RMSE", fontsize=14)

ax.grid(
    axis="y",
    alpha=0.3
)

# Add numerical values above bars
for bar, value in zip(bars, rmse_values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.5f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    output_dir / "Final_JV_RMSE_Mean_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 15. PLOT FINAL JV R²
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

r2_panels = list(curve_r2.keys())
r2_values = list(curve_r2.values())

# Use consistent panel colors
colors = [
    PANEL_COLORS[panel]
    for panel in r2_panels
]

bars = ax.bar(
    r2_panels,
    r2_values,
    color=colors
)

ax.set_xlabel("Panel", fontsize=12)
ax.set_ylabel("JV Curve R2", fontsize=12)
ax.set_title("JV Curve R2 Comparison", fontsize=14)

ax.grid(
    axis="y",
    alpha=0.3
)

# Add numerical values above bars
for bar, value in zip(bars, r2_values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.5f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    output_dir / "JV Curve R2 Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 16. PLOT FINAL JV PEARSON R
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

pearsonR_panels = list(curve_pearsonR.keys())
pearsonR_values = list(curve_pearsonR.values())

# Use consistent panel colors
colors = [
    PANEL_COLORS[panel]
    for panel in pearsonR_panels
]

bars = ax.bar(
    pearsonR_panels,
    pearsonR_values,
    color=colors
)

ax.set_xlabel("Panel", fontsize=12)
ax.set_ylabel("JV Curve Pearson r", fontsize=12)
ax.set_title("JV Curve Pearson r Comparison", fontsize=14)

ax.grid(
    axis="y",
    alpha=0.3
)

# Add numerical values above bars
for bar, value in zip(bars, pearsonR_values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.5f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    output_dir / "JV Curve Pearson r Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 17. PHYSICS RMSE VS FINAL JV RMSE
# ============================================================

physics_rmse = {}
final_rmse = {}

for panel_name, df in panel_data.items():

    # -----------------------------------------
    # Physics RMSE
    # -----------------------------------------

    physics_row = df[
        df["Parameter"] == "Physics JV RMSE mean"
    ]

    if physics_row.empty:

        print(
            f"Warning: 'Physics JV RMSE mean' "
            f"not found in {panel_name}"
        )

    else:

        physics_rmse[panel_name] = float(
            physics_row["MAE"].iloc[0]
        )

    # -----------------------------------------
    # Final RMSE
    # -----------------------------------------

    final_row = df[
        df["Parameter"] == "Final JV RMSE mean"
    ]

    if final_row.empty:

        print(
            f"Warning: 'Final JV RMSE mean' "
            f"not found in {panel_name}"
        )

    else:

        final_rmse[panel_name] = float(
            final_row["MAE"].iloc[0]
        )


# ============================================================
# 18. PLOT PHYSICS RMSE VS FINAL RMSE
# ============================================================

panels = [
    panel
    for panel in panel_names
    if panel in physics_rmse
    and panel in final_rmse
]

physics_values = [
    physics_rmse[panel]
    for panel in panels
]

final_values = [
    final_rmse[panel]
    for panel in panels
]

x = np.arange(len(panels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))


# ------------------------------------------------------------
# Physics model bars
# ------------------------------------------------------------

physics_bars = ax.bar(
    x - width / 2,
    physics_values,
    width,
    label="Single-Diode Model RMSE",
    color="#B0B0B0"
)


# ------------------------------------------------------------
# Final model bars
# ------------------------------------------------------------

final_bars = ax.bar(
    x + width / 2,
    final_values,
    width,
    label="Residual-Corrected RMSE",
    color="#3182BD"
)


# ============================================================
# 19. ADD VALUE LABELS
# ============================================================

ax.bar_label(
    physics_bars,
    labels=[
        f"{value:.5f}"
        for value in physics_values
    ],
    padding=3,
    fontsize=9
)

ax.bar_label(
    final_bars,
    labels=[
        f"{value:.5f}"
        for value in final_values
    ],
    padding=3,
    fontsize=9
)


# ============================================================
# 20. FORMATTING
# ============================================================

ax.set_xlabel("Panel", fontsize=12)
ax.set_ylabel("JV Curve RMSE", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(panels)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

# Leave space above bars for numerical labels
ax.margins(y=0.15)

plt.tight_layout()

plt.savefig(
    output_dir / "Physics_vs_Final_JV_RMSE.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 21. FINISHED
# ============================================================

print(
    f"\nAll plots saved in: "
    f"{output_dir.resolve()}"
)