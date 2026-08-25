import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# 1. DEFINE CSV FILES
# ============================================================

# panel_files = {
#     "Perovskite_5(SEPIV1)": "outputs/Perovskite_5&6/SEPIV1/ae/eval_outputs_0_2_diode_loss/metrics/error_metrics.csv",
#     "Perovskite_6(SEPIV3)": "outputs/Perovskite_5&6/SEPIV3/ae/eval_outputs_0_2_diode_loss/metrics/error_metrics.csv",
#     "SunPower_5_1(SEPIV3)": "outputs/SunPower_5-1/ae/eval_outputs/metrics/error_metrics.csv"
# }

panel_files = {
    "Perovskite": "outputs/Perovskite_5&6/ae/eval_outputs_0_2_diode_loss/metrics/error_metrics.csv",
    "Silicon": "outputs/SunPower_5-1/ae/eval_outputs/metrics/error_metrics.csv"
}

output_dir = Path("outputs/evaluation_plots_all_modules_V2")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD CSV FILES
# ============================================================

panel_data = {}

for panel_name, file_path in panel_files.items():
    df = pd.read_csv(file_path)

    # Clean column names and parameter names
    df.columns = df.columns.str.strip()
    df["Parameter"] = df["Parameter"].astype(str).str.strip()

    panel_data[panel_name] = df


# ============================================================
# 3. SELECT PARAMETERS
# ============================================================

# ============================================================
# SELECT ONLY THE PARAMETERS TO DISPLAY
# ============================================================

parameters = [
    "Voc",
    "Jsc",
    # "Vmpp",
    # "Jmpp",
    "Pmpp",
    "FF",
    "Iph",
    "Io",
    # "n",
    "Rs",
    "Rsh"
]

panel_names = list(panel_data.keys())


# ============================================================
# 4. FUNCTION TO PLOT A METRIC ACROSS ALL PARAMETERS
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

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=panel_name
        )

        # Add numerical values above/below bars
        for bar, value in zip(bars, values):

            if np.isnan(value):
                continue

            # Position label above positive bars and below negative bars
            if value >= 0:
                y_pos = value
                va = "bottom"
                offset_y = 0.02
            else:
                y_pos = value
                va = "top"
                offset_y = -0.02

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + offset_y,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=8,

            )

    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} Comparison Across Parameters", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(parameters, rotation=45, ha="right")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 5. PLOT R² FOR ALL PARAMETERS
# ============================================================

plot_parameter_metric(
    metric="R2",
    ylabel=r"$R^2$",
    filename="R2_Comparison_All_Parameters.png",
    ylim=None
)


# ============================================================
# 6. PLOT PEARSON R FOR ALL PARAMETERS
# ============================================================

plot_parameter_metric(
    metric="PearsonR",
    ylabel="Pearson R",
    filename="PearsonR_Comparison_All_Parameters.png",
    ylim=None
)


# ============================================================
# 7. FINAL JV RMSE MEAN COMPARISON
# ============================================================

final_jv_rmse = {}

for panel_name, df in panel_data.items():

    # Locate Final JV Curve row
    row = df[df["Parameter"] == "Final JV RMSE mean"]

    if row.empty:
        print(f"Warning: 'Final JV RMSE mean' not found in {panel_name}")
        continue

    # Use RMSE value from Final JV Curve
    final_jv_rmse[panel_name] = float(row["MAE"].iloc[0])

# R2 for curve

curve_r2 = {}
for panel_name, df in panel_data.items():

    # Locate Final JV Curve row
    row = df[df["Parameter"] == "Final JV Curve"]

    if row.empty:
        print(f"Warning: 'Final JV Curve' not found in {panel_name}")
        continue

    # Use RMSE value from Final JV Curve
    curve_r2[panel_name] = float(row["R2"].iloc[0])

# Pearson R for curve

curve_pearsonR = {}
for panel_name, df in panel_data.items():

    # Locate Final JV Curve row
    row = df[df["Parameter"] == "Final JV Curve"]

    if row.empty:
        print(f"Warning: 'Final JV Curve' not found in {panel_name}")
        continue

    # Use RMSE value from Final JV Curve
    curve_pearsonR[panel_name] = float(row["PearsonR"].iloc[0])

# ============================================================
# 8. PLOT FINAL JV RMSE MEAN
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

rmse_panels = list(final_jv_rmse.keys())
rmse_values = list(final_jv_rmse.values())

# Define one color for each panel
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green"
]

bars = ax.bar(
    rmse_panels,
    rmse_values,
    color=colors
)

ax.set_xlabel("Panel", fontsize=12)
ax.set_ylabel("JV Curve RMSE", fontsize=12)
ax.set_title("JV Curve RMSE", fontsize=14)

ax.grid(axis="y", alpha=0.3)

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
print(f"\nAll plots saved in: {output_dir.resolve()}")


# R2
fig, ax = plt.subplots(figsize=(7, 5))

r2_panels = list(curve_r2.keys())
r2_values = list(curve_r2.values())

# Define one color for each panel
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green"
]

bars = ax.bar(
    r2_panels,
    r2_values,
    color=colors
)

ax.set_xlabel("Panel", fontsize=12)
ax.set_ylabel("JV Curve R2", fontsize=12)
ax.set_title("JV Curve R2 Comparison", fontsize=14)

ax.grid(axis="y", alpha=0.3)

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

# Pearson R
fig, ax = plt.subplots(figsize=(7, 5))

pearsonR_panels = list(curve_pearsonR.keys())
pearsonR_values = list(curve_pearsonR.values())

# Define one color for each panel
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green"
]

bars = ax.bar(
    pearsonR_panels,
    pearsonR_values,
    color=colors
)

ax.set_xlabel("Panel", fontsize=12)
ax.set_ylabel("JV Curve Pearson R", fontsize=12)
ax.set_title("JV Curve Pearson R Comparison", fontsize=14)

ax.grid(axis="y", alpha=0.3)

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
    output_dir / "JV Curve Pearson R Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()