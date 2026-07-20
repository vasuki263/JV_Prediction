import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------
csv_file = "outputs/SunPower_5-1/ae/physics/metrics/jv_predictions_with_diode_params.csv"
dataset_file = "data/jv_autoencoder_dataset_silicon.npz"

save_dir = "outputs/SunPower_5-1/ae/physics/trend_plots"
os.makedirs(save_dir, exist_ok=True)


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
df = pd.read_csv(csv_file)

data = np.load(dataset_file, allow_pickle=True)

weather = data["weather"]

# Change index if temperature is not the first weather feature
temperature = weather[:, 0]


# ----------------------------------------------------------
# Plot function
# ----------------------------------------------------------
def plot_temperature_trend(
        temperature,
        true_values,
        pred_values,
        ylabel,
        title,
        filename,
        bins=12,
):

    temperature = np.asarray(temperature)

    true_values = np.asarray(true_values)
    pred_values = np.asarray(pred_values)

    # remove NaNs
    mask = (
        np.isfinite(temperature)
        & np.isfinite(true_values)
        & np.isfinite(pred_values)
    )

    temperature = temperature[mask]
    true_values = true_values[mask]
    pred_values = pred_values[mask]

    # -----------------------------------
    # Temperature bins
    # -----------------------------------
    edges = np.linspace(
        temperature.min(),
        temperature.max(),
        bins + 1,
    )

    centers = (edges[:-1] + edges[1:]) / 2

    true_mean = []
    true_std = []

    pred_mean = []
    pred_std = []

    for i in range(bins):

        m = (
            (temperature >= edges[i])
            &
            (temperature < edges[i + 1])
        )

        if np.sum(m) == 0:

            true_mean.append(np.nan)
            true_std.append(np.nan)

            pred_mean.append(np.nan)
            pred_std.append(np.nan)

            continue

        true_mean.append(np.mean(true_values[m]))
        true_std.append(np.std(true_values[m]))

        pred_mean.append(np.mean(pred_values[m]))
        pred_std.append(np.std(pred_values[m]))

    true_mean = np.array(true_mean)
    true_std = np.array(true_std)

    pred_mean = np.array(pred_mean)
    pred_std = np.array(pred_std)

    # -----------------------------------
    # Trend lines
    # -----------------------------------
    m_true = np.isfinite(true_mean)
    coef_true = np.polyfit(
        centers[m_true],
        true_mean[m_true],
        1,
    )

    m_pred = np.isfinite(pred_mean)
    coef_pred = np.polyfit(
        centers[m_pred],
        pred_mean[m_pred],
        1,
    )

    xfit = np.linspace(
        temperature.min(),
        temperature.max(),
        200,
    )

    # -----------------------------------
    # Plot
    # -----------------------------------
    plt.figure(figsize=(7,5))

    plt.errorbar(
        centers,
        true_mean,
        yerr=true_std,
        fmt='o',
        color='orange',
        capsize=4,
        label='True'
    )

    plt.errorbar(
        centers,
        pred_mean,
        yerr=pred_std,
        fmt='o',
        color='purple',
        capsize=4,
        label='Predicted'
    )

    plt.plot(
        xfit,
        coef_true[0]*xfit + coef_true[1],
        '--',
        color='orange',
        linewidth=2,
        label=f"True slope = {coef_true[0]:.4f}"
    )

    plt.plot(
        xfit,
        coef_pred[0]*xfit + coef_pred[1],
        '--',
        color='purple',
        linewidth=2,
        label=f"Pred slope = {coef_pred[0]:.4f}"
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ----------------------------------------------------------
# All parameters
# ----------------------------------------------------------

plots = [

    ("Voc_true", "Voc_pred",
     "Voc (V)",
     "Open-Circuit Voltage vs Temperature",
     "Voc_vs_Temperature.png"),

    ("Jsc_true", "Jsc_pred",
     "Jsc (A)",
     "Short-Circuit Current vs Temperature",
     "Jsc_vs_Temperature.png"),

    ("Vmpp_true", "Vmpp_pred",
     "Vmpp (V)",
     "Maximum Power Voltage vs Temperature",
     "Vmpp_vs_Temperature.png"),

    ("Jmpp_true", "Jmpp_pred",
     "Jmpp (A)",
     "Maximum Power Current vs Temperature",
     "Jmpp_vs_Temperature.png"),

    ("Pmpp_true", "Pmpp_pred",
     "Pmpp (W)",
     "Maximum Power vs Temperature",
     "Pmpp_vs_Temperature.png"),

    ("FF_true", "FF_pred",
     "Fill Factor",
     "Fill Factor vs Temperature",
     "FF_vs_Temperature.png"),

    ("Rs_true", "Rs_pred",
     "Series Resistance (Ω)",
     "Series Resistance vs Temperature",
     "Rs_vs_Temperature.png"),

    ("Rsh_true", "Rsh_pred",
     "Shunt Resistance (Ω)",
     "Shunt Resistance vs Temperature",
     "Rsh_vs_Temperature.png"),

]

for true_col, pred_col, ylabel, title, fname in plots:

    print("Creating", fname)

    plot_temperature_trend(
        temperature,
        df[true_col].values,
        df[pred_col].values,
        ylabel,
        title,
        fname,
    )

print("Done.")