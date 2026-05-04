# hysteresis_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# ---------------------------------------------------------
# DIRECTORY HELPERS
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SMOOTH = True
GAUSSIAN_SIGMA = 4.0


# ---------------------------------------------------------
# SMOOTHING FUNCTION
# ---------------------------------------------------------
def smooth_gaussian(v, i, sigma=GAUSSIAN_SIGMA):
    if len(i) < 5:
        return v, i
    return v, gaussian_filter1d(i, sigma=sigma)


# ---------------------------------------------------------
# PAIRING FORWARD/BACKWARD SWEEPS
# ---------------------------------------------------------
def group_forward_backward(raw_list):
    pairs = {}
    for entry in raw_list:
        key = (entry["file"], entry["timestamp"])
        if key not in pairs:
            pairs[key] = {}
        pairs[key][entry["sweep"]] = entry
    return {k: v for k, v in pairs.items() if "forward" in v and "backward" in v}


# ---------------------------------------------------------
# HYSTERESIS METRICS
# ---------------------------------------------------------
def hysteresis_index_area(fwd, bwd):
    v_f, i_f = fwd["voltage_raw"], fwd["current_raw"]
    v_b, i_b = bwd["voltage_raw"], bwd["current_raw"]

    if SMOOTH:
        v_f, i_f = smooth_gaussian(v_f, i_f)
        v_b, i_b = smooth_gaussian(v_b, i_b)

    v_min = max(min(v_f), min(v_b))
    v_max = min(max(v_f), max(v_b))
    v_common = np.linspace(v_min, v_max, 300)

    i_f_interp = np.interp(v_common, v_f, i_f)
    i_b_interp = np.interp(v_common, v_b, i_b)

    num = np.trapz(np.abs(i_f_interp - i_b_interp), v_common)
    den = np.trapz((np.abs(i_f_interp) + np.abs(i_b_interp)) / 2, v_common)

    return num / den if den > 0 else np.nan


def delta_voc(fwd, bwd):
    return fwd["voltage_raw"][-1] - bwd["voltage_raw"][-1]


def delta_jsc(fwd, bwd):
    return fwd["current_raw"][0] - bwd["current_raw"][0]


def delta_mpp(fwd, bwd):
    p_f = np.max(fwd["voltage_raw"] * fwd["current_raw"])
    p_b = np.max(bwd["voltage_raw"] * bwd["current_raw"])
    return p_f - p_b


# ---------------------------------------------------------
# PLOTTING RAW/SMOOTHED HYSTERESIS
# ---------------------------------------------------------
def plot_hysteresis_pair(fwd, bwd, idx, base_plot_dir):
    v_f, i_f = fwd["voltage_raw"], fwd["current_raw"]
    v_b, i_b = bwd["voltage_raw"], bwd["current_raw"]

    if SMOOTH:
        v_f, i_f = smooth_gaussian(v_f, i_f)
        v_b, i_b = smooth_gaussian(v_b, i_b)

    ensure_dir(f"{base_plot_dir}/hysteresis_pairs")

    plt.figure(figsize=(6,5))
    plt.plot(v_f, i_f, label="Forward sweep", linewidth=2)
    plt.plot(v_b, i_b, label="Backward sweep", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Hysteresis JV Pair #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/hysteresis_pairs/hysteresis_pair_{idx}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------
# PREDICTED HYSTERESIS
# ---------------------------------------------------------
def reconstruct_predicted_jv(Voc, Jsc, Pmpp, n=300):
    V = np.linspace(0, Voc, n)
    FF = np.clip(Pmpp / (Voc * Jsc + 1e-9), 0.01, 0.85)
    Vmpp = FF * Voc
    Impp = FF * Jsc
    a = np.log(1 - Impp / Jsc + 1e-9) / np.log(Vmpp / Voc + 1e-9)
    a = np.clip(a, 1.2, 10)
    J = Jsc * (1 - (V / Voc)**a)
    return V, J


def plot_predicted_hysteresis(preds, idx_fwd, idx_bwd, idx, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/predicted_hysteresis")

    Voc_f, Jsc_f, Pmpp_f = preds[idx_fwd]
    Voc_b, Jsc_b, Pmpp_b = preds[idx_bwd]

    V_f, J_f = reconstruct_predicted_jv(Voc_f, Jsc_f, Pmpp_f)
    V_b, J_b = reconstruct_predicted_jv(Voc_b, Jsc_b, Pmpp_b)

    plt.figure(figsize=(6,5))
    plt.plot(V_f, J_f, label="Predicted Forward", linewidth=2)
    plt.plot(V_b, J_b, label="Predicted Backward", linewidth=2)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"Predicted Hysteresis #{idx}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/predicted_hysteresis/predicted_hysteresis_{idx}.png", dpi=150)
    plt.close()


def predicted_hysteresis_index(preds, idx_fwd, idx_bwd):
    Voc_f, Jsc_f, Pmpp_f = preds[idx_fwd]
    Voc_b, Jsc_b, Pmpp_b = preds[idx_bwd]

    V_f, J_f = reconstruct_predicted_jv(Voc_f, Jsc_f, Pmpp_f)
    V_b, J_b = reconstruct_predicted_jv(Voc_b, Jsc_b, Pmpp_b)

    v_min = max(V_f.min(), V_b.min())
    v_max = min(V_f.max(), V_b.max())
    v_common = np.linspace(v_min, v_max, 300)

    J_fi = np.interp(v_common, V_f, J_f)
    J_bi = np.interp(v_common, V_b, J_b)

    num = np.trapz(np.abs(J_fi - J_bi), v_common)
    den = np.trapz((np.abs(J_fi) + np.abs(J_bi)) / 2, v_common)

    return num / den if den > 0 else np.nan


# ---------------------------------------------------------
# WEATHER CORRELATION + VISUALIZATION + MODEL
# ---------------------------------------------------------
def plot_correlation_heatmap(corr, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/heatmaps")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Hysteresis–Weather Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/heatmaps/hysteresis_weather_heatmap.png", dpi=150)
    plt.close()


def plot_hysteresis_vs_weather(merged_df, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/weather_scatter")

    variables = ["temp_C", "humidity", "global_radiation", "cloud_cover", "precip_mm"]

    for var in variables:
        if var not in merged_df.columns:
            continue

        plt.figure(figsize=(6,5))
        sns.regplot(
            x=merged_df[var],
            y=merged_df["HI_area"],
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red", "linewidth": 2}
        )

        r = merged_df[[var, "HI_area"]].corr().iloc[0,1]

        plt.xlabel(var)
        plt.ylabel("Hysteresis Index (HI_area)")
        plt.title(f"HI vs {var}  (r = {r:.2f})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_plot_dir}/weather_scatter/HI_vs_{var}.png", dpi=150)
        plt.close()


def plot_multi_panel_weather(merged_df, base_plot_dir):
    ensure_dir(f"{base_plot_dir}/weather_multi_panel")

    variables = ["temp_C", "humidity", "global_radiation", "cloud_cover", "precip_mm"]
    n = len(variables)

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=True)

    for ax, var in zip(axes, variables):
        if var not in merged_df.columns:
            ax.set_visible(False)
            continue

        sns.regplot(
            x=merged_df[var],
            y=merged_df["HI_area"],
            ax=ax,
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red", "linewidth": 2}
        )

        r = merged_df[[var, "HI_area"]].corr().iloc[0,1]
        ax.set_title(f"{var}\nr={r:.2f}")
        ax.set_xlabel(var)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Hysteresis Index (HI_area)")

    plt.tight_layout()
    plt.savefig(f"{base_plot_dir}/weather_multi_panel/HI_vs_weather_multi_panel.png", dpi=150)
    plt.close()


def compute_lagged_correlations(merged_df, base_metric_dir, lag_minutes=30):
    lag = pd.Timedelta(minutes=lag_minutes)
    lagged = merged_df.copy()

    weather_cols = ["temp_C", "humidity", "global_radiation", "cloud_cover", "precip_mm"]

    for col in weather_cols:
        if col in lagged.columns:
            lagged[f"{col}_lagged"] = lagged[col].shift(freq=lag)

    numeric_only = lagged.select_dtypes(include=[np.number])
    corr_lagged = numeric_only.corr()

    ensure_dir(base_metric_dir)
    corr_lagged.to_csv(f"{base_metric_dir}/hysteresis_weather_lagged_correlation.csv")
    print("Saved hysteresis_weather_lagged_correlation.csv")

    return corr_lagged


def build_weather_hysteresis_model(merged_df, base_metric_dir):
    features = ["temp_C", "humidity", "global_radiation", "cloud_cover", "precip_mm"]

    df = merged_df.dropna(subset=["HI_area"])
    features = [f for f in features if f in df.columns]

    if len(features) == 0:
        print("No usable weather features found — skipping model.")
        return None, None

    X = df[features]
    y = df["HI_area"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Weather-driven hysteresis prediction R² = {score:.3f}")

    coef_df = pd.DataFrame({"feature": features, "coef": model.coef_})

    ensure_dir(base_metric_dir)
    coef_df.to_csv(f"{base_metric_dir}/weather_hysteresis_model_coefficients.csv", index=False)

    return model, score


# ---------------------------------------------------------
# WEATHER CORRELATION WRAPPER
# ---------------------------------------------------------
def correlate_hysteresis_with_weather(hyst_results, weather_df, device):
    base_plot_dir = f"outputs/{device}/plots"
    base_metric_dir = f"outputs/{device}/metrics"

    df = pd.DataFrame(hyst_results)
    df = df.set_index("timestamp").sort_index()

    df.index = df.index.tz_localize(None)
    weather_df.index = weather_df.index.tz_localize(None)

    weather_aligned = weather_df.reindex(df.index, method="nearest")
    merged = pd.concat([df, weather_aligned], axis=1)

    numeric_only = merged.select_dtypes(include=[np.number])
    corr = numeric_only.corr()

    ensure_dir(base_metric_dir)
    corr.to_csv(f"{base_metric_dir}/hysteresis_weather_correlation.csv")
    print("Saved hysteresis_weather_correlation.csv")

    plot_correlation_heatmap(corr, base_plot_dir)
    plot_hysteresis_vs_weather(merged, base_plot_dir)
    plot_multi_panel_weather(merged, base_plot_dir)
    compute_lagged_correlations(merged, base_metric_dir)
    build_weather_hysteresis_model(merged, base_metric_dir)

    return corr


# ---------------------------------------------------------
# MAIN HYSTERESIS ANALYSIS
# ---------------------------------------------------------
def analyze_hysteresis(raw_test, preds, device):
    base_plot_dir = f"outputs/{device}/plots"
    base_metric_dir = f"outputs/{device}/metrics"

    pairs = group_forward_backward(raw_test)
    print(f"Found {len(pairs)} forward/backward JV pairs")

    results = []

    for idx, ((file, ts), sweeps) in enumerate(pairs.items()):
        fwd = sweeps["forward"]
        bwd = sweeps["backward"]

        hi = hysteresis_index_area(fwd, bwd)
        dv = delta_voc(fwd, bwd)
        dj = delta_jsc(fwd, bwd)
        dm = delta_mpp(fwd, bwd)

        if preds is not None:
            idx_fwd = idx * 2
            idx_bwd = idx * 2 + 1

            plot_predicted_hysteresis(preds, idx_fwd, idx_bwd, idx, base_plot_dir)
            HI_pred = predicted_hysteresis_index(preds, idx_fwd, idx_bwd)
        else:
            HI_pred = None

        results.append({
            "file": file,
            "timestamp": ts,
            "HI_area": hi,
            "HI_pred": HI_pred,
            "(delta)Voc": dv,
            "(delta)Jsc": dj,
            "(delta)Pmpp": dm
        })

        plot_hysteresis_pair(fwd, bwd, idx, base_plot_dir)

    print("Saved smoothed hysteresis plots.")
    return results
