import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from extract_clean_jv import extract_all_from_influx


def safe_interpolate_to_grid(V, I, V_grid):
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    # Remove NaN/inf from raw JV
    mask = np.isfinite(V) & np.isfinite(I)
    V, I = V[mask], I[mask]

    # If too few points → skip
    if len(V) < 5:
        return None

    # If voltage has duplicates → add tiny jitter
    if len(np.unique(V)) < len(V):
        V = V + np.random.normal(0, 1e-6, size=len(V))

    try:
        idx = np.argsort(V)
        f = interp1d(V[idx], I[idx], kind="linear", bounds_error=False, fill_value="extrapolate")
        I_grid = f(V_grid)
    except Exception:
        return None

    # Replace NaN/inf after interpolation
    if not np.all(np.isfinite(I_grid)):
        I_grid = pd.Series(I_grid).interpolate(limit_direction="both").to_numpy()

    # If still invalid → skip
    if not np.all(np.isfinite(I_grid)):
        return None

    return I_grid.astype(np.float32)


def safe_smooth_curve(y):
    y = np.asarray(y, dtype=float)

    # If still NaN/inf → skip
    if not np.all(np.isfinite(y)):
        return None

    # Window must be smaller than length
    window = min(11, len(y) - (len(y) % 2 == 0))
    if window < 5:
        window = 5

    try:
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
    except Exception:
        return None

    if not np.all(np.isfinite(y_smooth)):
        return None

    return y_smooth.astype(np.float32)



def build_jv_autoencoder_dataset(
    tech="PSK",
    weather_csv="data/weather_jv_cleaned.csv",
    out_file="data/jv_autoencoder_dataset.npz",
    num_points=200,
    V_min=None,
    V_max=None,
):
    print("Extracting JV samples from Influx/S3...")
    samples = extract_all_from_influx(tech)
    if not samples:
        print("No JV samples extracted.")
        return

    weather_df = pd.read_csv(weather_csv)
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    weather_df = weather_df.sort_values("timestamp").set_index("timestamp")

    weather_cols = [
        c for c in weather_df.columns
        if c != "timestamp" and np.issubdtype(weather_df[c].dtype, np.number)
    ]

    if V_min is None or V_max is None:
        all_v = np.concatenate([s["voltage"] for s in samples])
        V_min, V_max = float(all_v.min()), float(all_v.max())

    V_grid = np.linspace(V_min, V_max, num_points)

    jv_inputs, jv_targets, weather_inputs = [], [], []

    print("Building AE dataset...")
    for s in samples:
        V = np.asarray(s["voltage"], dtype=float)
        I = np.asarray(s["current"], dtype=float)
        ts = pd.Timestamp(s["timestamp"], tz="UTC")

        I_grid = safe_interpolate_to_grid(V, I, V_grid)
        if I_grid is None:
            continue

        # Clip extreme values
        I_grid = np.clip(I_grid, -100, 100)

        # Normalize
        max_val = np.max(np.abs(I_grid))
        if max_val > 0:
            I_grid = I_grid / max_val

        I_target = safe_smooth_curve(I_grid)
        if I_target is None:
            continue

        idx = weather_df.index.get_indexer([ts], method="nearest")[0]
        w = weather_df.iloc[idx][weather_cols].to_numpy(dtype=np.float32)

        jv_inputs.append(I_grid.astype(np.float32))
        jv_targets.append(I_target.astype(np.float32))
        weather_inputs.append(w)

    jv_inputs = np.stack(jv_inputs)
    jv_targets = np.stack(jv_targets)
    weather_inputs = np.stack(weather_inputs)

    np.savez_compressed(
        out_file,
        jv_input=jv_inputs,
        jv_target=jv_targets,
        weather=weather_inputs,
        V_grid=V_grid.astype(np.float32),
        weather_cols=np.array(weather_cols),
    )

    print(f"Saved AE dataset to {out_file}")
    print("Shapes:")
    print("  jv_input :", jv_inputs.shape)
    print("  jv_target:", jv_targets.shape)
    print("  weather  :", weather_inputs.shape)


if __name__ == "__main__":
    build_jv_autoencoder_dataset()
