import pandas as pd
import numpy as np
import os


def load_raw_weather(csv_path):
    print(f"--- MANUAL LOAD START: {csv_path} ---")

    target_cols = [
        "timestamp",
        "temp_C",
        "humidity",
        "cloud_cover",
        "precip_mm",
        "global_radiation",
    ]

    # -----------------------------------------------------
    # 1. Load CSV
    # -----------------------------------------------------
    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip",
            usecols=target_cols,
        )
    except Exception as e:
        print(f"CRITICAL ERROR during read_csv: {e}")
        # Fallback: load a small sample to inspect columns
        fallback = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", nrows=100)
        print("Columns actually found in file:", fallback.columns.tolist())
        raise e

    # -----------------------------------------------------
    # 2. Basic cleaning
    # -----------------------------------------------------
    df.columns = df.columns.str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # -----------------------------------------------------
    # 3. Timezone normalization
    # -----------------------------------------------------
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    df = df.set_index("timestamp").sort_index()

    print(f"--- MANUAL LOAD SUCCESS: {df.shape} rows ---")
    return df


# ---------------------------------------------------------
# WEATHER PIPELINE
# ---------------------------------------------------------
def build_jv_weather_pipeline(raw_path, clean_path):
    """
    Loads raw weather CSV, resamples to 10-min, interpolates,
    and saves a cleaned version. If cleaned file exists, loads it.
    """

    # -----------------------------------------------------
    # 1. Use existing cleaned file if available
    # -----------------------------------------------------
    if os.path.exists(clean_path):
        print(f"Using existing cleaned weather: {clean_path}")
        df = pd.read_csv(clean_path, index_col=0, parse_dates=True)

        # Ensure index is timezone-aware UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        return df

    # -----------------------------------------------------
    # 2. Load raw weather safely
    # -----------------------------------------------------
    df_raw = load_raw_weather(raw_path)

    # -----------------------------------------------------
    # 3. Resample to 10-min and interpolate
    # -----------------------------------------------------
    df_clean = (
        df_raw.resample("10min")
        .mean()
        .interpolate(method="linear")
    )

    # -----------------------------------------------------
    # 4. Save cleaned file
    # -----------------------------------------------------
    df_clean.to_csv(clean_path)
    return df_clean
