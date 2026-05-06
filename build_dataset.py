# build_dataset.py (NaN‑SAFE VERSION)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from extract_clean_jv import extract_all
import pandas as pd
import os

def downsample(v, i, n=100):
    n0 = min(len(v), len(i))
    idx = np.linspace(0, n0-1, n).astype(int)
    return v[idx], i[idx]

def build_dataset(directory, influx_csv="C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/perovskite_influx.csv", batch=32):

    # ---------------------------------------------------------
    # 1. Load JV samples
    # ---------------------------------------------------------
    samples = extract_all(directory)
    print("DEBUG: number of samples:", len(samples))

    # ---------------------------------------------------------
    # 2. Load InfluxDB data
    # ---------------------------------------------------------
    if not os.path.exists(influx_csv):
        raise FileNotFoundError(f"InfluxDB CSV not found: {influx_csv}")

    influx = pd.read_csv(influx_csv, parse_dates=["_time"])
    influx = influx.set_index("_time").sort_index()

    required_cols = ["Temp", "Irr", "I", "U", "P", "AmbHmd", "AmbTemp"]

    # Add missing columns
    for col in required_cols:
        if col not in influx.columns:
            influx[col] = np.nan

    # NaN‑safe: replace all NaN with 0
    influx = influx.ffill().bfill().fillna(0)

    # ---------------------------------------------------------
    # 3. Load weather data
    # ---------------------------------------------------------
    clean_weather_path = "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv"
    raw_weather_path = "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv"

    from jv_weather_pipeline import build_jv_weather_pipeline
    if not os.path.exists(clean_weather_path):
        build_jv_weather_pipeline(raw_weather_path, clean_weather_path)

    weather = pd.read_csv(clean_weather_path, parse_dates=["timestamp"])
    weather = weather.set_index("timestamp").sort_index()

    # NaN‑safe weather
    weather = weather.ffill().bfill().fillna(0)
    # ---------------------------------------------------------
    # 4. Fit scalers (NaN‑safe)
    # ---------------------------------------------------------
    scalers = {
        "voltage": MinMaxScaler(),
        "current": MinMaxScaler(),
        "temperature": MinMaxScaler(),
        "irradiance": MinMaxScaler(),
        "temp_amb": MinMaxScaler(),
        "humidity": MinMaxScaler(),
        "cloud": MinMaxScaler(),
        "precip": MinMaxScaler(),
        "glob": MinMaxScaler(),

        "influx_temp": MinMaxScaler(),
        "influx_irr": MinMaxScaler(),
        "influx_I": MinMaxScaler(),
        "influx_U": MinMaxScaler(),
        "influx_P": MinMaxScaler(),
        "influx_AmbHmd": MinMaxScaler(),
        "influx_AmbTemp": MinMaxScaler(),

        "targets": MinMaxScaler()
    }

    # JV scalers
    all_v = np.concatenate([s["voltage"] for s in samples])
    all_i = np.concatenate([s["current"] for s in samples])
    all_t = np.array([s["temperature"] for s in samples]).reshape(-1, 1)
    all_g = np.array([s["irradiance"] for s in samples]).reshape(-1, 1)

    scalers["voltage"].fit(all_v.reshape(-1, 1))
    scalers["current"].fit(all_i.reshape(-1, 1))
    scalers["temperature"].fit(all_t)
    scalers["irradiance"].fit(all_g)

    # Weather scalers
    scalers["temp_amb"].fit(weather["temp_C"].values.reshape(-1, 1))
    scalers["humidity"].fit(weather["humidity"].values.reshape(-1, 1))
    scalers["cloud"].fit(weather["cloud_cover"].values.reshape(-1, 1))
    scalers["precip"].fit(weather["precip_mm"].values.reshape(-1, 1))
    scalers["glob"].fit(weather["global_radiation"].values.reshape(-1, 1))

    # Influx scalers
    scalers["influx_temp"].fit(influx["Temp"].values.reshape(-1, 1))
    scalers["influx_irr"].fit(influx["Irr"].values.reshape(-1, 1))
    scalers["influx_I"].fit(influx["I"].values.reshape(-1, 1))
    scalers["influx_U"].fit(influx["U"].values.reshape(-1, 1))
    scalers["influx_P"].fit(influx["P"].values.reshape(-1, 1))
    scalers["influx_AmbHmd"].fit(influx["AmbHmd"].values.reshape(-1, 1))
    scalers["influx_AmbTemp"].fit(influx["AmbTemp"].values.reshape(-1, 1))

    # Target scalers
    targets = np.array([[s["Voc"], s["Jsc"], s["Pmpp"]] for s in samples])
    targets = np.nan_to_num(targets, nan=0.0)
    scalers["targets"].fit(targets)

    # ---------------------------------------------------------
    # 5. Build dataset
    # ---------------------------------------------------------
    final_X = []
    final_Y = []
    raw_curves = []

    for s in samples:

        # Store raw JV curve
        raw_curves.append({
            "file": s["file"],
            "sweep": s["sweep"],
            "timestamp": s["timestamp"],
            "voltage_raw": s["voltage"],
            "current_raw": s["current"],
            "temperature": s["temperature"],
            "irradiance": s["irradiance"]
        })

        # Downsample JV curve
        v_raw = s["voltage"]
        i_raw = s["current"]
        v, i = downsample(v_raw, i_raw)

        v = scalers["voltage"].transform(v.reshape(-1, 1))
        i = scalers["current"].transform(i.reshape(-1, 1))
        sweep = np.linspace(0, 1, len(v)).reshape(-1, 1)

        # JV metadata
        temp_mod = scalers["temperature"].transform([[s["temperature"]]])[0][0]
        irr_mod = scalers["irradiance"].transform([[s["irradiance"]]])[0][0]

        # Timestamp alignment
        ts = pd.Timestamp(s["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")

        # Weather alignment
        w = weather.iloc[np.argmin(np.abs(weather.index - ts))]
        w_scaled = np.array([
            scalers["temp_amb"].transform([[w["temp_C"]]])[0][0],
            scalers["humidity"].transform([[w["humidity"]]])[0][0],
            scalers["cloud"].transform([[w["cloud_cover"]]])[0][0],
            scalers["precip"].transform([[w["precip_mm"]]])[0][0],
            scalers["glob"].transform([[w["global_radiation"]]])[0][0]
        ])

        # Influx alignment
        inf = influx.iloc[np.argmin(np.abs(influx.index - ts))]

        influx_scaled = np.array([
            scalers["influx_temp"].transform([[inf["Temp"]]])[0][0],
            scalers["influx_irr"].transform([[inf["Irr"]]])[0][0],
            scalers["influx_I"].transform([[inf["I"]]])[0][0],
            scalers["influx_U"].transform([[inf["U"]]])[0][0],
            scalers["influx_P"].transform([[inf["P"]]])[0][0],
            scalers["influx_AmbHmd"].transform([[inf["AmbHmd"]]])[0][0],
            scalers["influx_AmbTemp"].transform([[inf["AmbTemp"]]])[0][0]
        ])

        # ---------------------------------------------------------
        # Final feature block (17 features)
        # ---------------------------------------------------------
        sample_block = np.zeros((len(v), 17))
        sample_block[:, 0] = v.flatten()
        sample_block[:, 1] = i.flatten()
        sample_block[:, 2] = sweep.flatten()
        sample_block[:, 3] = temp_mod
        sample_block[:, 4] = irr_mod
        sample_block[:, 5:10] = w_scaled
        sample_block[:, 10:17] = influx_scaled

        # NaN‑safe
        sample_block = np.nan_to_num(sample_block, nan=0.0)

        final_X.append(sample_block)

        y = scalers["targets"].transform([[s["Voc"], s["Jsc"], s["Pmpp"]]])[0]
        y = np.nan_to_num(y, nan=0.0)
        final_Y.append(y)

    # ---------------------------------------------------------
    # 6. Convert to tensors
    # ---------------------------------------------------------
    X = torch.from_numpy(np.stack(final_X)).float()
    Y = torch.from_numpy(np.stack(final_Y)).float()

    print("Any NaN in X:", torch.isnan(X).any().item())
    print("Any NaN in Y:", torch.isnan(Y).any().item())

    # ---------------------------------------------------------
    # 7. Train/val/test split
    # ---------------------------------------------------------
    X_train, X_temp, Y_train, Y_temp, raw_train, raw_temp = train_test_split(
        X, Y, raw_curves, test_size=0.30, random_state=42
    )

    X_val, X_test, Y_val, Y_test, raw_val, raw_test = train_test_split(
        X_temp, Y_temp, raw_temp, test_size=0.50, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch, shuffle=False)

    return train_loader, val_loader, test_loader, scalers, raw_test
