# build_dataset.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from extract_clean_jv import extract_all
import pandas as pd
from jv_weather_pipeline import build_jv_weather_pipeline
import os


def downsample(v, i, n=100):
    n0 = min(len(v), len(i))
    idx = np.linspace(0, n0-1, n).astype(int)
    return v[idx], i[idx]


def build_dataset(directory, batch=32):
    samples = extract_all(directory)
    print("DEBUG: number of samples:", len(samples))

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
        "targets": MinMaxScaler()
    }

    clean_weather_path = "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv"
    raw_weather_path = "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv"

    if not os.path.exists(clean_weather_path):
        build_jv_weather_pipeline(raw_weather_path, clean_weather_path)

    weather = pd.read_csv(clean_weather_path, parse_dates=["timestamp"])
    weather = weather.set_index("timestamp").sort_index()

    # Fit scalers
    all_v = np.concatenate([s["voltage"] for s in samples])
    all_i = np.concatenate([s["current"] for s in samples])
    all_t = np.array([s["temperature"] for s in samples]).reshape(-1, 1)
    all_g = np.array([s["irradiance"] for s in samples]).reshape(-1, 1)

    scalers["voltage"].fit(all_v.reshape(-1, 1))
    scalers["current"].fit(all_i.reshape(-1, 1))
    scalers["temperature"].fit(all_t)
    scalers["irradiance"].fit(all_g)

    scalers["temp_amb"].fit(weather["temp_C"].values.reshape(-1, 1))
    scalers["humidity"].fit(weather["humidity"].values.reshape(-1, 1))
    scalers["cloud"].fit(weather["cloud_cover"].values.reshape(-1, 1))
    scalers["precip"].fit(weather["precip_mm"].values.reshape(-1, 1))
    scalers["glob"].fit(weather["global_radiation"].values.reshape(-1, 1))

    targets = np.array([[s["Voc"], s["Jsc"], s["Pmpp"]] for s in samples])
    scalers["targets"].fit(targets)

    final_X = []
    final_Y = []
    raw_curves = []   # <-- store raw JV curves

    for s in samples:
        v_raw = s["voltage"]
        i_raw = s["current"]
        raw_curves.append((v_raw, i_raw))

        v, i = downsample(v_raw, i_raw)
        v = scalers["voltage"].transform(v.reshape(-1, 1))
        i = scalers["current"].transform(i.reshape(-1, 1))

        sweep = np.linspace(0, 1, len(v)).reshape(-1, 1)

        temp_mod = scalers["temperature"].transform([[s["temperature"]]])[0][0]
        irr_mod = scalers["irradiance"].transform([[s["irradiance"]]])[0][0]

        ts = pd.Timestamp(s["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")

        idx = np.argmin(np.abs(weather.index - ts))
        w = weather.iloc[idx]

        w_scaled = np.array([
            scalers["temp_amb"].transform([[w["temp_C"]]])[0][0],
            scalers["humidity"].transform([[w["humidity"]]])[0][0],
            scalers["cloud"].transform([[w["cloud_cover"]]])[0][0],
            scalers["precip"].transform([[w["precip_mm"]]])[0][0],
            scalers["glob"].transform([[w["global_radiation"]]])[0][0]
        ])

        sample_block = np.zeros((len(v), 10))
        sample_block[:, 0] = v.flatten()
        sample_block[:, 1] = i.flatten()
        sample_block[:, 2] = sweep.flatten()
        sample_block[:, 3] = temp_mod
        sample_block[:, 4] = irr_mod
        sample_block[:, 5:] = w_scaled

        final_X.append(sample_block)
        final_Y.append(scalers["targets"].transform([[s["Voc"], s["Jsc"], s["Pmpp"]]])[0])

    X = torch.from_numpy(np.stack(final_X)).float()
    Y = torch.from_numpy(np.stack(final_Y)).float()

    # -----------------------------
    # TRAIN / VAL / TEST SPLIT
    # -----------------------------
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
