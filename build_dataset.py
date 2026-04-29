# build_dataset.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
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
    print("DEBUG: number of samples from extract_all:", len(samples))

    X, Y = [], []
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

    print("DEBUG: about to load weather CSV")

    # Build cleaned weather file ONCE
    clean_weather_path = "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv"
    raw_weather_path = "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv"

    if not os.path.exists(clean_weather_path):
        build_jv_weather_pipeline(raw_weather_path, clean_weather_path)

    # Load cleaned weather
    weather = pd.read_csv(clean_weather_path, parse_dates=["timestamp"])
    weather = weather.set_index("timestamp").sort_index()

    print("DEBUG: weather final:", weather.shape)
    print("DEBUG: weather tz:", weather.index.tz)


    print("DEBUG: weather loaded:", weather.shape)
    print("DEBUG: weather tz:", weather.index.tz)



    all_temp_amb = weather["temp_C"].values.reshape(-1,1)
    all_humidity = weather["humidity"].values.reshape(-1,1)
    all_cloud = weather["cloud_cover"].values.reshape(-1,1)
    all_precip = weather["precip_mm"].values.reshape(-1,1)
    all_glob = weather["global_radiation"].values.reshape(-1,1)

    # Fit feature scalers
    all_v = np.concatenate([s["voltage"] for s in samples])
    all_i = np.concatenate([s["current"] for s in samples])
    all_t = np.array([s["temperature"] for s in samples]).reshape(-1,1)
    all_g = np.array([s["irradiance"] for s in samples]).reshape(-1,1)

    scalers["voltage"].fit(all_v.reshape(-1,1))
    scalers["current"].fit(all_i.reshape(-1,1))
    scalers["temperature"].fit(all_t)
    scalers["irradiance"].fit(all_g)

    scalers["temp_amb"].fit(all_temp_amb)
    scalers["humidity"].fit(all_humidity)
    scalers["cloud"].fit(all_cloud)
    scalers["precip"].fit(all_precip)
    scalers["glob"].fit(all_glob)

    # Fit target scaler (Voc, Jsc, Pmpp)
    targets = np.array([[s["Voc"], s["Jsc"], s["Pmpp"]] for s in samples])
    scalers["targets"].fit(targets)

    # ... inside build_dataset.py ...
    
    final_X = []
    final_Y = []

    for s in samples:
        v, i = downsample(s["voltage"], s["current"])
        v = scalers["voltage"].transform(v.reshape(-1,1))
        i = scalers["current"].transform(i.reshape(-1,1))

        sweep = np.linspace(0,1,len(v)).reshape(-1,1)

        # Get module-level data
        temp_mod = scalers["temperature"].transform([[s["temperature"]]])[0][0]
        irr_mod  = scalers["irradiance"].transform([[s["irradiance"]]])[0][0]

        # Match weather data (Ensure UTC)
        ts = pd.Timestamp(s["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
            
        # Optimization: Don't use .abs().argmin() inside the loop if performance is slow
        # But for 1000 samples it should be okay.
        idx = np.argmin(np.abs(weather.index - ts))
        w = weather.iloc[idx]

        # Scale weather features
        env_feats = np.array([
            w["temp_C"], w["humidity"], w["cloud_cover"], w["precip_mm"], w["global_radiation"]
        ]).reshape(1, -1)
        
        # Scale everything at once
        w_scaled = np.array([
            scalers["temp_amb"].transform([[w["temp_C"]]])[0][0],
            scalers["humidity"].transform([[w["humidity"]]])[0][0],
            scalers["cloud"].transform([[w["cloud_cover"]]])[0][0],
            scalers["precip"].transform([[w["precip_mm"]]])[0][0],
            scalers["glob"].transform([[w["global_radiation"]]])[0][0]
        ])

        # Create the 100x10 feature block for this sample
        # [v, i, sweep, temp_m, irr_m, temp_a, hum, cloud, precip, glob]
        sample_block = np.zeros((len(v), 10))
        sample_block[:, 0] = v.flatten()
        sample_block[:, 1] = i.flatten()
        sample_block[:, 2] = sweep.flatten()
        sample_block[:, 3] = temp_mod
        sample_block[:, 4] = irr_mod
        sample_block[:, 5:] = w_scaled  # Broadcasts to all 100 rows

        final_X.append(sample_block)
        final_Y.append(scalers["targets"].transform([[s["Voc"], s["Jsc"], s["Pmpp"]]])[0])

    # Convert to Tensor one-by-one to save RAM
    X = torch.from_numpy(np.stack(final_X)).float()
    Y = torch.from_numpy(np.stack(final_Y)).float()

    ds = TensorDataset(X,Y)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    return loader, scalers
