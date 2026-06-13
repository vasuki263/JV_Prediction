# build_dataset.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from extract_clean_jv import extract_all_from_influx
from two_diode_fit import fit_two_diode
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def nearest_timestamp(df, ts):
    idx = df.index.get_indexer([ts], method="nearest")[0]
    return df.iloc[idx]

def downsample(v, i, n=100):
    n0 = min(len(v), len(i))
    idx = np.linspace(0, n0 - 1, n).astype(int)
    return v[idx], i[idx]


# ---------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------
def build_dataset(tech, clean_weather_path, batch=32):

    # -----------------------------------------------------
    # 1. Extract JV samples
    # -----------------------------------------------------
    samples = extract_all_from_influx(tech)
    clean_samples = []

    for s in samples:

        Voc = float(s["Voc"])
        Jsc = float(s["Jsc"])
        Pmpp = float(s["Pmpp"])
        FF = Pmpp / (Voc * Jsc + 1e-12)


        if not (
                np.isfinite(Voc)
                and np.isfinite(Jsc)
                and np.isfinite(Pmpp)
        ):
            continue

        if Voc < 20:
            continue

        if Jsc <= 0:
            continue

        if Pmpp <= 0:
            continue

        # if FF < 0.3 or FF > 0.95:
        #     continue

        clean_samples.append(s)

    print(
        f"Removed {len(samples) - len(clean_samples)} invalid samples"
    )

    samples = clean_samples
    print("DEBUG: number of samples:", len(samples))

    if len(samples) == 0:
        raise RuntimeError(f"No JV samples found for tech={tech}")

    # -----------------------------------------------------
    # 2. Load & clean weather data
    # -----------------------------------------------------
    weather = (
        pd.read_csv(clean_weather_path, parse_dates=["timestamp"])
        .set_index("timestamp")
        .sort_index()
        .ffill()
        .bfill()
        .fillna(0)
    )

    # -----------------------------------------------------
    # 3. Prepare scalers
    # -----------------------------------------------------
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
        "y": MinMaxScaler(),  # Voc, Jsc, Pmpp
    }

    # scalers = {
    #     "voltage": StandardScaler(),
    #     "current": StandardScaler(),
    #     "temperature": StandardScaler(),
    #     "irradiance": StandardScaler(),
    #     "temp_amb": StandardScaler(),
    #     "humidity": StandardScaler(),
    #     "cloud": StandardScaler(),
    #     "precip": StandardScaler(),
    #     "glob": StandardScaler(),
    #     "y": StandardScaler(),  # Voc, Jsc, Pmpp
    # }


    # Fit JV scalers
    all_v = np.concatenate([s["voltage"] for s in samples])
    all_i = np.concatenate([s["current"] for s in samples])
    all_t = np.array([s["temperature"] for s in samples]).reshape(-1, 1)
    all_g = np.array([s["irradiance"] for s in samples]).reshape(-1, 1)

    scalers["voltage"].fit(all_v.reshape(-1, 1))
    scalers["current"].fit(all_i.reshape(-1, 1))
    scalers["temperature"].fit(all_t)
    scalers["irradiance"].fit(all_g)

    # Fit weather scalers
    # scalers["temp_amb"].fit(weather["temp_C"].values.reshape(-1, 1))
    # scalers["humidity"].fit(weather["humidity"].values.reshape(-1, 1))
    # scalers["cloud"].fit(weather["cloud_cover"].values.reshape(-1, 1))
    # scalers["precip"].fit(weather["precip_mm"].values.reshape(-1, 1))
    # scalers["glob"].fit(weather["global_radiation"].values.reshape(-1, 1))
    scalers["glob"] = None
    scalers["temp_amb"]=None
    scalers["humidity"]=None
    scalers["cloud"]=None
    scalers["precip"]=None

    # Fit target scaler
    targets = np.array([[s["Voc"], s["Jsc"], s["Pmpp"]]for s in samples])
    print("Voc :", targets[:, 0].min(), targets[:, 0].max())
    print("Jsc :", targets[:, 1].min(), targets[:, 1].max())
    print("Pmpp:", targets[:, 2].min(), targets[:, 2].max())
    scalers["y"].fit(np.nan_to_num(targets, nan=0.0))

    # -----------------------------------------------------
    # 4. Build feature blocks
    # -----------------------------------------------------
    final_X = []
    final_Y = []
    raw_curves = []

    for s in samples:
        v_raw = s["voltage"]
        i_raw = s["current"]

        # Two-diode fitting
        params = fit_two_diode(v_raw, i_raw)

        raw_curves.append({
            "file": s["file"],
            "timestamp": s["timestamp"],
            "voltage_raw": v_raw,
            "current_raw": i_raw,
            **params
        })

        # Downsample JV
        v, i = downsample(v_raw, i_raw)

        # Scale JV
        v_scaled = scalers["voltage"].transform(v.reshape(-1, 1))
        i_scaled = scalers["current"].transform(i.reshape(-1, 1))
        sweep = np.linspace(0, 1, len(v)).reshape(-1, 1)

        # Module conditions
        temp_mod = scalers["temperature"].transform([[s["temperature"]]])[0][0]
        irr_mod = scalers["irradiance"].transform([[s["irradiance"]]])[0][0]

        # Weather alignment
        ts = pd.Timestamp(s["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")

        w = nearest_timestamp(weather, ts)

        # w_scaled = np.array([
        #     scalers["temp_amb"].transform([[w["temp_C"]]])[0][0],
        #     scalers["humidity"].transform([[w["humidity"]]])[0][0],
        #     scalers["cloud"].transform([[w["cloud_cover"]]])[0][0],
        #     scalers["precip"].transform([[w["precip_mm"]]])[0][0],
        #     # scalers["glob"].transform([[w["global_radiation"]]])[0][0],
        #     0.0,
        # ])
        # Weather features disabled → fill with zeros
        w_scaled = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Build 10-feature block
        block = np.zeros((len(v), 10))
        block[:, 0] = v_scaled.flatten()
        block[:, 1] = i_scaled.flatten()
        block[:, 2] = sweep.flatten()
        block[:, 3] = temp_mod
        block[:, 4] = irr_mod
        block[:, 5:10] = w_scaled

        final_X.append(np.nan_to_num(block, nan=0.0))

        # Targets
        y = scalers["y"].transform([[s["Voc"],s["Jsc"],s["Pmpp"]]])[0]
        final_Y.append(np.nan_to_num(y, nan=0.0))

    # -----------------------------------------------------
    # 5. Convert to tensors
    # -----------------------------------------------------
    X = torch.from_numpy(np.stack(final_X)).float()
    Y = torch.from_numpy(np.stack(final_Y)).float()

    # -----------------------------------------------------
    # 6. Train/Val/Test split
    # -----------------------------------------------------
    X_train, X_temp, Y_train, Y_temp, raw_train, raw_temp = train_test_split(
        X, Y, raw_curves, test_size=0.30, random_state=42
    )

    X_val, X_test, Y_val, Y_test, raw_val, raw_test = train_test_split(
        X_temp, Y_temp, raw_temp, test_size=0.50, random_state=42
    )

    # -----------------------------------------------------
    # 7. DataLoaders
    # -----------------------------------------------------
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch, shuffle=False)
    # -----------------------------------------------------
    # 8. Save dataset for LSTM temperature analysis
    # -----------------------------------------------------
    np.savez(
        f"data/jv_dataset_{tech}.npz",
        X=X.numpy(),
        Y=Y.numpy(),
        scalers=scalers
    )
    print(f"[INFO] Saved LSTM dataset --> data/jv_dataset_{tech}.npz")


    return train_loader, val_loader, test_loader, scalers, raw_test
