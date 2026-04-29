# preprocess_jv.py

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

from train_lstm_model import MultiOutputDataset, split_train_val_from_sequences


# ============================================================
# 1. QUALITY-CONTROL JV LOADER
# ============================================================

MAX_IRRADIANCE_VARIATION = 45
MAX_IRRADIANCE_DIFFERENCE_FWD_BWD = 30
TRIM_START = 5
TRIM_END = 5


def extract_temperature_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            import re
            match = re.search(r'(\d+\.?\d*)\s*°?C', first_line)
            if match:
                return float(match.group(1))
    except:
        pass
    return None


def load_jv_data(filepath):
    try:
        temp = extract_temperature_from_file(filepath)
        if temp is None:
            return None

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            f.readline()
            data = np.genfromtxt(f, delimiter=';', skip_header=0)

        if data.size == 0 or data.ndim != 2 or data.shape[1] < 5:
            return None

        voltage = data[:, 2]
        current = data[:, 3]
        irradiance = data[:, 4]

        sep = min(899, len(voltage) // 2)
        irr_fwd = irradiance[:sep]
        irr_bwd = irradiance[sep:]

        if len(irr_fwd) < 100 or len(irr_bwd) < 100:
            return None
        if np.max(irr_fwd) - np.min(irr_fwd) > MAX_IRRADIANCE_VARIATION:
            return None
        if np.max(irr_bwd) - np.min(irr_bwd) > MAX_IRRADIANCE_VARIATION:
            return None
        if abs(np.mean(irr_fwd) - np.mean(irr_bwd)) > MAX_IRRADIANCE_DIFFERENCE_FWD_BWD:
            return None

        v_fwd = voltage[:sep]
        i_fwd = current[:sep]

        if len(v_fwd) > (TRIM_START + TRIM_END):
            v_fwd = v_fwd[TRIM_START:-TRIM_END]
            i_fwd = i_fwd[TRIM_START:-TRIM_END]

        return {
            "voltage_fwd": v_fwd,
            "current_fwd": i_fwd,
            "irradiance": float(np.mean(irr_fwd)),
            "temperature": temp,
            "file": filepath
        }

    except:
        return None


# ============================================================
# 2. ROBUST DOWNSAMPLING
# ============================================================

def downsample_curve(voltage, current, n_points=100):
    n = min(len(voltage), len(current))
    voltage = voltage[:n]
    current = current[:n]
    idx = np.linspace(0, n - 1, n_points).astype(int)
    return voltage[idx], current[idx]


# ============================================================
# 3. BUILD CLEAN JV DATASET
# ============================================================

def build_clean_jv_dataset(root_folder, n_points=100):
    root = Path(root_folder)
    all_files = [f for f in root.rglob("*.txt") if "SEPIV" in f.name.upper()]

    print(f"Found {len(all_files)} JV files")

    curves = []
    for f in all_files:
        data = load_jv_data(f)
        if data is None:
            continue

        v_ds, i_ds = downsample_curve(data["voltage_fwd"], data["current_fwd"], n_points)

        curves.append({
            "file": str(f),
            "irradiance": data["irradiance"],
            "temperature": data["temperature"],
            "voltage": v_ds,
            "current": i_ds
        })

    print(f"Valid JV curves after QC: {len(curves)}")
    return curves


# ============================================================
# 4. CONVERT CURVES → DATAFRAME
# ============================================================

def curves_to_dataframe(curves):
    rows = []
    for idx, c in enumerate(curves):
        sweep = np.linspace(0, 100, len(c["voltage"]))
        for s, v, i in zip(sweep, c["voltage"], c["current"]):
            rows.append({
                "timestamp": pd.Timestamp(idx, unit="s"),
                "sweep_percent": s,
                "voltage": v,
                "current": i,
                "irradiance": c["irradiance"],
                "day_id": idx
            })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


# ============================================================
# 5. MODULE-LEVEL VOC (LINEAR FIT)
# ============================================================

def compute_voc(voltage, current, fraction=0.15):
    n = len(voltage)
    k = max(5, int(n * fraction))
    v_top = voltage[:k]
    i_top = current[:k]
    try:
        m, b = np.polyfit(v_top, i_top, 1)
        if m == 0:
            return np.nan
        return -b / m
    except:
        return np.nan


def compute_jsc(voltage, current):
    idx = np.where(voltage <= 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return current[0]
    v1, v2 = voltage[i-1], voltage[i]
    c1, c2 = current[i-1], current[i]
    return c1 + (0 - v1) * (c2 - c1) / (v2 - v1)


def compute_pmax(voltage, current):
    power = voltage * current
    idx = np.argmax(power)
    return voltage[idx], current[idx], power[idx]


def compute_ff(voc, jsc, pmax):
    if np.isnan(voc) or np.isnan(jsc):
        return np.nan
    return pmax / (voc * jsc)


def compute_efficiency(pmax, area_cm2=1.0, irradiance=1000):
    area_m2 = area_cm2 * 1e-4
    return pmax / (irradiance * area_m2)


def add_jv_metrics(df):
    metrics = []
    for day, df_day in df.groupby("day_id"):
        v = df_day["voltage"].values
        i = df_day["current"].values
        irr = df_day["irradiance"].iloc[0]

        voc = compute_voc(v, i)
        jsc = compute_jsc(v, i)
        vmp, imp, pmax = compute_pmax(v, i)
        ff = compute_ff(voc, jsc, pmax)
        eff = compute_efficiency(pmax, irradiance=irr)

        metrics.append({
            "day_id": day,
            "Voc": voc,
            "Jsc": jsc,
            "Vmp": vmp,
            "Imp": imp,
            "Pmax": pmax,
            "FF": ff,
            "Efficiency": eff
        })

    return df.merge(pd.DataFrame(metrics), on="day_id", how="left")


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def build_dataloaders(root_folder, n_points=100, batch_size=32):
    curves = build_clean_jv_dataset(root_folder, n_points=n_points)

    df = curves_to_dataframe(curves)
    print("Initial dataframe:", df.shape)

    df = add_jv_metrics(df)
    print("Added JV metrics:", df.shape)

    # -----------------------------
    # SCALE TARGETS
    # -----------------------------
    scaler_dict = {}

    voc_scaler = MinMaxScaler()
    jsc_scaler = MinMaxScaler()

    df["Voc"] = voc_scaler.fit_transform(df[["Voc"]])
    df["Jsc"] = jsc_scaler.fit_transform(df[["Jsc"]])

    scaler_dict["Voc"] = voc_scaler
    scaler_dict["Jsc"] = jsc_scaler

    # -----------------------------
    # SCALE FEATURES
    # -----------------------------
    for col in ["voltage", "current"]:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scaler_dict[col] = scaler

    print("Normalization complete.")

    # -----------------------------
    # DATASET
    # -----------------------------
    scalar_dataset = MultiOutputDataset(
        df=df,
        feature_cols=["irradiance", "sweep_percent", "day_id"],
        target_cols=["Voc", "Jsc"],
        window=30,
        horizon=1,
        use_bad_day=False,
        mask_bad_days=False
    )

    X_seq, y_seq, bad_seq, ts_seq = [], [], [], []
    for i in range(len(scalar_dataset)):
        xb, yb, bmb = scalar_dataset[i]
        X_seq.append(xb.numpy())
        y_seq.append(yb.numpy().squeeze())
        bad_seq.append(bmb.numpy().squeeze())
        ts_seq.append(scalar_dataset.get_timestamps_for_sample(i))

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    bad_seq = np.array(bad_seq)

    print("Sequence shapes:", X_seq.shape, y_seq.shape)

    X_train, X_val, y_train, y_val, bad_train, bad_val, ts_train, ts_val = \
        split_train_val_from_sequences(X_seq, y_seq, bad_seq, ts_seq)

    print("Train/Val sizes:", len(X_train), len(X_val))

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(bad_train, dtype=torch.float32)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(bad_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler_dict


def get_dataloaders():
    root = r"C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw"
    return build_dataloaders(root)


if __name__ == "__main__":
    train_loader, val_loader, scaler_dict = get_dataloaders()
    print("DataLoaders ready.")
