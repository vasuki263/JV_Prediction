# build_dataset.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from extract_clean_jv import extract_all

def downsample(v, i, n=100):
    n0 = min(len(v), len(i))
    idx = np.linspace(0, n0-1, n).astype(int)
    return v[idx], i[idx]

def build_dataset(directory, batch=32):
    samples = extract_all(directory)

    X, Y = [], []
    scalers = {
        "voltage": MinMaxScaler(),
        "current": MinMaxScaler(),
        "temperature": MinMaxScaler(),
        "irradiance": MinMaxScaler(),
        "targets": MinMaxScaler()
    }

    # Fit feature scalers
    all_v = np.concatenate([s["voltage"] for s in samples])
    all_i = np.concatenate([s["current"] for s in samples])
    all_t = np.array([s["temperature"] for s in samples]).reshape(-1,1)
    all_g = np.array([s["irradiance"] for s in samples]).reshape(-1,1)

    scalers["voltage"].fit(all_v.reshape(-1,1))
    scalers["current"].fit(all_i.reshape(-1,1))
    scalers["temperature"].fit(all_t)
    scalers["irradiance"].fit(all_g)

    # Fit target scaler (Voc, Jsc, Pmpp)
    targets = np.array([[s["Voc"], s["Jsc"], s["Pmpp"]] for s in samples])
    scalers["targets"].fit(targets)

    for s in samples:
        v, i = downsample(s["voltage"], s["current"])
        v = scalers["voltage"].transform(v.reshape(-1,1))
        i = scalers["current"].transform(i.reshape(-1,1))

        sweep = np.linspace(0,1,len(v)).reshape(-1,1)

        temp = scalers["temperature"].transform([[s["temperature"]]])[0][0]
        irr  = scalers["irradiance"].transform([[s["irradiance"]]])[0][0]

        temp_seq = np.full((len(v),1), temp)
        irr_seq  = np.full((len(v),1), irr)

        X.append(np.hstack([v,i,sweep,temp_seq,irr_seq]))

        Y.append(scalers["targets"].transform([[s["Voc"], s["Jsc"], s["Pmpp"]]])[0])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    ds = TensorDataset(X,Y)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    return loader, scalers
