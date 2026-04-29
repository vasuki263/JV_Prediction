# evaluate_voc_jsc.py

import torch
import numpy as np
import pandas as pd
import os

from model_lstm import VocJscLSTM
from preprocess_jv import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("outputs/voc_jsc_lstm.pth", map_location=device)
state_dict = checkpoint["model_state"]
scaler_dict = checkpoint["scalers"]

model = VocJscLSTM()
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

_, val_loader, _ = get_dataloaders()

preds_scaled = []
trues_scaled = []

with torch.no_grad():
    for xb, yb, _ in val_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        true = yb.numpy()
        preds_scaled.append(pred)
        trues_scaled.append(true)

preds_scaled = np.vstack(preds_scaled)
trues_scaled = np.vstack(trues_scaled)

voc_scaler = scaler_dict["Voc"]
jsc_scaler = scaler_dict["Jsc"]

voc_pred = voc_scaler.inverse_transform(preds_scaled[:, 0].reshape(-1, 1)).ravel()
voc_true = voc_scaler.inverse_transform(trues_scaled[:, 0].reshape(-1, 1)).ravel()

jsc_pred = jsc_scaler.inverse_transform(preds_scaled[:, 1].reshape(-1, 1)).ravel()
jsc_true = jsc_scaler.inverse_transform(trues_scaled[:, 1].reshape(-1, 1)).ravel()

df = pd.DataFrame({
    "Voc_true": voc_true,
    "Voc_pred": voc_pred,
    "Jsc_true": jsc_true,
    "Jsc_pred": jsc_pred
})

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/predictions.csv", index=False)

print("Saved predictions to outputs/predictions.csv")

def regression_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    return mae, mse, rmse, r2

voc_mae, voc_mse, voc_rmse, voc_r2 = regression_metrics(voc_true, voc_pred)
jsc_mae, jsc_mse, jsc_rmse, jsc_r2 = regression_metrics(jsc_true, jsc_pred)

print("VOC Metrics:")
print("  MAE :", voc_mae)
print("  MSE :", voc_mse)
print("  RMSE:", voc_rmse)
print("  R²  :", voc_r2)

print("\nJSC Metrics:")
print("  MAE :", jsc_mae)
print("  MSE :", jsc_mse)
print("  RMSE:", jsc_rmse)
print("  R²  :", jsc_r2)
