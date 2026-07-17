import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_autoencoder import JVPhysicsDataset
from model_autoencoder import PhysicsInformedJVModel


# ---------------------------------------------------------
# Electrical parameter extraction
# ---------------------------------------------------------
def calculate_voc(V, I):
    """
    Calculate Voc by finding the zero-current crossing.
    """

    sign_change = np.where(np.diff(np.sign(I)))[0]

    if len(sign_change) == 0:
        return np.nan

    idx = sign_change[0]

    V1, V2 = V[idx], V[idx + 1]
    I1, I2 = I[idx], I[idx + 1]

    Voc = V1 - I1 * (V2 - V1) / (I2 - I1)

    return Voc

def extract_electrical_params(V, I):
    """
    Extract electrical parameters from a JV curve.

    Parameters
    ----------
    V : numpy array
        Voltage grid

    I : numpy array
        Current values

    Returns
    -------
    dict
    """

    # Short-circuit current
    Isc = np.interp(0.0, V, I)

    # Open-circuit voltage
    # Voc = np.interp(0.0, I[::-1], V[::-1])
    Voc = calculate_voc(V, I)

    # Power
    P = V * I

    idx = np.argmax(P)

    Vmpp = V[idx]
    Impp = I[idx]
    Pmax = P[idx]

    FF = Pmax / (Voc * Isc + 1e-12)

    return {
        "Voc": Voc,
        "Isc": Isc,
        "Vmpp": Vmpp,
        "Impp": Impp,
        "Pmax": Pmax,
        "FF": FF,
    }


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "models/jv_autoencoder_perovskite.pth",
    map_location=device,
)

model = PhysicsInformedJVModel(
    jv_dim=checkpoint["jv_dim"],
    weather_dim=checkpoint["weather_dim"],
    voltage_grid=checkpoint["V_grid"],
    latent_dim=checkpoint["latent_dim"],
    diode_mean=checkpoint["diode_mean"],
    diode_std=checkpoint["diode_std"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
dataset = JVPhysicsDataset("data/jv_autoencoder_dataset_perovskite.npz")

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
)

V_grid = dataset.V_grid

rows = []


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------
with torch.no_grad():

    for jv_in, jv_target, weather, diode_gt in loader:

        jv_in = jv_in.to(device)
        weather = weather.to(device)

        I_pred, I_phys, delta_I, z, diode_pred_norm = model(
            jv_in,
            weather,
        )

        # -------------------------------
        # Denormalize diode parameters
        # -------------------------------
        diode_raw = (
            diode_pred_norm * model.diode_std
            + model.diode_mean
        )

        diode_raw = diode_raw.cpu()

        I_pred = I_pred.cpu().numpy()

        for i in range(I_pred.shape[0]):

            curve = I_pred[i]

            # -----------------------
            # Physical diode params
            # -----------------------
            raw = diode_raw[i]

            Iph = raw[0].item()

            Io = np.exp(raw[1].item())

            n = 0.8 + 2.2 / (
                1 + np.exp(-raw[2].item())
            )

            Rs = 20.0 / (
                1 + np.exp(-raw[3].item())
            )

            # Rsh = 100 + (1e5 - 100) / (
            #     1 + np.exp(-raw[4].item())
            # )
            # Rsh = raw[4].item()
            Rsh = max(raw[4].item(), 0.0)

            # -----------------------
            # Electrical parameters
            # -----------------------
            elec = extract_electrical_params(
                V_grid,
                curve,
            )

            row = {
                "Iph": Iph,
                "Io": Io,
                "n": n,
                "Rs": Rs,
                "Rsh": Rsh,
                "Voc": elec["Voc"],
                "Isc": elec["Isc"],
                "Vmpp": elec["Vmpp"],
                "Impp": elec["Impp"],
                "Pmax": elec["Pmax"],
                "FF": elec["FF"],
            }

            rows.append(row)
            print("raw[4] =", raw[4].item())
            print("Rsh =", Rsh)


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------
df = pd.DataFrame(rows)

df.to_csv(
    "outputs/predicted_diode_and_electrical_parameters.csv",
    index=False,
)

print(df.head())

print("\nSaved to predicted_diode_and_electrical_parameters.csv")

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(V_grid, curve)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Voltage")
plt.ylabel("Current")
plt.grid(True)
plt.show()

plt.plot(V_grid, I_phys[0], label="Physics")
plt.plot(V_grid, I_pred[0], label="Prediction")
plt.plot(V_grid, jv_target[0], label="Ground Truth")
plt.axhline(0, color="k", linestyle="--")
plt.legend()
plt.show()