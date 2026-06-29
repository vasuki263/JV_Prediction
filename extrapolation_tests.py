import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from model_autoencoder import JVAutoencoder
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------
# Smoothing
# ---------------------------------------------------------
def smooth_jv(V, J):
    return V, gaussian_filter1d(J, sigma=3.0)


# ---------------------------------------------------------
# Load AE model
# ---------------------------------------------------------
def load_ae_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = JVAutoencoder(
        ckpt["jv_dim"],
        ckpt["weather_dim"],
        ckpt["latent_dim"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------
# Load AE dataset
# ---------------------------------------------------------
def load_ae_dataset(path):
    data = np.load(path, allow_pickle=True)
    V_grid = data["V_grid"]
    jv_inputs = data["jv_input"]
    jv_in = np.mean(jv_inputs, axis=0)
    return jv_in, V_grid


# ---------------------------------------------------------
# AE JV extrapolation
# ---------------------------------------------------------
def ae_extrapolation_jv(ae_model, jv_in, V_grid,
                        irr_values, temp_values,
                        fixed_temp=25.0, fixed_irr=800.0):

    jv_tensor = torch.tensor(jv_in, dtype=torch.float32).unsqueeze(0)

    irr_curves = []
    temp_curves = []

    for irr in irr_values:
        w = torch.tensor([[fixed_temp, irr]], dtype=torch.float32)
        with torch.no_grad():
            recon, _ = ae_model(jv_tensor, w)
        irr_curves.append(recon.numpy().flatten())

    for temp in temp_values:
        w = torch.tensor([[temp, fixed_irr]], dtype=torch.float32)
        with torch.no_grad():
            recon, _ = ae_model(jv_tensor, w)
        temp_curves.append(recon.numpy().flatten())

    return np.array(irr_curves), np.array(temp_curves)


# ---------------------------------------------------------
# Plot JV curves
# ---------------------------------------------------------
def plot_jv_extrapolation(V_grid, irr_values, irr_curves,
                          temp_values, temp_curves,
                          outdir):

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Irradiance sweep
    plt.figure(figsize=(7,6))
    for irr, J in zip(irr_values, irr_curves):
        V_s, J_s = smooth_jv(V_grid, J)
        plt.plot(V_s, J_s, label=f"Irr={irr:.0f}")
    plt.title("AE JV Extrapolation — Irradiance Sweep")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "jv_extrap_irr.png", dpi=300)
    plt.close()

    # Temperature sweep
    plt.figure(figsize=(7,6))
    for temp, J in zip(temp_values, temp_curves):
        V_s, J_s = smooth_jv(V_grid, J)
        plt.plot(V_s, J_s, label=f"T={temp:.0f}°C")
    plt.title("AE JV Extrapolation — Temperature Sweep")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "jv_extrap_temp.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    # -----------------------------
    # Read tech from command line
    # -----------------------------
    if len(sys.argv) < 2:
        print("Usage: python ae_extrapolate.py <tech>")
        print("Example: python ae_extrapolate.py perovskite")
        sys.exit(1)

    tech = sys.argv[1].lower()

    # -----------------------------
    # Define paths per technology
    # -----------------------------
    model_path = f"models/jv_autoencoder_{tech}.pth"
    dataset_path = f"data/jv_autoencoder_dataset_{tech}.npz"
    outdir = f"outputs/{tech}/plots/ae_jv_extrapolation"

    # Validate paths
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # -----------------------------
    # Load model + dataset
    # -----------------------------
    ae_model = load_ae_model(model_path)
    jv_in, V_grid = load_ae_dataset(dataset_path)

    # -----------------------------
    # Define sweep ranges
    # -----------------------------
    irr_values = np.linspace(50, 1500, 8)
    temp_values = np.linspace(0, 80, 8)

    # -----------------------------
    # Run extrapolation
    # -----------------------------
    irr_curves, temp_curves = ae_extrapolation_jv(
        ae_model,
        jv_in,
        V_grid,
        irr_values,
        temp_values
    )

    # -----------------------------
    # Plot results
    # -----------------------------
    plot_jv_extrapolation(
        V_grid,
        irr_values,
        irr_curves,
        temp_values,
        temp_curves,
        outdir
    )

    print(f"AE JV extrapolation complete for tech: {tech}")
