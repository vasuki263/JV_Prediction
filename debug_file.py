import numpy as np
import torch
from model_autoencoder import PhysicsInformedJVModel

def debug_nan(dataset_path, n_checks=5):
    print("\n=== HARD NAN DEBUGGER ===")

    data = np.load(dataset_path, allow_pickle=True)
    jv_input = data["jv_input"].astype(np.float32)
    jv_target = data["jv_target"].astype(np.float32)
    weather = data["weather"].astype(np.float32)
    V_grid = data["V_grid"].astype(np.float32)

    print(f"Dataset loaded: {jv_input.shape[0]} samples")

    # -------------------------
    # 1. Check raw JV
    # -------------------------
    print("\nChecking raw JV...")
    for i in range(n_checks):
        print(f"\nSample {i}:")
        print("  JV input min/max:", np.nanmin(jv_input[i]), np.nanmax(jv_input[i]))
        print("  JV target min/max:", np.nanmin(jv_target[i]), np.nanmax(jv_target[i]))
        print("  JV input has NaN:", np.isnan(jv_input[i]).any())
        print("  JV target has NaN:", np.isnan(jv_target[i]).any())

    # -------------------------
    # 2. Initialize model
    # -------------------------
    print("\nInitializing model...")
    model = PhysicsInformedJVModel(
        jv_dim=jv_input.shape[1],
        weather_dim=weather.shape[1],
        voltage_grid=V_grid,
    )
    model.eval()

    # -------------------------
    # 3. Check model forward
    # -------------------------
    print("\nChecking model forward pass...")

    for i in range(n_checks):
        print(f"\n--- Forward sample {i} ---")

        jv = torch.tensor(jv_input[i:i+1])
        w  = torch.tensor(weather[i:i+1])

        print("  Input JV has NaN:", torch.isnan(jv).any().item())
        print("  Input weather:", w)

        try:
            with torch.no_grad():
                I_pred, I_phys, delta_I, z, params = model(jv, w)
        except Exception as e:
            print("  ❌ Exception during forward:", e)
            return

        print("  Params:", params)
        print("  Params contain NaN:", torch.isnan(params).any().item())

        print("  Physics JV contains NaN:", torch.isnan(I_phys).any().item())
        print("  Residual contains NaN:", torch.isnan(delta_I).any().item())
        print("  Prediction contains NaN:", torch.isnan(I_pred).any().item())

    print("\n=== HARD DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_nan("data/jv_autoencoder_dataset_perovskite.npz")
