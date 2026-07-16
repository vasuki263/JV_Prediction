import os
import shutil
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from model_autoencoder import PhysicsInformedJVModel


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def extract_jv_physics(V, I):

    V = np.asarray(V, float)
    I = np.asarray(I, float)

    mask = np.isfinite(V) & np.isfinite(I)
    V = V[mask]
    I = I[mask]

    if len(V) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    idx = np.argsort(V)
    V = V[idx]
    I = I[idx]

    # ------------------------
    # Jsc
    # ------------------------
    Jsc = np.interp(0.0, V, I)

    # ------------------------
    # Voc
    # ------------------------
    sign_change = np.where(np.diff(np.sign(I)) != 0)[0]

    if len(sign_change):

        k = sign_change[0]

        Voc = np.interp(
            0.0,
            [I[k], I[k + 1]],
            [V[k], V[k + 1]]
        )

    else:
        # nearest point to zero current
        idx0 = np.argmin(np.abs(I))

        # tolerance depends on your current scale
        if abs(I[idx0]) < 0.02:
            Voc = V[idx0]
        else:
            Voc = np.nan

    # ------------------------
    # MPP
    # ------------------------
    P = V * I
    k = np.argmax(P)

    Vmpp = V[k]
    Jmpp = I[k]
    Pmpp = P[k]

    FF = Pmpp / (Voc * Jsc + 1e-12)

    return Voc, Jsc, Vmpp, Jmpp, Pmpp, FF


def unnormalize_params(params_norm, stats):
    """Undo normalization: x = x_norm * std + mean"""
    return params_norm * stats["std"] + stats["mean"]


def evaluate_diode_model(
    model_path,
    dataset_path,
    tech,
    num_curves=10,
):

    base_dir = f"outputs/{tech}/ae/physics"
    curve_dir = os.path.join(base_dir, "curves")
    metrics_dir = os.path.join(base_dir, "metrics")

    reset_dir(base_dir)
    reset_dir(curve_dir)
    reset_dir(metrics_dir)

    # ----------------------------
    # Load dataset
    # ----------------------------
    data = np.load(dataset_path, allow_pickle=True)

    jv_in = data["jv_input"].astype(np.float32)
    jv_tgt = data["jv_target"].astype(np.float32)
    weather = data["weather"].astype(np.float32)
    V_grid = data["V_grid"].astype(np.float32)

    # Normalized true params
    diode_true_norm = data["diode_params_norm"].astype(np.float32)

    # Normalization stats
    diode_stats = data["diode_params_stats"].item()

    # ----------------------------
    # Load model
    # ----------------------------
    ckpt = torch.load(model_path, map_location="cpu")

    model = PhysicsInformedJVModel(
        jv_dim=jv_in.shape[1],
        weather_dim=weather.shape[1],
        voltage_grid=V_grid,
        latent_dim=ckpt["latent_dim"],
        diode_mean=ckpt["diode_mean"],
        diode_std=ckpt["diode_std"],
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds = []
    params_preds = []

    # ----------------------------
    # Inference
    # ----------------------------
    with torch.no_grad():
        for jv, w in zip(jv_in, weather):

            jv_t = torch.tensor(jv, dtype=torch.float32, device=device).unsqueeze(0)
            w_t = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)

            # CORRECT UNPACKING
            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

            preds.append(I_pred.squeeze(0).cpu().numpy())

            # Unnormalize predicted params
            params_pred = unnormalize_params(
                diode_pred_norm.squeeze(0).cpu().numpy(),
                diode_stats
            )

            params_preds.append(params_pred)

    preds = np.array(preds, dtype=np.float32)
    params_preds = np.array(params_preds, dtype=np.float32)
    print("pred_params=", params_preds)

    # ----------------------------
    # Unnormalize true params
    # ----------------------------
    diode_true = unnormalize_params(diode_true_norm, diode_stats)
    print("diode_true=", diode_true)

    # ----------------------------
    # Save results
    # ----------------------------
    rows = []

    for idx in range(len(preds)):

        J_true = jv_tgt[idx]
        J_pred = preds[idx]

        Voc_t, Jsc_t, Vmpp_t, Jmpp_t, Pmpp_t, FF_t = extract_jv_physics(V_grid, J_true)
        Voc_p, Jsc_p, Vmpp_p, Jmpp_p, Pmpp_p, FF_p = extract_jv_physics(V_grid, J_pred)

        Iph_t, Io_t, n_t, Rs_t, Rsh_t = diode_true[idx]
        Iph_p, Io_p, n_p, Rs_p, Rsh_p = params_preds[idx]




        rows.append({
            "idx": idx,
            "Voc_true": Voc_t,
            "Voc_pred": Voc_p,
            "Jsc_true": Jsc_t,
            "Jsc_pred": Jsc_p,
            "Vmpp_true": Vmpp_t,
            "Vmpp_pred": Vmpp_p,
            "Jmpp_true": Jmpp_t,
            "Jmpp_pred": Jmpp_p,
            "Pmpp_true": Pmpp_t,
            "Pmpp_pred": Pmpp_p,
            "FF_true": FF_t,
            "FF_pred": FF_p,
            "Iph_true": Iph_t,
            "Iph_pred": Iph_p,
            "Io_true": Io_t,
            "Io_pred": Io_p,
            "n_true": n_t,
            "n_pred": n_p,
            "Rs_true": Rs_t,
            "Rs_pred": Rs_p,
            "Rsh_true": Rsh_t,
            "Rsh_pred": Rsh_p,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(metrics_dir, "jv_predictions_with_diode_params.csv"), index=False)

    # ----------------------------
    # Save curves
    # ----------------------------
    np.savez(
        os.path.join(metrics_dir, "curves.npz"),
        V=V_grid,
        J_true=jv_tgt,
        J_pred=preds,
        diode_true=diode_true,
        diode_pred=params_preds,
    )

    # ----------------------------
    # Plot curves
    # ----------------------------
    plot_dir = os.path.join(curve_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    N = min(num_curves, len(preds))

    for idx in range(N):

        plt.figure(figsize=(6, 5))
        plt.plot(V_grid, jv_tgt[idx], label="True JV", linewidth=2)
        plt.plot(V_grid, preds[idx], "--", label="Pred JV", linewidth=2)

        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"JV Curve {idx}")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"jv_{idx:04d}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Evaluation complete → {base_dir}")

    return preds, df


if __name__ == "__main__":
    preds, physics_results = evaluate_diode_model(
        model_path="models/jv_autoencoder_silicon.pth",
        dataset_path="data/jv_autoencoder_dataset_silicon.npz",
        tech="SunPower_5-1",
        num_curves=10,
    )
