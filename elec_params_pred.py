import os
import shutil
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from model_autoencoder import PhysicsInformedJVModel


# =========================================================
# Utility
# =========================================================
def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# =========================================================
# Robust Voc extraction
# =========================================================
def extract_voc(V, I):

    V = np.asarray(V, float)
    I = np.asarray(I, float)

    # Zero crossing
    for k in range(len(I) - 1):

        if (I[k] <= 0 and I[k + 1] >= 0) or (
            I[k] >= 0 and I[k + 1] <= 0
        ):

            return np.interp(
                0.0,
                [I[k], I[k + 1]],
                [V[k], V[k + 1]],
            )

    # Fallback
    return V[np.argmin(np.abs(I))]


# =========================================================
# Extract electrical parameters
# =========================================================
def extract_electrical_parameters(V, I):

    V = np.asarray(V, float)
    I = np.asarray(I, float)

    idx = np.argsort(V)

    V = V[idx]
    I = I[idx]

    # -----------------------------
    # Jsc
    # -----------------------------
    Jsc = np.interp(0.0, V, I)

    # -----------------------------
    # Voc
    # -----------------------------
    Voc = extract_voc(V, I)

    # -----------------------------
    # Positive voltage region
    # -----------------------------
    mask = V >= 0

    Vp = V[mask]
    Ip = I[mask]

    # -----------------------------
    # Maximum power point
    # -----------------------------
    P = np.abs(Vp * Ip)

    k = np.argmax(P)

    Vmpp = Vp[k]
    Jmpp = Ip[k]
    Pmpp = P[k]

    # -----------------------------
    # Fill factor
    # -----------------------------
    # FF = Pmpp / (np.abs(Voc * Jsc) + 1e-12)
    if np.isnan(Voc) or abs(Voc) < 1e-6 or abs(Jsc) < 1e-6:
        FF = np.nan
    else:
        FF = Pmpp / abs(Voc * Jsc)

    return Voc, Jsc, Vmpp, Jmpp, Pmpp, FF


# =========================================================
# Error metrics
# =========================================================
# def compute_error_metrics(true, pred):
#
#     true = np.asarray(true, dtype=float)
#     pred = np.asarray(pred, dtype=float)
#
#     mask = np.isfinite(true) & np.isfinite(pred)
#
#     true = true[mask]
#     pred = pred[mask]
#
#     if len(true) == 0:
#
#         return {
#             "MAE": np.nan,
#             "RMSE": np.nan,
#             "MAPE": np.nan,
#             "R2": np.nan,
#         }
#
#     mae = mean_absolute_error(true, pred)
#
#     rmse = np.sqrt(mean_squared_error(true, pred))
#
#     mape = np.mean(
#         np.abs((true - pred) / (true + 1e-12))
#     ) * 100
#
#     r2 = (
#         r2_score(true, pred)
#         if len(true) > 1
#         else np.nan
#     )
#
#     return {
#         "MAE": mae,
#         "RMSE": rmse,
#         "MAPE": mape,
#         "R2": r2,
#     }

def compute_error_metrics(true, pred):

    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]

    if len(true) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "sMAPE": np.nan,
            "NRMSE": np.nan,
            "R2": np.nan,
            "Corr": np.nan,
        }

    # Absolute metrics
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    # Symmetric MAPE (stable)
    smape = np.mean(
        np.abs(true - pred) / (np.abs(true) + np.abs(pred) + 1e-12)
    ) * 100

    # Normalized RMSE (scale-invariant)
    nrmse = rmse / (true.max() - true.min() + 1e-12)

    # R2 (protect against low variance)
    if np.var(true) < 1e-12:
        r2 = np.nan
    else:
        r2 = r2_score(true, pred)

    # Pearson correlation
    if len(true) > 1:
        corr = np.corrcoef(true, pred)[0, 1]
    else:
        corr = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "NRMSE": nrmse,
        "R2": r2,
        "Corr": corr,
    }

# =========================================================
# Main evaluation
# =========================================================
def evaluate_electrical_model(
    model_path,
    dataset_path,
    out_dir="outputs_electrical",
):

    reset_dir(out_dir)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    data = np.load(dataset_path, allow_pickle=True)

    jv_in = data["jv_input"].astype(np.float32)

    jv_tgt = data["jv_target"].astype(np.float32)

    weather = data["weather"].astype(np.float32)

    V_grid = data["V_grid"].astype(np.float32)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    # -----------------------------------------------------
    # Predict JV curves
    # -----------------------------------------------------
    preds = []

    with torch.no_grad():

        for jv, w in zip(jv_in, weather):

            jv_t = torch.tensor(
                jv,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            w_t = torch.tensor(
                w,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            I_pred, I_phys, delta_I, z, diode_pred_norm = model(jv_t, w_t)

            preds.append(
                I_pred.squeeze(0)
                .cpu()
                .numpy()
            )

    preds = np.asarray(preds, dtype=np.float32)

    # -----------------------------------------------------
    # Extract electrical parameters
    # -----------------------------------------------------
    print("Extracting electrical parameters...")

    rows = []

    for i in range(len(preds)):

        Voc_t, Jsc_t, Vmpp_t, Jmpp_t, Pmpp_t, FF_t = (
            extract_electrical_parameters(
                V_grid,
                jv_tgt[i],
            )
        )

        Voc_p, Jsc_p, Vmpp_p, Jmpp_p, Pmpp_p, FF_p = (
            extract_electrical_parameters(
                V_grid,
                preds[i],
            )
        )

        rows.append(
            {
                "idx": i,

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
            }
        )

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(preds)}")

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(
            out_dir,
            "electrical_parameters.csv",
        ),
        index=False,
    )

    # -----------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------
    # metrics = {}
    #
    # params = [
    #     "Voc",
    #     "Jsc",
    #     "Vmpp",
    #     "Jmpp",
    #     "Pmpp",
    #     "FF",
    # ]
    #
    # for p in params:
    #
    #     metrics[p] = compute_error_metrics(
    #         df[f"{p}_true"],
    #         df[f"{p}_pred"],
    #     )
    #
    # metrics_df = pd.DataFrame(metrics).T
    #
    # metrics_df.to_csv(
    #     os.path.join(
    #         out_dir,
    #         "electrical_metrics.csv",
    #     )
    # )
    metrics = {}
    params = ["Voc", "Jsc", "Vmpp", "Jmpp", "Pmpp", "FF"]

    for p in params:
        metrics[p] = compute_error_metrics(
            df[f"{p}_true"],
            df[f"{p}_pred"],
        )

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(out_dir, "electrical_metrics.csv"))

    print("\n=== Electrical Parameter Metrics ===")

    print(metrics_df)

    # -----------------------------------------------------
    # Plot comparisons
    # -----------------------------------------------------
    plot_dir = os.path.join(out_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)

    for p in params:

        true = df[f"{p}_true"].values
        pred = df[f"{p}_pred"].values

        mask = np.isfinite(true) & np.isfinite(pred)

        true = true[mask]
        pred = pred[mask]

        plt.figure(figsize=(5, 5))

        plt.scatter(
            true,
            pred,
            alpha=0.6,
            edgecolor="k",
        )

        lims = [
            min(true.min(), pred.min()),
            max(true.max(), pred.max()),
        ]

        plt.plot(lims, lims, "r--")

        plt.xlabel(f"True {p}")

        plt.ylabel(f"Predicted {p}")

        plt.title(f"{p}: True vs Predicted")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                plot_dir,
                f"{p}_scatter.png",
            ),
            dpi=200,
        )

        plt.close()

    print(f"\nResults saved to: {out_dir}")

    return df


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate electrical parameters from predicted JV curves"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs_electrical",
    )

    args = parser.parse_args()

    evaluate_electrical_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        out_dir=args.out_dir,
    )
