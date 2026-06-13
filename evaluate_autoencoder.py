import numpy as np
import torch
from pathlib import Path
import shutil
import os
import pandas as pd

from model_autoencoder import JVAutoencoder
from ae_metrics import (
    compute_basic_errors,
    compute_physics_metrics,
    extract_physics_from_curve,
    extract_physics_single_curve
)
from ae_plotting import (
    plot_error_histogram,
    plot_voltage_error,
    plot_pred_vs_actual,
    plot_reconstructed_curves,
    plot_physics_error,
    plot_error_vs_physics_params,
    plot_error_heatmap,
    plot_param
)
from ae_latent import (
    compute_latent_vectors,
    plot_latent_umap,
    plot_latent_tsne
)


# ---------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------
def evaluate_autoencoder(model_path, dataset_path, tech, physics_params=None):

    # -----------------------------------------------------
    # Output directories
    # -----------------------------------------------------
    base_dir = Path(f"outputs/{tech}/ae")
    plot_dir    = base_dir / "plots"
    metric_dir  = base_dir / "metrics"
    curve_dir   = base_dir / "curves"
    latent_dir  = base_dir / "latent"
    physics_dir = base_dir / "physics"

    for d in [plot_dir, metric_dir, curve_dir, latent_dir, physics_dir]:
        reset_dir(d)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    data = np.load(dataset_path, allow_pickle=True)
    jv_in   = data["jv_input"]
    jv_tgt  = data["jv_target"]
    weather = data["weather"]
    V_grid  = data["V_grid"]

    jv_dim      = jv_in.shape[1]
    weather_dim = weather.shape[1]

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    ckpt = torch.load(model_path, map_location="cpu")
    model = JVAutoencoder(jv_dim, weather_dim, ckpt["latent_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------
    preds = []
    with torch.no_grad():
        for jv, w in zip(jv_in, weather):
            jv_t = torch.tensor(jv).unsqueeze(0)
            w_t  = torch.tensor(w).unsqueeze(0)
            jv_recon, _ = model(jv_t, w_t)
            preds.append(jv_recon.squeeze().numpy())

    preds = np.array(preds)

    # -----------------------------------------------------
    # Basic metrics
    # -----------------------------------------------------
    mse, mae, rmse, r2 = compute_basic_errors(preds, jv_tgt)

    np.savetxt(
        metric_dir / "basic_errors.csv",
        np.vstack([mse, mae, rmse, r2]).T,
        delimiter=",",
        header="MSE,MAE,RMSE,R2",
        comments=""
    )

    # -----------------------------------------------------
    # Extract physics parameters for measured & predicted
    # -----------------------------------------------------
    Voc_true_list = []
    Jsc_true_list = []
    Pmpp_true_list = []
    FF_true_list = []

    Voc_pred_list = []
    Jsc_pred_list = []
    Pmpp_pred_list = []
    FF_pred_list = []

    for true_curve, pred_curve in zip(jv_tgt, preds):
        Voc_t, Jsc_t, Pmpp_t, FF_t = extract_physics_single_curve(true_curve, V_grid)
        Voc_p, Jsc_p, Pmpp_p, FF_p = extract_physics_single_curve(pred_curve, V_grid)

        Voc_true_list.append(Voc_t)
        Jsc_true_list.append(Jsc_t)
        Pmpp_true_list.append(Pmpp_t)
        FF_true_list.append(FF_t)

        Voc_pred_list.append(Voc_p)
        Jsc_pred_list.append(Jsc_p)
        Pmpp_pred_list.append(Pmpp_p)
        FF_pred_list.append(FF_p)

    # -----------------------------------------------------
    # Save physics parameters to CSV
    # -----------------------------------------------------
    physics_df = pd.DataFrame({
        "Voc_true": Voc_true_list,
        "Voc_pred": Voc_pred_list,
        "Jsc_true": Jsc_true_list,
        "Jsc_pred": Jsc_pred_list,
        "Pmpp_true": Pmpp_true_list,
        "Pmpp_pred": Pmpp_pred_list,
        "FF_true": FF_true_list,
        "FF_pred": FF_pred_list,
    })

    physics_df.to_csv(physics_dir / "physics_parameters.csv", index=False)

    # -----------------------------------------------------
    # Physics metrics
    # -----------------------------------------------------
    Voc_t, Jsc_t, Pmpp_t, FF_t = extract_physics_from_curve(jv_tgt, V_grid)
    Voc_p, Jsc_p, Pmpp_p, FF_p = extract_physics_from_curve(preds, V_grid)

    physics_metrics = compute_physics_metrics(
        Voc_t, Voc_p, Jsc_t, Jsc_p, Pmpp_t, Pmpp_p, FF_t, FF_p
    )

    np.savetxt(
        physics_dir / "physics_errors.csv",
        physics_metrics,
        delimiter=",",
        header="dVoc,dJsc,dPmpp,dFF",
        comments=""
    )

    plot_error_histogram(mse, mae, rmse, r2, plot_dir)
    # plot_voltage_error(V_grid, preds, jv_tgt, plot_dir)
    plot_pred_vs_actual(preds, jv_tgt, plot_dir)
    plot_reconstructed_curves(V_grid, jv_in, jv_tgt, preds, curve_dir)
    # plot_physics_error(Voc_t, Voc_p, Jsc_t, Jsc_p, Pmpp_t, Pmpp_p, FF_t, FF_p, physics_dir)
    plot_error_heatmap(preds, jv_tgt, plot_dir)
    plot_param(Voc_true_list, Voc_pred_list, "Voc", physics_dir)
    plot_param(Jsc_true_list, Jsc_pred_list, "Jsc", physics_dir)
    plot_param(Pmpp_true_list, Pmpp_pred_list, "Pmpp", physics_dir)
    plot_param(FF_true_list, FF_pred_list, "FF", physics_dir)

    # -----------------------------------------------------
    # Physics-parameter correlations
    # -----------------------------------------------------
    if physics_params is not None:
        plot_error_vs_physics_params(mse, physics_params, physics_dir)

    # -----------------------------------------------------
    # Latent space
    # -----------------------------------------------------
    Z = compute_latent_vectors(model, jv_in, weather)
    # plot_latent_umap(Z, latent_dir)
    # plot_latent_tsne(Z, latent_dir)

    print(f"AE evaluation complete → outputs/{tech}/ae/")
    return preds
