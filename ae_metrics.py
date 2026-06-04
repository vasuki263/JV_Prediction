import numpy as np


# ---------------------------------------------------------
# Basic error metrics (per sample)
# ---------------------------------------------------------
def compute_basic_errors(preds, targets):
    """
    Computes MSE, MAE, RMSE, and R² for each sample.
    preds, targets: arrays of shape [N, M]
    """
    mse = np.mean((preds - targets) ** 2, axis=1)
    mae = np.mean(np.abs(preds - targets), axis=1)
    rmse = np.sqrt(mse)

    # R² per sample
    r2 = []
    for p, t in zip(preds, targets):
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2.append(1 - ss_res / (ss_tot + 1e-9))

    return mse, mae, rmse, np.array(r2)


def extract_physics_from_curve(jv, V_grid):
    """
    Extracts Voc, Jsc, Pmpp, FF from JV curves.
    jv: array [N, M]
    V_grid: array [M]
    """
    Voc, Jsc, Pmpp, FF = [], [], [], []

    for J in jv:
        Voc_i = V_grid[np.argmin(np.abs(J))]
        Jsc_i = J[0]
        P = V_grid * J
        Pmpp_i = np.max(P)
        FF_i = Pmpp_i / (Voc_i * Jsc_i + 1e-9)

        Voc.append(Voc_i)
        Jsc.append(Jsc_i)
        Pmpp.append(Pmpp_i)
        FF.append(FF_i)

    return (
        np.array(Voc),
        np.array(Jsc),
        np.array(Pmpp),
        np.array(FF),
    )


# ---------------------------------------------------------
# Compute physics‑metric deltas
# ---------------------------------------------------------
def compute_physics_metrics(Voc_t, Voc_p, Jsc_t, Jsc_p, Pmpp_t, Pmpp_p, FF_t, FF_p):
    """
    Computes ΔVoc, ΔJsc, ΔPmpp, ΔFF for each sample.
    """
    dVoc = Voc_p - Voc_t
    dJsc = Jsc_p - Jsc_t
    dPmpp = Pmpp_p - Pmpp_t
    dFF = FF_p - FF_t

    return np.vstack([dVoc, dJsc, dPmpp, dFF]).T
