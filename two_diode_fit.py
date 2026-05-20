# two_diode_fit.py — Stable Two‑Diode Model Fitting (Overflow‑Safe, Robust)

import numpy as np
from scipy.optimize import curve_fit

# Physical constants
k = 1.380649e-23
q = 1.602176634e-19
T = 298.15
Vt = k * T / q   # thermal voltage ~ 25.8 mV


# ---------------------------------------------------------
# Safe exponential (prevents overflow)
# ---------------------------------------------------------
def safe_exp(x):
    return np.exp(np.clip(x, -50, 50))   # clamps exponent to avoid overflow


# ---------------------------------------------------------
# Two‑diode model (explicit approximation)
# ---------------------------------------------------------
def two_diode_model(V, Jph, J01, J02, n1, n2, Rs, Rsh):
    """
    Explicit two‑diode model (Rs handled weakly but stable).
    Good enough for extracting recombination trends.
    """
    Vd = V  # simplified diode voltage (implicit Rs would require iteration)

    term1 = J01 * (safe_exp(Vd / (n1 * Vt)) - 1.0)
    term2 = J02 * (safe_exp(Vd / (n2 * Vt)) - 1.0)
    shunt = Vd / Rsh

    return Jph - term1 - term2 - shunt


# ---------------------------------------------------------
# Two‑diode fitting wrapper
# ---------------------------------------------------------
def fit_two_diode(V, I):
    V = np.array(V, dtype=float)
    I = np.array(I, dtype=float)

    # Not enough points → return NaNs
    if len(V) < 10:
        return {
            "Jph": np.nan, "J01": np.nan, "J02": np.nan,
            "n1": np.nan, "n2": np.nan, "Rs": np.nan, "Rsh": np.nan
        }

    # Initial guesses
    Jph_guess = max(I) if np.max(I) > 0 else np.abs(np.min(I))

    p0 = [
        Jph_guess,   # Jph
        1e-12,       # J01
        1e-10,       # J02
        1.2,         # n1
        2.0,         # n2
        1.0,         # Rs
        1000.0       # Rsh
    ]

    # Physically meaningful bounds
    bounds = (
        [0.0, 1e-15, 1e-15, 1.0, 1.0, 0.0, 1.0],       # lower bounds
        [2*Jph_guess, 1e-6, 1e-6, 2.0, 4.0, 50.0, 1e7] # upper bounds
    )

    try:
        popt, _ = curve_fit(
            two_diode_model,
            V,
            I,
            p0=p0,
            bounds=bounds,
            method="trf",          # Trust Region Reflective (robust)
            loss="soft_l1",        # robust to noise/outliers
            f_scale=0.1,
            maxfev=20000
        )

        Jph, J01, J02, n1, n2, Rs, Rsh = popt

        return {
            "Jph": float(Jph),
            "J01": float(J01),
            "J02": float(J02),
            "n1": float(n1),
            "n2": float(n2),
            "Rs": float(Rs),
            "Rsh": float(Rsh)
        }

    except Exception:
        # Fallback if fitting fails
        return {
            "Jph": np.nan,
            "J01": np.nan,
            "J02": np.nan,
            "n1": np.nan,
            "n2": np.nan,
            "Rs": np.nan,
            "Rsh": np.nan
        }
