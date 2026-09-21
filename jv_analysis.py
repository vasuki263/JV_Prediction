# jv_analysis.py

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

def extract_voc(V, I):
    """
    Extract Voc from a JV curve.

    Parameters
    ----------
    V : array-like
        Voltage [V]
    I : array-like
        Current [A]

    Returns
    -------
    Voc : float
        Open-circuit voltage [V]
    """

    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    mask = np.isfinite(V) & np.isfinite(I)

    V = V[mask]
    I = I[mask]

    if len(V) < 2:
        return np.nan

    # Sort by voltage
    idx = np.argsort(V)

    V = V[idx]
    I = I[idx]

    # Exact zero current
    zero_idx = np.where(
        np.isclose(
            I,
            0.0,
            atol=1e-10
        )
    )[0]

    if len(zero_idx) > 0:
        return V[zero_idx[0]]

    # Zero crossing
    for k in range(len(I) - 1):

        if (
            (I[k] <= 0 and I[k + 1] >= 0)
            or
            (I[k] >= 0 and I[k + 1] <= 0)
        ):

            if I[k + 1] == I[k]:
                return V[k]

            return np.interp(
                0.0,
                [I[k], I[k + 1]],
                [V[k], V[k + 1]]
            )

    # No zero crossing
    k = np.argmin(np.abs(I))

    return V[k]


def extract_jv_physics(
    V,
    I,
    area_m2=1.0,
):
    """
    Extract photovoltaic electrical parameters from a JV curve.

    Parameters
    ----------
    V : array-like
        Voltage [V]

    I : array-like
        Current [A]

    area_m2 : float
        Active device area [m²].

        Current density is calculated as:

            J = I / area_m2

        Therefore for area_m2 = 1:

            J [A/m²] = I [A]

        numerically, although the physical units differ.

    Returns
    -------
    tuple
        Voc
        Jsc
        Vmpp
        Jmpp
        Pmpp
        FF
    """

    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    if area_m2 <= 0:
        raise ValueError(
            "area_m2 must be greater than zero."
        )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    mask = np.isfinite(V) & np.isfinite(I)

    V = V[mask]
    I = I[mask]

    if len(V) < 5:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # --------------------------------------------------------
    # Sort by voltage
    # --------------------------------------------------------

    idx = np.argsort(V)

    V = V[idx]
    I = I[idx]

    # --------------------------------------------------------
    # Current -> current density
    #
    # I [A]
    # area [m²]
    # J [A/m²]
    # --------------------------------------------------------

    J = I / area_m2

    # --------------------------------------------------------
    # Jsc
    #
    # Current density at V = 0
    # --------------------------------------------------------

    Jsc = np.interp(
        0.0,
        V,
        J
    )

    # --------------------------------------------------------
    # Voc
    #
    # Voltage where I = 0
    # --------------------------------------------------------

    Voc = extract_voc(
        V,
        I
    )

    # --------------------------------------------------------
    # MPP region
    # --------------------------------------------------------

    # --------------------------------------------------------
    # MPP region
    # --------------------------------------------------------

    Voc_abs = abs(Voc)

    if not np.isfinite(Voc_abs):
        return (
            Voc,
            Jsc,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # Only search inside the physical PV operating region
    # mask_mpp = (
    #         (V >= 0)
    #         & (V <= Voc_abs)
    # )
    #
    # V_mpp_region = V[mask_mpp]
    # J_mpp_region = J[mask_mpp]
    #
    # if len(V_mpp_region) < 2:
    #     return (
    #         Voc,
    #         Jsc,
    #         np.nan,
    #         np.nan,
    #         np.nan,
    #         np.nan,
    #     )
    #
    # P = V_mpp_region * J_mpp_region
    #
    # # If your JV convention gives negative photovoltaic current,
    # # use absolute power.
    # P_abs = np.abs(P)
    #
    # k = np.argmax(P_abs)
    #
    # Vmpp = V_mpp_region[k]
    # Jmpp = J_mpp_region[k]
    # Pmpp = P_abs[k]
    #
    # # --------------------------------------------------------
    # # Fill factor
    # #
    # # FF = Pmpp / (|Voc| |Jsc|)
    # # --------------------------------------------------------
    #
    # denom = (
    #     np.abs(Voc)
    #     * np.abs(Jsc)
    # )
    #
    # if (
    #     not np.isfinite(Voc)
    #     or not np.isfinite(Jsc)
    #     or denom < 1e-12
    # ):
    #     FF = np.nan
    #
    # else:
    #     FF = Pmpp / denom
    #
    # # # Physical sanity check
    # # if FF < 0 or FF > 1.0:
    # #     FF = np.nan
    # FF = np.clip(FF, 0, 1)
    # Positive PV operating region
    mask_mpp = (
            (V >= 0) &
            (V <= Voc_abs)
    )

    V_region = V[mask_mpp]
    J_region = J[mask_mpp]

    if len(V_region) < 3:
        return (
            Voc,
            Jsc,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # Interpolate JV curve
    j_interp = PchipInterpolator(
        V_region,
        J_region
    )

    def power(v):
        return abs(v * j_interp(v))

    result = minimize_scalar(
        lambda v: -power(v),
        bounds=(
            V_region.min(),
            V_region.max()
        ),
        method="bounded"
    )

    Vmpp = result.x
    Jmpp = float(j_interp(Vmpp))
    Pmpp = abs(Vmpp * Jmpp)

    denom = abs(Voc) * abs(Jsc)

    FF = Pmpp / denom if denom > 1e-12 else np.nan

    return (
        Voc,
        Jsc,
        Vmpp,
        Jmpp,
        Pmpp,
        FF,
    )