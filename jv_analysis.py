# jv_analysis.py

import numpy as np


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

    mask_vpos = V >= 0

    V_pos = V[mask_vpos]
    J_pos = J[mask_vpos]

    if len(V_pos) == 0:
        return (
            Voc,
            Jsc,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # --------------------------------------------------------
    # Electrical power density
    #
    # P = V * J
    #
    # [V] * [A/m²] = [W/m²]
    #
    # We use absolute power because the JV sign convention
    # may represent photovoltaic current as negative.
    # --------------------------------------------------------

    P = V_pos * J_pos

    P_abs = np.abs(P)

    k = np.argmax(P_abs)

    Vmpp = V_pos[k]
    Jmpp = J_pos[k]
    Pmpp = P_abs[k]

    # --------------------------------------------------------
    # Fill factor
    #
    # FF = Pmpp / (|Voc| |Jsc|)
    # --------------------------------------------------------

    denom = (
        np.abs(Voc)
        * np.abs(Jsc)
    )

    if (
        not np.isfinite(Voc)
        or not np.isfinite(Jsc)
        or denom < 1e-12
    ):
        FF = np.nan

    else:
        FF = Pmpp / denom

    return (
        Voc,
        Jsc,
        Vmpp,
        Jmpp,
        Pmpp,
        FF,
    )