import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# ============================================================
# Smooth curve for visualization ONLY
# ============================================================

def smooth_curve(y, sigma=2.0):

    y = np.asarray(
        y,
        dtype=float
    )

    if len(y) < 5:
        return y

    return gaussian_filter1d(
        y,
        sigma=sigma,
        mode="nearest"
    )


# ============================================================
# Plot TRUE vs PREDICTED residuals
#
# True residual:
#     J_true - J_physics
#
# Predicted residual:
#     Neural-network residual
# ============================================================

def plot_residual_comparison(

    curves_path,

    output_dir,

    num_curves=15,

    smooth=True,

    sigma=8.0,

):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # Load existing predictions
    # ========================================================

    data = np.load(
        curves_path
    )

    V = data["V"]

    # --------------------------------------------------------
    # Already-generated residuals
    # --------------------------------------------------------

    J_residual_true = data[
        "J_residual_true"
    ]

    J_residual_pred = data[
        "J_residual_pred"
    ]

    # ========================================================
    # Number of curves
    # ========================================================

    N = min(
        num_curves,
        len(J_residual_true)
    )

    # ========================================================
    # Plot each curve
    # ========================================================

    for idx in range(N):

        true_residual = (
            J_residual_true[idx]
        )

        predicted_residual = (
            J_residual_pred[idx]
        )

        # ----------------------------------------------------
        # Smooth ONLY for visualization
        # ----------------------------------------------------

        if smooth:

            true_residual_plot = smooth_curve(
                true_residual,
                sigma=sigma
            )

            predicted_residual_plot = smooth_curve(
                predicted_residual,
                sigma=sigma
            )

        else:

            true_residual_plot = (
                true_residual
            )

            predicted_residual_plot = (
                predicted_residual
            )

        # ====================================================
        # Figure
        # ====================================================

        plt.figure(
            figsize=(7, 5)
        )

        # ----------------------------------------------------
        # True residual
        # ----------------------------------------------------

        plt.plot(
            V,
            true_residual_plot,
            label="True residual",
            linewidth=2.5
        )

        # ----------------------------------------------------
        # Predicted residual
        # ----------------------------------------------------

        plt.plot(
            V,
            predicted_residual_plot,
            "--",
            label="Predicted residual",
            linewidth=2.5
        )

        # ----------------------------------------------------
        # Zero reference
        # ----------------------------------------------------

        plt.axhline(
            0,
            linestyle="--",
            linewidth=1.2
        )

        # ====================================================
        # Labels
        # ====================================================

        plt.xlabel(
            "Voltage (V)",
            fontsize=12
        )

        plt.ylabel(
            r"J (Residual) (A/m$^2$)",
            fontsize=12
        )

        plt.title(
            f"J-V Residual — Curve {idx}",
            fontsize=13
        )

        plt.grid(
            alpha=0.25
        )

        plt.legend(
            frameon=True
        )

        plt.tight_layout()

        # ====================================================
        # Save
        # ====================================================

        plt.savefig(
            os.path.join(
                output_dir,
                f"residual_comparison_{idx:04d}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print(
        f"Saved {N} residual comparison plots → "
        f"{output_dir}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    plot_residual_comparison(

        curves_path=(
            "outputs/Perovskite_5&6/SEPIV3/"
            "ae/eval_ops_res/metrics/curves.npz"
        ),

        output_dir=(
            "outputs/Perovskite_5&6/SEPIV3/"
            "ae/eval_ops_res/curves/residual_comparison"
        ),

        num_curves=30,

        smooth=False,

        sigma=8.0,
    )