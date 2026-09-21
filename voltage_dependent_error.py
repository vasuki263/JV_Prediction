import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

print("Script directory:")
print(BASE_DIR)


# ============================================================
# CONFIGURATION
# ============================================================

panel_files = {

    "Perovskite_5":
        BASE_DIR /
        "outputs/Perovskite_5&6/SEPIV1/ae/eval_outputs_0_2_diode_loss_v4/metrics/curves.npz",

    "Perovskite_6":
BASE_DIR /
        "outputs/Perovskite_5&6/SEPIV3/ae/eval_outputs_0_2_diode_loss_v4/metrics/curves.npz",

    "SunPower_5_1":
BASE_DIR /
        "outputs/SunPower_5-1/ae/eval_outputs_0_2_diode_loss_v2/metrics/curves.npz",
}


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = Path(
    "outputs/evaluation_plots_V3/voltage_dependent_error"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_voltage_errors(
    J_true,
    J_pred,
    J_physics
):

    # --------------------------------------------------------
    # Errors for every curve and voltage point
    # --------------------------------------------------------

    final_error = (
        J_pred - J_true
    )

    physics_error = (
        J_physics - J_true
    )

    # --------------------------------------------------------
    # RMSE at each voltage
    # --------------------------------------------------------

    final_rmse = np.sqrt(
        np.mean(
            final_error ** 2,
            axis=0
        )
    )

    physics_rmse = np.sqrt(
        np.mean(
            physics_error ** 2,
            axis=0
        )
    )

    # --------------------------------------------------------
    # MAE at each voltage
    # --------------------------------------------------------

    final_mae = np.mean(
        np.abs(final_error),
        axis=0
    )

    physics_mae = np.mean(
        np.abs(physics_error),
        axis=0
    )

    # --------------------------------------------------------
    # RMSE improvement
    #
    # Positive = final model is better
    # Negative = final model is worse
    # --------------------------------------------------------

    improvement = np.full_like(
        physics_rmse,
        np.nan
    )

    valid = (
        physics_rmse > 1e-12
    )

    improvement[valid] = (
        (
            physics_rmse[valid]
            -
            final_rmse[valid]
        )
        /
        physics_rmse[valid]
    ) * 100.0

    return (
        physics_rmse,
        final_rmse,
        physics_mae,
        final_mae,
        improvement
    )


# ============================================================
# REGIONAL ERROR ANALYSIS
# ============================================================

def calculate_regional_errors(
    V,
    J_true,
    J_pred,
    J_physics
):

    rows = []

    n_curves = J_true.shape[0]

    for idx in range(n_curves):

        true_curve = J_true[idx]

        pred_curve = J_pred[idx]

        physics_curve = J_physics[idx]

        # ----------------------------------------------------
        # Estimate Voc from zero crossing
        # ----------------------------------------------------

        # Find last valid voltage/current pair
        valid = (
            np.isfinite(V)
            &
            np.isfinite(true_curve)
        )

        V_valid = V[valid]
        J_valid = true_curve[valid]

        if len(V_valid) < 2:
            continue

        # Find location where current becomes closest to zero
        voc_index = np.argmin(
            np.abs(J_valid)
        )

        Voc = V_valid[voc_index]

        if Voc <= 0:
            continue

        # ----------------------------------------------------
        # Define voltage regions relative to Voc
        # ----------------------------------------------------

        low_mask = (
            V <= 0.30 * Voc
        )

        knee_mask = (
            (V > 0.30 * Voc)
            &
            (V <= 0.80 * Voc)
        )

        high_mask = (
            V > 0.80 * Voc
        )

        # ----------------------------------------------------
        # Function for RMSE
        # ----------------------------------------------------

        def rmse(true, pred):

            valid = (
                np.isfinite(true)
                &
                np.isfinite(pred)
            )

            if not np.any(valid):
                return np.nan

            return np.sqrt(
                np.mean(
                    (
                        true[valid]
                        -
                        pred[valid]
                    ) ** 2
                )
            )

        # ----------------------------------------------------
        # Calculate regional errors
        # ----------------------------------------------------

        physics_low = rmse(
            true_curve[low_mask],
            physics_curve[low_mask]
        )

        final_low = rmse(
            true_curve[low_mask],
            pred_curve[low_mask]
        )

        physics_knee = rmse(
            true_curve[knee_mask],
            physics_curve[knee_mask]
        )

        final_knee = rmse(
            true_curve[knee_mask],
            pred_curve[knee_mask]
        )

        physics_high = rmse(
            true_curve[high_mask],
            physics_curve[high_mask]
        )

        final_high = rmse(
            true_curve[high_mask],
            pred_curve[high_mask]
        )

        # ----------------------------------------------------
        # Improvement calculation
        # ----------------------------------------------------

        def improvement(
            physics,
            final
        ):

            if (
                not np.isfinite(physics)
                or physics <= 1e-12
            ):
                return np.nan

            return (
                (physics - final)
                / physics
            ) * 100.0

        rows.append({

            "Curve": idx,

            "Voc": Voc,

            "Physics_Low_RMSE":
                physics_low,

            "Final_Low_RMSE":
                final_low,

            "Low_Improvement_%":
                improvement(
                    physics_low,
                    final_low
                ),

            "Physics_Knee_RMSE":
                physics_knee,

            "Final_Knee_RMSE":
                final_knee,

            "Knee_Improvement_%":
                improvement(
                    physics_knee,
                    final_knee
                ),

            "Physics_High_RMSE":
                physics_high,

            "Final_High_RMSE":
                final_high,

            "High_Improvement_%":
                improvement(
                    physics_high,
                    final_high
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# PROCESS EACH PANEL
# ============================================================

for panel_name, file_path in panel_files.items():

    print("\n" + "=" * 60)
    print(f"Processing: {panel_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load curves
    # --------------------------------------------------------

    if not os.path.exists(file_path):

        print(
            f"ERROR: File not found:\n{file_path}"
        )

        continue

    data = np.load(
        file_path
    )

    V = data["V"]

    J_true = data["J_true"]

    J_pred = data["J_pred"]

    J_physics = data["J_physics"]

    print(
        f"Number of curves: {J_true.shape[0]}"
    )

    print(
        f"Voltage points: {J_true.shape[1]}"
    )

    # --------------------------------------------------------
    # Calculate voltage-dependent errors
    # --------------------------------------------------------

    (
        physics_rmse,
        final_rmse,
        physics_mae,
        final_mae,
        improvement
    ) = calculate_voltage_errors(
        J_true,
        J_pred,
        J_physics
    )

    # ========================================================
    # SAVE VOLTAGE-DEPENDENT RESULTS
    # ========================================================

    voltage_results = pd.DataFrame({

        "Voltage": V,

        "Physics_RMSE":
            physics_rmse,

        "Final_RMSE":
            final_rmse,

        "Physics_MAE":
            physics_mae,

        "Final_MAE":
            final_mae,

        "RMSE_Improvement_%":
            improvement
    })

    voltage_csv = (
        output_dir /
        f"{panel_name}_voltage_error.csv"
    )

    voltage_results.to_csv(
        voltage_csv,
        index=False
    )

    print(
        f"\nSaved:\n{voltage_csv}"
    )

    # ========================================================
    # PLOT 1
    # RMSE VS VOLTAGE
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        V,
        physics_rmse,
        linewidth=2,
        label="Single-Diode Physics"
    )

    plt.plot(
        V,
        final_rmse,
        linewidth=2,
        label="Residual-Corrected Model"
    )

    plt.xlabel(
        "Voltage (V)",
        fontsize=12
    )

    plt.ylabel(
        "J–V RMSE",
        fontsize=12
    )

    plt.title(
        f"Voltage-Dependent J–V RMSE — {panel_name}",
        fontsize=14
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{panel_name}_RMSE_vs_voltage.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # ========================================================
    # PLOT 2
    # MAE VS VOLTAGE
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        V,
        physics_mae,
        linewidth=2,
        label="Single-Diode Physics"
    )

    plt.plot(
        V,
        final_mae,
        linewidth=2,
        label="Residual-Corrected Model"
    )

    plt.xlabel(
        "Voltage (V)",
        fontsize=12
    )

    plt.ylabel(
        "J–V MAE",
        fontsize=12
    )

    plt.title(
        f"Voltage-Dependent J–V MAE — {panel_name}",
        fontsize=14
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{panel_name}_MAE_vs_voltage.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # ========================================================
    # PLOT 3
    # RMSE IMPROVEMENT VS VOLTAGE
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        V,
        improvement,
        linewidth=2
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Voltage (V)",
        fontsize=12
    )

    plt.ylabel(
        "RMSE Improvement (%)",
        fontsize=12
    )

    plt.title(
        f"Residual Correction Improvement vs Voltage — {panel_name}",
        fontsize=14
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{panel_name}_RMSE_improvement_vs_voltage.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # ========================================================
    # REGIONAL ANALYSIS
    # ========================================================

    regional_df = calculate_regional_errors(
        V,
        J_true,
        J_pred,
        J_physics
    )

    regional_csv = (
        output_dir /
        f"{panel_name}_regional_error.csv"
    )

    regional_df.to_csv(
        regional_csv,
        index=False
    )

    print(
        f"Saved:\n{regional_csv}"
    )

    # ========================================================
    # REGIONAL SUMMARY
    # ========================================================

    if len(regional_df) > 0:

        print(
            "\nRegional RMSE:"
        )

        print(
            "\nLOW-VOLTAGE REGION"
        )

        print(
            f"Physics: "
            f"{regional_df['Physics_Low_RMSE'].mean():.6f}"
        )

        print(
            f"Final:   "
            f"{regional_df['Final_Low_RMSE'].mean():.6f}"
        )

        print(
            f"Improvement: "
            f"{regional_df['Low_Improvement_%'].mean():.2f}%"
        )

        print(
            "\nKNEE / INTERMEDIATE-VOLTAGE REGION"
        )

        print(
            f"Physics: "
            f"{regional_df['Physics_Knee_RMSE'].mean():.6f}"
        )

        print(
            f"Final:   "
            f"{regional_df['Final_Knee_RMSE'].mean():.6f}"
        )

        print(
            f"Improvement: "
            f"{regional_df['Knee_Improvement_%'].mean():.2f}%"
        )

        print(
            "\nHIGH-VOLTAGE REGION"
        )

        print(
            f"Physics: "
            f"{regional_df['Physics_High_RMSE'].mean():.6f}"
        )

        print(
            f"Final:   "
            f"{regional_df['Final_High_RMSE'].mean():.6f}"
        )

        print(
            f"Improvement: "
            f"{regional_df['High_Improvement_%'].mean():.2f}%"
        )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "Voltage-dependent JV analysis complete."
)

print(
    f"Results saved to:\n{output_dir.resolve()}"
)

print(
    "=" * 60
)