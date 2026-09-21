import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

NPZ_FILE = "outputs/SunPower_5-1/ae/eval_with_temporal/metrics/curves.npz"

# Perturbation values in A/m^2
# Example: shift the physics curve from -1 to +1 A/m^2
PERTURBATIONS = np.linspace(-1.0, 1.0, 41)

# Output directory
OUTPUT_DIR = "outputs/SunPower_5-1/physics_perturbation_analysis_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(NPZ_FILE, allow_pickle=True)

print("Available NPZ keys:")
print(data.files)


# ------------------------------------------------------------
# Helper function to find a key from possible names
# ------------------------------------------------------------

def get_array(data, possible_names, required=True):
    for name in possible_names:
        if name in data.files:
            return np.asarray(data[name], dtype=float)

    if required:
        raise KeyError(
            f"Could not find any of these keys: {possible_names}\n"
            f"Available keys: {data.files}"
        )

    return None


# ============================================================
# EXTRACT ARRAYS
# ============================================================

V = get_array(
    data,
    ["V", "voltage", "voltages", "V_grid"]
)

Jphys = get_array(
    data,
    ["J_physics", "jphys", "I_phys", "Iphys"]
)

Jpred = get_array(
    data,
    ["J_pred", "jpred", "I_pred", "Ipred", "prediction", "predicted"]
)

Jtrue = get_array(
    data,
    ["J_true", "jtrue", "I_true", "Itrue", "target", "true"]
)


# ============================================================
# CHECK SHAPES
# ============================================================

print("\nArray shapes:")
print("V     :", V.shape)
print("J_phys :", Jphys.shape)
print("J_pred :", Jpred.shape)
print("J_true :", Jtrue.shape)


if Jphys.shape != Jpred.shape:
    raise ValueError("Jphys and Jpred must have the same shape.")

if Jphys.shape != Jtrue.shape:
    raise ValueError("Jphys and Jtrue must have the same shape.")


# ============================================================
# HANDLE DIFFERENT DATA ORGANIZATIONS
# ============================================================
#
# Expected:
#
#   (number_of_samples, number_of_voltage_points)
#
# Example:
#   (5000, 200)
#
# If your arrays are instead (200, samples), transpose them.
#

if Jphys.ndim != 2:
    raise ValueError(
        "Expected Jphys to be a 2D array: "
        "(samples, voltage_points)"
    )

n_samples, n_voltage = Jphys.shape

print(f"\nNumber of samples       : {n_samples}")
print(f"Voltage points per curve: {n_voltage}")


# ------------------------------------------------------------
# Voltage array
# ------------------------------------------------------------

if V.ndim == 1:

    if len(V) != n_voltage:
        raise ValueError(
            f"V has {len(V)} points but curves have "
            f"{n_voltage} voltage points."
        )

elif V.ndim == 2:

    if V.shape != Jphys.shape:
        raise ValueError(
            "2D V array must have the same shape as Jphys."
        )

else:
    raise ValueError("V must be either 1D or 2D.")


# ============================================================
# REMOVE INVALID CURVES
# ============================================================

valid_samples = (
    np.all(np.isfinite(Jphys), axis=1)
    & np.all(np.isfinite(Jpred), axis=1)
    & np.all(np.isfinite(Jtrue), axis=1)
)

if V.ndim == 2:
    valid_samples &= np.all(np.isfinite(V), axis=1)

print(
    f"\nValid samples: "
    f"{valid_samples.sum()} / {len(valid_samples)}"
)

Jphys = Jphys[valid_samples]
Jpred = Jpred[valid_samples]
Jtrue = Jtrue[valid_samples]

if V.ndim == 2:
    V = V[valid_samples]


# ============================================================
# ORIGINAL RESIDUAL COMPONENT
# ============================================================
#
# Since:
#
#       Jpred = Jphys + Jresidual
#
# the learned residual component can be recovered as:
#
#       Jresidual = Jpred - Jphys
#
# We then KEEP THIS RESIDUAL FIXED.
#

Jresidual = Jpred - Jphys


# Verify reconstruction
Jpred_reconstructed = Jphys + Jresidual

reconstruction_error = np.max(
    np.abs(Jpred_reconstructed - Jpred)
)

print(
    f"\nMaximum reconstruction error: "
    f"{reconstruction_error:.3e}"
)


# ============================================================
# RMSE FUNCTION
# ============================================================

def calculate_curve_rmse(predicted, true):
    """
    Calculate one RMSE value per J-V curve.
    """

    return np.sqrt(
        np.mean((predicted - true) ** 2, axis=1)
    )


# ============================================================
# ORIGINAL RMSE VALUES
# ============================================================

physics_rmse_original = calculate_curve_rmse(
    Jphys,
    Jtrue
)

final_rmse_original = calculate_curve_rmse(
    Jpred,
    Jtrue
)


print("\nOriginal results")
print("----------------")
print(
    f"Physics RMSE : "
    f"{np.mean(physics_rmse_original):.6f}"
)

print(
    f"Final RMSE   : "
    f"{np.mean(final_rmse_original):.6f}"
)


# ============================================================
# PERTURBATION ANALYSIS
# ============================================================

mean_final_rmse = []
mean_physics_rmse = []

rmse_change_final = []
rmse_change_physics = []

# Store individual sample RMSE values too
all_final_rmse = []
all_physics_rmse = []


for delta in PERTURBATIONS:

    # --------------------------------------------------------
    # Perturb ONLY the physics component
    # --------------------------------------------------------

    Jphys_perturbed = Jphys + delta

    # --------------------------------------------------------
    # Physics-only curve
    # --------------------------------------------------------

    physics_rmse = calculate_curve_rmse(
        Jphys_perturbed,
        Jtrue
    )

    # --------------------------------------------------------
    # Keep the learned residual FIXED
    #
    # Jpred* =
    #     Jphys_perturbed + Jresidual
    #
    # Since Jresidual = Jpred - Jphys:
    #
    # Jpred* = Jpred + delta
    # --------------------------------------------------------

    Jpred_perturbed = (
        Jphys_perturbed
        + Jresidual
    )

    # --------------------------------------------------------
    # Final prediction RMSE
    # --------------------------------------------------------

    final_rmse = calculate_curve_rmse(
        Jpred_perturbed,
        Jtrue
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    mean_physics_rmse.append(
        np.mean(physics_rmse)
    )

    mean_final_rmse.append(
        np.mean(final_rmse)
    )

    rmse_change_physics.append(
        np.mean(physics_rmse)
        - np.mean(physics_rmse_original)
    )

    rmse_change_final.append(
        np.mean(final_rmse)
        - np.mean(final_rmse_original)
    )

    all_physics_rmse.append(physics_rmse)
    all_final_rmse.append(final_rmse)


mean_physics_rmse = np.asarray(mean_physics_rmse)
mean_final_rmse = np.asarray(mean_final_rmse)

rmse_change_physics = np.asarray(rmse_change_physics)
rmse_change_final = np.asarray(rmse_change_final)

all_physics_rmse = np.asarray(all_physics_rmse)
all_final_rmse = np.asarray(all_final_rmse)


# ============================================================
# FIND OPTIMAL CONSTANT SHIFT
# ============================================================

best_index = np.argmin(mean_final_rmse)

best_delta = PERTURBATIONS[best_index]
best_rmse = mean_final_rmse[best_index]

print("\nPerturbation analysis")
print("---------------------")

print(
    f"Original final RMSE : "
    f"{np.mean(final_rmse_original):.6f}"
)

print(
    f"Best perturbation    : "
    f"{best_delta:+.6f} A/m²"
)

print(
    f"RMSE at best shift   : "
    f"{best_rmse:.6f}"
)

print(
    f"RMSE change          : "
    f"{best_rmse - np.mean(final_rmse_original):+.6f}"
)


# ============================================================
# THEORETICAL OPTIMAL SHIFT
# ============================================================
#
# For:
#
#   Jpred* = Jpred + delta
#
# the RMSE-minimizing constant shift is:
#
#   delta_opt = -mean(Jpred - Jtrue)
#
# ------------------------------------------------------------

prediction_error = Jpred - Jtrue

theoretical_delta = np.mean(prediction_error)

print(
    f"\nTheoretical optimal constant shift: "
    f"{theoretical_delta:+.6f} A/m²"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = np.column_stack(
    [
        PERTURBATIONS,
        mean_physics_rmse,
        mean_final_rmse,
        rmse_change_physics,
        rmse_change_final
    ]
)

header = (
    "delta_Jphys,"
    "mean_physics_RMSE,"
    "mean_final_RMSE,"
    "physics_RMSE_change,"
    "final_RMSE_change"
)

csv_file = os.path.join(
    OUTPUT_DIR,
    "physics_perturbation_results.csv"
)

np.savetxt(
    csv_file,
    results,
    delimiter=",",
    header=header,
    comments=""
)

print(f"\nResults saved to:")
print(csv_file)


# ============================================================
# PLOT 1
# FINAL J-V RMSE VS PHYSICS PERTURBATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    PERTURBATIONS,
    mean_final_rmse,
    marker="o",
    markersize=4,
    linewidth=2,
    label="Final prediction"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
    label="Original physics"
)

plt.scatter(
    best_delta,
    best_rmse,
    s=70,
    zorder=5,
    label=f"Minimum RMSE ({best_delta:+.3f} A/m²)"
)

plt.xlabel(r"Perturbation of $J_{\mathrm{phys}}$ (A/m$^2$)")
plt.ylabel("J–V RMSE (A/m²)")
plt.title(
    "Sensitivity of J–V RMSE to Physics-Based Current Perturbation"
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot1 = os.path.join(
    OUTPUT_DIR,
    "final_rmse_vs_physics_perturbation.png"
)

plt.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 2
# RMSE CHANGE RELATIVE TO ORIGINAL
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    PERTURBATIONS,
    rmse_change_final,
    marker="o",
    markersize=4,
    linewidth=2
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel(r"Perturbation of $J_{\mathrm{phys}}$ (A/m$^2$)")
plt.ylabel(r"$\Delta$ J–V RMSE (A/m$^2$)")
plt.title(
    "Change in J–V RMSE Relative to Original Prediction"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plot2 = os.path.join(
    OUTPUT_DIR,
    "change_in_final_rmse.png"
)

plt.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 3
# PHYSICS RMSE VS FINAL RMSE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    PERTURBATIONS,
    mean_physics_rmse,
    marker="o",
    markersize=4,
    linewidth=2,
    label="Physics-only RMSE"
)

plt.plot(
    PERTURBATIONS,
    mean_final_rmse,
    marker="o",
    markersize=4,
    linewidth=2,
    label="Final prediction RMSE"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel(r"Perturbation of $J_{\mathrm{phys}}$ (A/m$^2$)")
plt.ylabel("J–V RMSE (A/m²)")
plt.title(
    "Physics and Final J–V RMSE Under Physics Perturbation"
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot3 = os.path.join(
    OUTPUT_DIR,
    "physics_vs_final_rmse.png"
)

plt.savefig(
    plot3,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 4
# EXAMPLE CURVE BEFORE AND AFTER PERTURBATION
# ============================================================

sample_index = 0

# Choose a perturbation for visualization
example_delta = 1.0

Jphys_example = (
    Jphys[sample_index]
    + example_delta
)

Jpred_example = (
    Jpred[sample_index]
    + example_delta
)

Jtrue_example = Jtrue[sample_index]

if V.ndim == 1:
    V_example = V
else:
    V_example = V[sample_index]


plt.figure(figsize=(8, 5))

plt.plot(
    V_example,
    Jtrue_example,
    linewidth=2,
    label="Measured J–V"
)

plt.plot(
    V_example,
    Jphys[sample_index],
    linewidth=2,
    label="Original physics"
)

plt.plot(
    V_example,
    Jphys_example,
    linestyle="--",
    linewidth=2,
    label=fr"Physics + {example_delta:g} A/m²"
)

plt.plot(
    V_example,
    Jpred[sample_index],
    linewidth=2,
    label="Original prediction"
)

plt.plot(
    V_example,
    Jpred_example,
    linestyle="--",
    linewidth=2,
    label=fr"Prediction + {example_delta:g} A/m²"
)

plt.xlabel("Voltage (V)")
plt.ylabel("Current density (A/m²)")
plt.title(
    "Effect of Physics-Based Current Perturbation on J–V Curve"
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot4 = os.path.join(
    OUTPUT_DIR,
    "example_curve_perturbation.png"
)

plt.savefig(
    plot4,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nAll plots saved in:")
print(OUTPUT_DIR)