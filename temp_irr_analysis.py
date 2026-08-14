import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from model_autoencoder import PhysicsInformedJVModel
from jv_analysis import extract_jv_physics

# ============================================================
# Configuration
# ============================================================

DIODE_PARAMS = [
    "Iph",
    "Io",
    "n",
    "Rs",
    "Rsh",
]

ELECTRICAL_PARAMS = [
    "Voc",
    "Jsc",
    "Vmpp",
    "Jmpp",
    "Pmpp",
    "FF",
]

ALL_PARAMS = DIODE_PARAMS + ELECTRICAL_PARAMS



# ============================================================
# Load model
# ============================================================

def load_model(model_path, dataset):

    V_grid = dataset["V_grid"].astype(np.float32)

    weather = dataset["weather"].astype(np.float32)

    ckpt = torch.load(
        model_path,
        map_location="cpu"
    )

    model = PhysicsInformedJVModel(
        jv_dim=dataset["jv_target"].shape[1],
        weather_dim=weather.shape[1],
        voltage_grid=V_grid,
        latent_dim=ckpt["latent_dim"],
        diode_mean=ckpt["diode_mean"],
        diode_std=ckpt["diode_std"],
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return model, V_grid


# ============================================================
# Generate predictions
# ============================================================

def generate_predictions(
    model,
    dataset,
    device,
):

    weather = dataset["weather"].astype(
        np.float32
    )

    jv_true = dataset["jv_target"].astype(
        np.float32
    )

    V_grid = dataset["V_grid"].astype(
        np.float32
    )

    # --------------------------------------------------------
    # Fitted diode parameters
    # --------------------------------------------------------

    if "diode_params" in dataset:

        diode_true = dataset[
            "diode_params"
        ].astype(np.float32)

    elif "diode_params_norm" in dataset:

        diode_norm = dataset[
            "diode_params_norm"
        ].astype(np.float32)

        if "diode_params_stats" in dataset:

            stats = dataset[
                "diode_params_stats"
            ].item()

            diode_true = (
                diode_norm
                * stats["std"]
                + stats["mean"]
            )

        else:

            raise ValueError(
                "Found diode_params_norm but "
                "no diode_params_stats."
            )

    else:

        raise ValueError(
            "Dataset must contain either "
            "'diode_params' or "
            "'diode_params_norm'."
        )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    weather_t = torch.tensor(
        weather,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        (
            I_pred,
            I_phys,
            delta_I,
            z,
            diode_pred_norm,
        ) = model(weather_t)

    I_pred = (
        I_pred
        .detach()
        .cpu()
        .numpy()
    )

    diode_pred_norm = (
        diode_pred_norm
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Convert predicted normalized diode parameters using
    # the SAME statistics used by the trained model.
    # --------------------------------------------------------

    diode_mean = (
        model.diode_mean
        .detach()
        .cpu()
        .numpy()
    )

    diode_std = (
        model.diode_std
        .detach()
        .cpu()
        .numpy()
    )

    diode_pred = (
        diode_pred_norm
        * diode_std
        + diode_mean
    )

    # --------------------------------------------------------
    # Electrical parameters
    #
    # IMPORTANT:
    #
    # These are calculated from FINAL JV prediction:
    #
    # I_pred = I_phys + delta_I
    #
    # NOT from I_phys.
    # --------------------------------------------------------

    electrical_true = []

    electrical_pred = []

    for i in range(len(jv_true)):

        true_values = extract_jv_physics(
            V_grid,
            jv_true[i], area_m2=1
        )

        pred_values = extract_jv_physics(
            V_grid,
            I_pred[i], area_m2=1
        )

        electrical_true.append(
            true_values
        )

        electrical_pred.append(
            pred_values
        )

    electrical_true = np.asarray(
        electrical_true,
        dtype=np.float32
    )

    electrical_pred = np.asarray(
        electrical_pred,
        dtype=np.float32
    )

    return (
        weather,
        diode_true,
        diode_pred,
        electrical_true,
        electrical_pred,
        I_pred,
        jv_true,
    )


# ============================================================
# Build analysis dataframe
# ============================================================

def build_dataframe(
    weather,
    diode_true,
    diode_pred,
    electrical_true,
    electrical_pred,
):

    rows = []

    for i in range(len(weather)):

        # ----------------------------------------------------
        # Assumption:
        #
        # weather[:,0] = module temperature
        # weather[:,1] = irradiance
        #
        # If your dataset uses a different ordering,
        # change these two lines.
        # ----------------------------------------------------

        temperature = weather[i, 0]

        irradiance = weather[i, 1]

        row = {

            "index": i,

            "temperature": temperature,

            "irradiance": irradiance,
        }

        # ----------------------------------------------------
        # Diode parameters
        # ----------------------------------------------------

        for j, name in enumerate(
            DIODE_PARAMS
        ):

            row[
                f"{name}_true"
            ] = diode_true[i, j]

            row[
                f"{name}_pred"
            ] = diode_pred[i, j]

            row[
                f"{name}_error"
            ] = (
                diode_pred[i, j]
                - diode_true[i, j]
            )

        # ----------------------------------------------------
        # Electrical parameters
        # ----------------------------------------------------

        for j, name in enumerate(
            ELECTRICAL_PARAMS
        ):

            row[
                f"{name}_true"
            ] = electrical_true[i, j]

            row[
                f"{name}_pred"
            ] = electrical_pred[i, j]

            row[
                f"{name}_error"
            ] = (
                electrical_pred[i, j]
                - electrical_true[i, j]
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 3D surface plotting
# ============================================================

def plot_surface(
    df,
    parameter,
    value_type,
    output_path,
    grid_resolution=60,
):
    """
    value_type:
        true
        pred
        error
    """

    x = df["temperature"].values
    y = df["irradiance"].values

    z = df[
        f"{parameter}_{value_type}"
    ].values

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    x = x[mask]
    y = y[mask]
    z = z[mask]

    if len(x) < 4:

        print(
            f"Not enough points for "
            f"{parameter} {value_type}"
        )

        return

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    x_grid = np.linspace(
        x.min(),
        x.max(),
        grid_resolution
    )

    y_grid = np.linspace(
        y.min(),
        y.max(),
        grid_resolution
    )

    X, Y = np.meshgrid(
        x_grid,
        y_grid
    )

    # --------------------------------------------------------
    # Linear interpolation
    # --------------------------------------------------------

    Z = griddata(
        points=(x, y),
        values=z,
        xi=(X, Y),
        method="linear"
    )

    # --------------------------------------------------------
    # Fill outside convex hull using nearest
    #
    # This is useful for visualization, but we explicitly
    # mark that these areas are extrapolated/interpolated.
    # --------------------------------------------------------

    Z_nearest = griddata(
        points=(x, y),
        values=z,
        xi=(X, Y),
        method="nearest"
    )

    missing = ~np.isfinite(Z)

    Z[missing] = Z_nearest[missing]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="none",
        alpha=0.85
    )

    # --------------------------------------------------------
    # Actual observations
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        z,
        color="black",
        s=15,
        alpha=0.5,
        label="Observed samples"
    )

    ax.set_xlabel(
        "Module Temperature (°C)"
    )

    ax.set_ylabel(
        "Irradiance (W/m²)"
    )

    if value_type == "true":

        title = (
            f"{parameter}: "
            "Fitted / True Response Surface"
        )

        zlabel = (
            f"{parameter} true"
        )

    elif value_type == "pred":

        title = (
            f"{parameter}: "
            "Predicted Response Surface"
        )

        zlabel = (
            f"{parameter} predicted"
        )

    else:

        title = (
            f"{parameter}: "
            "Prediction Error Surface"
        )

        zlabel = (
            f"{parameter} error"
        )

    ax.set_zlabel(
        zlabel
    )

    ax.set_title(title)

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.6,
        pad=0.1
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Actual observation scatter in T-G plane
# ============================================================

def plot_sampling_domain(
    df,
    output_path,
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        df["temperature"],
        df["irradiance"],
        s=20,
        alpha=0.7
    )

    plt.xlabel(
        "Module Temperature (°C)"
    )

    plt.ylabel(
        "Irradiance (W/m²)"
    )

    plt.title(
        "Observed Temperature–Irradiance Domain"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250
    )

    plt.close()


# ============================================================
# Temperature slices
# ============================================================

def plot_temperature_slices(
    df,
    parameter,
    output_dir,
    n_slices=5,
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    irradiance = df[
        "irradiance"
    ].values

    temperature = df[
        "temperature"
    ].values

    pred = df[
        f"{parameter}_pred"
    ].values

    true = df[
        f"{parameter}_true"
    ].values

    finite = (
        np.isfinite(
            irradiance
        )
        &
        np.isfinite(
            temperature
        )
        &
        np.isfinite(pred)
        &
        np.isfinite(true)
    )

    irradiance = irradiance[finite]

    temperature = temperature[finite]

    pred = pred[finite]

    true = true[finite]

    if len(temperature) < 5:

        return

    # --------------------------------------------------------
    # Select temperature levels based on quantiles
    # --------------------------------------------------------

    levels = np.linspace(
        np.min(temperature),
        np.max(temperature),
        n_slices
    )

    tolerance = (
        np.ptp(temperature)
        / max(n_slices * 2, 1)
    )

    plt.figure(
        figsize=(9, 6)
    )

    for T in levels:

        mask = (
            np.abs(
                temperature - T
            )
            <= tolerance
        )

        if np.sum(mask) < 3:

            continue

        order = np.argsort(
            irradiance[mask]
        )

        G = irradiance[mask][order]

        p = pred[mask][order]

        t = true[mask][order]

        plt.plot(
            G,
            p,
            "--",
            marker="o",
            label=f"T ≈ {T:.1f} °C"
        )

        plt.plot(
            G,
            t,
            "-",
            alpha=0.7
        )

    plt.xlabel(
        "Irradiance (W/m²)"
    )

    plt.ylabel(
        parameter
    )

    plt.title(
        f"{parameter}: Irradiance Dependence "
        "at Temperature Slices"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{parameter}_temperature_slices.png"
        ),
        dpi=250
    )

    plt.close()


# ============================================================
# Irradiance slices
# ============================================================

def plot_irradiance_slices(
    df,
    parameter,
    output_dir,
    n_slices=5,
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    irradiance = df[
        "irradiance"
    ].values

    temperature = df[
        "temperature"
    ].values

    pred = df[
        f"{parameter}_pred"
    ].values

    true = df[
        f"{parameter}_true"
    ].values

    finite = (
        np.isfinite(
            irradiance
        )
        &
        np.isfinite(
            temperature
        )
        &
        np.isfinite(pred)
        &
        np.isfinite(true)
    )

    irradiance = irradiance[finite]

    temperature = temperature[finite]

    pred = pred[finite]

    true = true[finite]

    if len(irradiance) < 5:

        return

    levels = np.linspace(
        np.min(irradiance),
        np.max(irradiance),
        n_slices
    )

    tolerance = (
        np.ptp(irradiance)
        / max(n_slices * 2, 1)
    )

    plt.figure(
        figsize=(9, 6)
    )

    for G in levels:

        mask = (
            np.abs(
                irradiance - G
            )
            <= tolerance
        )

        if np.sum(mask) < 3:

            continue

        order = np.argsort(
            temperature[mask]
        )

        T = temperature[mask][order]

        p = pred[mask][order]

        t = true[mask][order]

        plt.plot(
            T,
            p,
            "--",
            marker="o",
            label=f"G ≈ {G:.0f} W/m²"
        )

        plt.plot(
            T,
            t,
            "-",
            alpha=0.7
        )

    plt.xlabel(
        "Module Temperature (°C)"
    )

    plt.ylabel(
        parameter
    )

    plt.title(
        f"{parameter}: Temperature Dependence "
        "at Irradiance Slices"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{parameter}_irradiance_slices.png"
        ),
        dpi=250
    )

    plt.close()


# ============================================================
# Save compact surface grids
# ============================================================

def save_surface_grid(
    df,
    parameter,
    value_type,
    output_path,
    grid_resolution=60,
):

    x = df[
        "temperature"
    ].values

    y = df[
        "irradiance"
    ].values

    z = df[
        f"{parameter}_{value_type}"
    ].values

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
        &
        np.isfinite(z)
    )

    x = x[mask]
    y = y[mask]
    z = z[mask]

    if len(x) < 4:

        return

    x_grid = np.linspace(
        x.min(),
        x.max(),
        grid_resolution
    )

    y_grid = np.linspace(
        y.min(),
        y.max(),
        grid_resolution
    )

    X, Y = np.meshgrid(
        x_grid,
        y_grid
    )

    Z = griddata(
        (x, y),
        z,
        (X, Y),
        method="linear"
    )

    np.savez(
        output_path,
        temperature_grid=X,
        irradiance_grid=Y,
        value_grid=Z
    )


# ============================================================
# Main analysis
# ============================================================

def run_level4_analysis(
    model_path,
    dataset_path,
    tech,
    output_root="outputs",
    grid_resolution=60,
    n_slices=5,
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    base_dir = os.path.join(
        output_root,
        tech,
        "level4_response_surfaces"
    )

    surface_dir = os.path.join(
        base_dir,
        "3d_surfaces"
    )

    slice_dir = os.path.join(
        base_dir,
        "slices"
    )

    grid_dir = os.path.join(
        base_dir,
        "surface_grids"
    )

    data_dir = os.path.join(
        base_dir,
        "data"
    )

    os.makedirs(
        surface_dir,
        exist_ok=True
    )

    os.makedirs(
        slice_dir,
        exist_ok=True
    )

    os.makedirs(
        grid_dir,
        exist_ok=True
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    dataset = np.load(
        dataset_path,
        allow_pickle=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading model..."
    )

    model, V_grid = load_model(
        model_path,
        dataset
    )

    model.to(device)

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "Running inference..."
    )

    (
        weather,
        diode_true,
        diode_pred,
        electrical_true,
        electrical_pred,
        I_pred,
        jv_true,
    ) = generate_predictions(
        model,
        dataset,
        device
    )

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------

    df = build_dataframe(
        weather,
        diode_true,
        diode_pred,
        electrical_true,
        electrical_pred,
    )

    # --------------------------------------------------------
    # Save complete point-wise dataset
    # --------------------------------------------------------

    csv_path = os.path.join(
        data_dir,
        "response_surface_data.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nSaved point-wise data:\n"
        f"{csv_path}"
    )

    # --------------------------------------------------------
    # Sampling domain
    # --------------------------------------------------------

    plot_sampling_domain(
        df,
        os.path.join(
            base_dir,
            "temperature_irradiance_sampling.png"
        )
    )

    # --------------------------------------------------------
    # Analyze every parameter
    # --------------------------------------------------------

    for parameter in ALL_PARAMS:

        print(
            f"\nAnalyzing {parameter}..."
        )

        parameter_surface_dir = os.path.join(
            surface_dir,
            parameter
        )

        parameter_slice_dir = os.path.join(
            slice_dir,
            parameter
        )

        os.makedirs(
            parameter_surface_dir,
            exist_ok=True
        )

        os.makedirs(
            parameter_slice_dir,
            exist_ok=True
        )

        # ====================================================
        # 3D surfaces
        # ====================================================

        for value_type in [
            "true",
            "pred",
            "error",
        ]:

            plot_surface(
                df=df,
                parameter=parameter,
                value_type=value_type,
                output_path=os.path.join(
                    parameter_surface_dir,
                    f"{value_type}_surface.png"
                ),
                grid_resolution=grid_resolution,
            )

            save_surface_grid(
                df=df,
                parameter=parameter,
                value_type=value_type,
                output_path=os.path.join(
                    grid_dir,
                    f"{parameter}_{value_type}.npz"
                ),
                grid_resolution=grid_resolution,
            )

        # ====================================================
        # Temperature slices
        # ====================================================

        plot_temperature_slices(
            df=df,
            parameter=parameter,
            output_dir=parameter_slice_dir,
            n_slices=n_slices,
        )

        # ====================================================
        # Irradiance slices
        # ====================================================

        plot_irradiance_slices(
            df=df,
            parameter=parameter,
            output_dir=parameter_slice_dir,
            n_slices=n_slices,
        )

    # --------------------------------------------------------
    # Save raw predicted JV curves too
    # --------------------------------------------------------

    np.savez(
        os.path.join(
            data_dir,
            "predicted_jv.npz"
        ),
        V_grid=V_grid,
        J_true=jv_true,
        J_pred=I_pred,
        weather=weather,
    )

    print(
        "\n=========================================="
    )

    print(
        "Level 4 response-surface analysis complete"
    )

    print(
        "=========================================="
    )

    print(
        f"Output directory:\n{base_dir}"
    )


# ============================================================
# Command line
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Level 4: Analyze predicted diode and "
            "electrical parameters as functions of "
            "module temperature and irradiance."
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained .pth model"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset .npz"
    )

    parser.add_argument(
        "--tech",
        type=str,
        required=True,
        help=(
            "Technology name, e.g. "
            "Perovskite_5&6"
        )
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs"
    )

    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=60
    )

    parser.add_argument(
        "--n-slices",
        type=int,
        default=5
    )

    args = parser.parse_args()

    run_level4_analysis(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        tech=args.tech,
        output_root=args.output_root,
        grid_resolution=args.grid_resolution,
        n_slices=args.n_slices,
    )