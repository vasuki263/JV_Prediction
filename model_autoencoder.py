import torch
import torch.nn as nn
import numpy as np

# ============================================================
# Differentiable Single-Diode Solver
# ============================================================
#
class SingleDiodeSolver(nn.Module):
    """
    Differentiable Newton solver for the single-diode equation.

    I = Iph
        - Io * (exp((V + I*Rs)/(n*Ns*Vt)) - 1)
        - (V + I*Rs)/Rsh
    """

    def __init__(
        self,
        Ns=1,
        Vt=0.02585,
        n_iter=40,
    ):
        super().__init__()

        self.Ns = Ns
        self.Vt = Vt
        self.n_iter = n_iter

    def forward(
        self,
        voltage,
        Iph,
        Io,
        n,
        Rs,
        Rsh,
    ):

        # Initial current estimate
        I = Iph.clone()

        for _ in range(self.n_iter):

            exp_arg = (
                voltage + I * Rs
            ) / (
                n * self.Ns * self.Vt
            )

            # Prevent numerical overflow
            exp_arg = torch.clamp(
                exp_arg,
                min=-80.0,
                max=80.0,
            )

            exp_term = torch.exp(exp_arg)

            # ------------------------------------------------
            # Single diode equation
            # ------------------------------------------------

            f = (
                Iph
                - Io * (exp_term - 1.0)
                - (voltage + I * Rs) / Rsh
                - I
            )

            # ------------------------------------------------
            # Derivative df/dI
            # ------------------------------------------------

            df = (
                -Io
                * exp_term
                * (
                    Rs
                    / (
                        n
                        * self.Ns
                        * self.Vt
                    )
                )
                - Rs / Rsh
                - 1.0
            )

            # Newton update
            I = I - f / (
                df + 1e-8
            )

            # Numerical safety
            I = torch.nan_to_num(
                I,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )

        return I


# ============================================================
# Physics-Informed JV Model
# ============================================================

class PhysicsInformedJVModel(nn.Module):

    def __init__(
        self,
        jv_dim,
        weather_dim,
        voltage_grid,
        latent_dim=144,
        hidden_dims=(256, 128),
        diode_mean=None,
        diode_std=None,
    ):
        super().__init__()

        # ----------------------------------------------------
        # IMPORTANT
        #
        # jv_dim is ONLY the output dimension.
        #
        # It is NOT an input to the encoder.
        # ----------------------------------------------------

        self.jv_dim = jv_dim
        self.weather_dim = weather_dim
        self.latent_dim = latent_dim

        # ====================================================
        # Encoder
        # ====================================================

        # WEATHER ONLY
        input_dim = weather_dim

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:

            encoder_layers.append(
                nn.Linear(
                    prev_dim,
                    h
                )
            )

            encoder_layers.append(
                nn.ReLU()
            )

            prev_dim = h

        encoder_layers.append(
            nn.Linear(
                prev_dim,
                latent_dim
            )
        )

        self.encoder = nn.Sequential(
            *encoder_layers
        )

        # ====================================================
        # Conditioning
        # ====================================================

        z_cond_dim = (
            latent_dim
            + weather_dim
        )

        self.z_cond_dim = z_cond_dim

        # ====================================================
        # Diode Parameter Head
        # ====================================================

        # Predicts normalized:
        #
        # [Iph, Io, n_raw, Rs_raw, Rsh]
        #
        # These are supervised against the fitted
        # single-diode parameters extracted from the
        # measured JV curves.
        # ====================================================

        self.diode_head = nn.Sequential(

            nn.Linear(
                z_cond_dim,
                256
            ),

            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(
                256,
                256
            ),

            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                5
            )
        )

        # ====================================================
        # JV Residual Decoder
        # ====================================================

        self.residual_decoder = nn.Sequential(

            nn.Linear(
                z_cond_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                jv_dim
            ),
        )

        # ====================================================
        # Voltage Grid
        # ====================================================

        self.register_buffer(
            "voltage_grid",
            torch.tensor(
                voltage_grid,
                dtype=torch.float32
            )
        )

        # ====================================================
        # Diode normalization
        # ====================================================

        if diode_mean is None:
            diode_mean = np.zeros(5)

        if diode_std is None:
            diode_std = np.ones(5)

        self.register_buffer(
            "diode_mean",
            torch.as_tensor(
                diode_mean,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "diode_std",
            torch.as_tensor(
                diode_std,
                dtype=torch.float32
            )
        )

        # ====================================================
        # Physics solver
        # ====================================================

        self.solver = SingleDiodeSolver(
            Ns=144
        )

    # ========================================================
    # Convert normalized diode output to physical parameters
    # ========================================================

    def decode_diode_parameters(
        self,
        diode_pred_norm
    ):

        diode_raw = (
            diode_pred_norm
            * self.diode_std
            + self.diode_mean
        )

        # ----------------------------------------------------
        # Iph
        # ----------------------------------------------------

        Iph = diode_raw[:, 0]

        # ----------------------------------------------------
        # Io
        # ----------------------------------------------------

        Io = diode_raw[:, 1]

        # ----------------------------------------------------
        # Ideality factor
        #
        # n ∈ [0.8, 3.0]
        # ----------------------------------------------------

        n = (
            0.8
            + 2.2
            * torch.sigmoid(
                diode_raw[:, 2]
            )
        )

        # ----------------------------------------------------
        # Series resistance
        #
        # Rs ∈ [0, 20]
        # ----------------------------------------------------

        Rs = (
            20.0
            * torch.sigmoid(
                diode_raw[:, 3]
            )
        )

        # ----------------------------------------------------
        # Shunt resistance
        #
        # Rsh must be positive.
        #
        # Softplus avoids negative/zero values.
        # ----------------------------------------------------

        Rsh = (
            torch.nn.functional.softplus(
                diode_raw[:, 4]
            )
            + 1e-6
        )

        return (
            Iph,
            Io,
            n,
            Rs,
            Rsh,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        weather
    ):

        # ====================================================
        # 1. Weather -> latent
        # ====================================================

        z = self.encoder(
            weather
        )

        # ====================================================
        # 2. Latent + weather
        # ====================================================

        z_cond = torch.cat(
            [
                z,
                weather
            ],
            dim=-1
        )

        # ====================================================
        # 3. Predict normalized diode parameters
        # ====================================================

        diode_pred_norm = (
            self.diode_head(
                z_cond
            )
        )

        # ====================================================
        # 4. Convert diode parameters
        # ====================================================

        (
            Iph,
            Io,
            n,
            Rs,
            Rsh,
        ) = self.decode_diode_parameters(
            diode_pred_norm
        )

        # ====================================================
        # 5. Voltage grid
        # ====================================================

        batch_size = (
            weather.shape[0]
        )

        V = (
            self.voltage_grid
            .unsqueeze(0)
            .expand(
                batch_size,
                -1
            )
        )

        # ====================================================
        # 6. Physics JV
        # ====================================================

        I_phys = self.solver(

            V,

            Iph.unsqueeze(1),
            Io.unsqueeze(1),
            n.unsqueeze(1),
            Rs.unsqueeze(1),
            Rsh.unsqueeze(1),
        )

        # ====================================================
        # 7. Neural JV residual
        # ====================================================

        delta_I = (
            self.residual_decoder(
                z_cond
            )
        )

        # ====================================================
        # 8. Final JV
        # ====================================================

        I_pred = (
            I_phys
            + delta_I
        )

        return (
            I_pred,
            I_phys,
            delta_I,
            z,
            diode_pred_norm,
        )


