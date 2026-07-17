import torch
import torch.nn as nn
import numpy as np

class JVAutoencoder(nn.Module):
    """
    Conditional autoencoder for JV curve reconstruction.
    Input  = JV features + weather features
    Latent = compressed representation
    Output = reconstructed JV curve
    """

    def __init__(self, jv_dim, weather_dim, latent_dim=32, hidden_dims=(256, 128)):
        super().__init__()

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------
        input_dim = jv_dim + weather_dim
        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------
        decoder_input_dim = latent_dim + weather_dim
        decoder_layers = []
        prev_dim = decoder_input_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, jv_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------
    def forward(self, jv, weather):
        # Encode JV + weather
        x = torch.cat([jv, weather], dim=-1)
        z = self.encoder(x)

        # Decode latent + weather
        z_cond = torch.cat([z, weather], dim=-1)
        jv_recon = self.decoder(z_cond)

        return jv_recon, z

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleDiodeSolver(nn.Module):
    """
    Differentiable Newton solver for the single diode equation.

    I = Iph
        - Io*(exp((V + I*Rs)/(n*Vt)) - 1)
        - (V + I*Rs)/Rsh
    """

    def __init__(self, Ns=1, Vt=0.02585, n_iter=40):
        super().__init__()
        self.Ns = Ns
        self.Vt = Vt
        self.n_iter = n_iter

    def forward(self, voltage, Iph, Io, n, Rs, Rsh):

        I = Iph.clone()

        for _ in range(self.n_iter):

            exp_arg = (voltage + I * Rs) / (n * self.Ns * self.Vt)
            exp_arg = torch.clamp(exp_arg, max=80)

            exp_term = torch.exp(exp_arg)

            f = (
                Iph
                - Io * (exp_term - 1.0)
                - (voltage + I * Rs) / Rsh
                - I
            )

            df = (
                -Io * exp_term * (Rs / (n * self.Ns * self.Vt))
                - Rs / Rsh
                - 1.0
            )

            I = I - f / (df + 1e-8)

        return I


class PhysicsInformedJVModel(nn.Module):
    """
    Physics-informed JV model:
    - Encoder: JV + weather → latent z
    - Diode head: (z + weather) → normalized diode params
    - Solver: voltage_grid + *denormalized* diode params → I_phys
    - Residual decoder: (z + weather) → delta_I
    - Output: I_pred = I_phys + delta_I
    """

    def __init__(
        self,
        jv_dim,
        weather_dim,
        voltage_grid,
        latent_dim=32,
        hidden_dims=(256, 128),
        diode_mean=None,
        diode_std=None,
    ):
        super().__init__()

        # -----------------------------
        # Encoder
        # -----------------------------
        input_dim = jv_dim + weather_dim
        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # -----------------------------
        # Residual decoder (JV residual)
        # -----------------------------
        self.residual_decoder = nn.Sequential(
            nn.Linear(latent_dim + weather_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, jv_dim),
        )

        # -----------------------------
        # Diode parameter head
        # -----------------------------
        self.diode_head = nn.Sequential(
            nn.Linear(latent_dim + weather_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # Iph, Io, n, Rs, Rsh (normalized)
        )

        # -----------------------------
        # Voltage grid
        # -----------------------------
        self.register_buffer(
            "voltage_grid",
            torch.tensor(voltage_grid, dtype=torch.float32)
        )

        # -----------------------------
        # Normalization stats
        # -----------------------------
        if diode_mean is None:
            diode_mean = torch.zeros(5, dtype=torch.float32)
        if diode_std is None:
            diode_std = torch.ones(5, dtype=torch.float32)

        self.register_buffer("diode_mean", torch.tensor(diode_mean, dtype=torch.float32))
        self.register_buffer("diode_std", torch.tensor(diode_std, dtype=torch.float32))

        # -----------------------------
        # Solver
        # -----------------------------
        self.solver = SingleDiodeSolver(Ns=1)

    def forward(
            self,
            jv,
            weather,
            diode_params_norm=None,
    ):
        # -----------------------------
        # Encode JV + weather
        # -----------------------------
        x = torch.cat([jv, weather], dim=-1)
        z = self.encoder(x)

        # -----------------------------
        # Diode parameter prediction (normalized)
        # -----------------------------
        z_cond = torch.cat([z, weather], dim=-1)
        diode_pred_norm = self.diode_head(z_cond)
        # print("diode_norm=", diode_pred_norm)

        # Denormalize
        diode_raw = diode_pred_norm * self.diode_std + self.diode_mean
        # print("diode_raw=", diode_raw)

        # -----------------------------
        # Bounded physical parameters
        # -----------------------------
        # Iph: allow positive up to some reasonable max (e.g. 2 A)
        # Iph = 2.0 * torch.sigmoid(diode_raw[:, 0])
        Iph = diode_raw[:, 0]
        # # Iph = torch.relu(diode_raw[:, 0])
        #
        # # Io: tiny positive range (log-scale)
        # # Io = torch.exp(torch.clamp(diode_raw[:, 1], min=-30.0, max=-5.0))
        logIo = diode_raw[:, 1]
        Io = diode_raw[:, 1]#torch.exp(logIo)
        # Iph = 3.0 * torch.sigmoid(diode_raw[:, 0])
        #
        # logIo = -30 + 25 * torch.sigmoid(diode_raw[:, 1])
        # Io = torch.exp(logIo)

        # n: ideality factor in [0.8, 3.0]
        n = 0.8 + 2.2 * torch.sigmoid(diode_raw[:, 2])

        # Rs: series resistance in [0, 20]
        Rs = 20.0 * torch.sigmoid(diode_raw[:, 3])

        # Rsh: shunt resistance in [10, 1e5]
        # Rsh = 10.0 + (1e5 - 10.0) * torch.sigmoid(diode_raw[:, 4])
        # Rsh = 100.0 + (1e5 - 100.0) * torch.sigmoid(diode_raw[:, 4])
        Rsh = diode_raw[:, 4]

        batch_size = jv.shape[0]
        V = self.voltage_grid.unsqueeze(0).expand(batch_size, -1)

        # -----------------------------
        # Physics prediction
        # -----------------------------
        I_phys = self.solver(
            V,
            Iph.unsqueeze(1),
            Io.unsqueeze(1),
            n.unsqueeze(1),
            Rs.unsqueeze(1),
            Rsh.unsqueeze(1),
        )

        # -----------------------------
        # Neural residual
        # -----------------------------
        delta_I = self.residual_decoder(z_cond)

        I_pred = I_phys + delta_I

        return I_pred, I_phys, delta_I, z, diode_pred_norm
