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

    def __init__(self, Ns=72, Vt=0.02585, n_iter=20):
        super().__init__()
        self.Ns = Ns
        self.Vt = Vt
        self.n_iter = n_iter

    def forward(self, voltage, Iph, Io, n, Rs, Rsh):

        # initial guess
        I = Iph.clone()

        for _ in range(self.n_iter):

            # exp_arg = (voltage + I * Rs) / (n * self.Vt)
            exp_arg = (
                              voltage + I * Rs
                      ) / (
                              n * self.Ns * self.Vt
                      )
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

    def __init__(
            self,
            jv_dim,
            weather_dim,
            voltage_grid,
            latent_dim=32,
            hidden_dims=(256, 128),
    ):
        super().__init__()

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        input_dim = jv_dim + weather_dim

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # --------------------------------------------------
        # Parameter predictor
        # --------------------------------------------------

        self.param_head = nn.Sequential(
            nn.Linear(latent_dim + weather_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

        # --------------------------------------------------
        # Voltage grid
        # --------------------------------------------------

        self.register_buffer(
            "voltage_grid",
            torch.tensor(
                voltage_grid,
                dtype=torch.float32
            )
        )

        self.solver = SingleDiodeSolver()

    def forward(self, jv, weather):

        x = torch.cat([jv, weather], dim=-1)

        # latent representation
        z = self.encoder(x)

        z_cond = torch.cat([z, weather], dim=-1)

        raw_params = self.param_head(z_cond)

        # -------------------------------------------
        # Physical constraints
        # -------------------------------------------

        # Iph = F.softplus(raw_params[:, 0]) + 1e-6
        # Io = F.softplus(raw_params[:, 1]) + 1e-12
        #
        # n = F.softplus(raw_params[:, 2]) + 0.5
        #
        # Rs = F.softplus(raw_params[:, 3])
        #
        # Rsh = F.softplus(raw_params[:, 4]) + 1.0
        # Photocurrent
        Iph = 2.0 * torch.sigmoid(raw_params[:, 0])

        # Saturation current
        log_Io = -20.0 + 14.0 * torch.sigmoid(raw_params[:, 1])
        Io = torch.exp(log_Io)

        # Ideality factor
        n = 1.0 + 1.5 * torch.sigmoid(raw_params[:, 2])

        # Series resistance
        Rs = 5.0 * torch.sigmoid(raw_params[:, 3])

        # Shunt resistance
        log_Rsh = (
                np.log(50.0)
                + (np.log(5000.0) - np.log(50.0))
                * torch.sigmoid(raw_params[:, 4])
        )
        Rsh = torch.exp(log_Rsh)

        batch_size = jv.shape[0]

        V = self.voltage_grid.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        I_pred = self.solver(
            V,
            Iph.unsqueeze(1),
            Io.unsqueeze(1),
            n.unsqueeze(1),
            Rs.unsqueeze(1),
            Rsh.unsqueeze(1),
        )


        params = {
            "Iph": Iph,
            "Io": Io,
            "n": n,
            "Rs": Rs,
            "Rsh": Rsh,
        }

        return I_pred, z, params