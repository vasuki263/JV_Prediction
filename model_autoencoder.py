import torch
import torch.nn as nn


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
