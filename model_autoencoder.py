# model_autoencoder.py

import torch
import torch.nn as nn


class JVAutoencoder(nn.Module):
    def __init__(self, jv_dim, weather_dim, latent_dim=32, hidden_dims=(256, 128)):
        super().__init__()
        input_dim = jv_dim + weather_dim

        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim + weather_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec += [nn.Linear(prev, jv_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, jv, weather):
        x = torch.cat([jv, weather], dim=-1)
        z = self.encoder(x)
        z_cond = torch.cat([z, weather], dim=-1)
        jv_recon = self.decoder(z_cond)
        return jv_recon, z
