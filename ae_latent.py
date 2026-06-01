import numpy as np
import torch
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from pathlib import Path


def compute_latent_vectors(model, jv_in, weather):
    Z = []
    with torch.no_grad():
        for i in range(len(jv_in)):
            _, z = model(
                torch.tensor(jv_in[i]).unsqueeze(0),
                torch.tensor(weather[i]).unsqueeze(0)
            )
            Z.append(z.squeeze().numpy())
    return np.array(Z)


def plot_latent_umap(Z, out_dir):
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1)
    Z2 = reducer.fit_transform(Z)

    plt.figure(figsize=(7,6))
    plt.scatter(Z2[:,0], Z2[:,1], s=8, alpha=0.7)
    plt.title("Latent Space (UMAP)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_umap.png", dpi=150)
    plt.close()


def plot_latent_tsne(Z, out_dir):
    Z2 = TSNE(n_components=2, perplexity=30).fit_transform(Z)

    plt.figure(figsize=(7,6))
    plt.scatter(Z2[:,0], Z2[:,1], s=8, alpha=0.7)
    plt.title("Latent Space (t‑SNE)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_tsne.png", dpi=150)
    plt.close()
