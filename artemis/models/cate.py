"""ARTEMIS networks for binary-treatment CATE estimation (IHDP, Jobs)."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class CATEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.15):
        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, latent_dim)), nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.network(x)


class OutcomeHead(nn.Module):
    def __init__(self, latent_dim, num_treatments=2, use_output_clip=True, clip_val=5.0, dropout=0.1):
        super().__init__()
        self.num_treatments = num_treatments
        self.heads = nn.ModuleList()
        for _ in range(num_treatments):
            layers = [
                spectral_norm(nn.Linear(latent_dim, 64)), nn.GELU(),
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(64, 1)),
            ]
            if use_output_clip:
                layers.append(nn.Hardtanh(min_val=-clip_val, max_val=clip_val))
            self.heads.append(nn.Sequential(*layers))

    def forward(self, z):
        outs = [head(z) for head in self.heads]
        return torch.cat(outs, dim=1)


class TreatmentClassifier(nn.Module):
    def __init__(self, latent_dim, num_treatments=2, dropout=0.1):
        super().__init__()
        out_dim = 1 if num_treatments == 2 else num_treatments
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 64)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(64, out_dim)),
        )

    def forward(self, z):
        return self.net(z)
