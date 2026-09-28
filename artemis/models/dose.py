"""ARTEMIS networks for multi-treatment, continuous-dose estimation (TCGA)."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, latent_dim)), nn.LayerNorm(latent_dim),
        )
    def forward(self, x):
        return self.net(x)

class DoseAwareNet(nn.Module):
    def __init__(self, input_dim, num_treatments, hidden_dim=256, latent_dim=128, dropout=0.15):
        super().__init__()
        self.num_treatments = num_treatments
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.t_embed = nn.Embedding(num_treatments, latent_dim)
        self.dose_net = nn.Sequential(
            spectral_norm(nn.Linear(1, latent_dim // 2)), nn.GELU(),
            spectral_norm(nn.Linear(latent_dim // 2, latent_dim // 2)), nn.GELU(),
        )
        fusion_dim = latent_dim + latent_dim + latent_dim // 2
        self.outcome_net = nn.Sequential(
            spectral_norm(nn.Linear(fusion_dim, hidden_dim)), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )
        self.dose_reg_head = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, latent_dim // 2)), nn.GELU(),
            spectral_norm(nn.Linear(latent_dim // 2, 1)), nn.Sigmoid(),
        )

    def forward(self, x, a, d):
        z = self.encoder(x)
        a_emb = self.t_embed(a)
        d_feat = self.dose_net(d.unsqueeze(1))
        y = self.outcome_net(torch.cat([z, a_emb, d_feat], dim=1))
        d_hat = self.dose_reg_head(z)
        return z, y, d_hat

    @torch.no_grad()
    def predict_all_treatments_at_eval_d(self, x: torch.Tensor, eval_d: torch.Tensor) -> torch.Tensor:
        self.eval()
        B = x.shape[0]
        preds = []
        for k in range(self.num_treatments):
            ak = torch.full((B,), k, dtype=torch.long, device=x.device)
            if k == 0:
                # Treatment 0 is the untreated control and therefore has no
                # active-treatment dose.  The legacy [N, K-1] representation
                # previously indexed column -1 here.
                dk = torch.zeros(B, dtype=eval_d.dtype, device=x.device)
            elif eval_d.shape[1] == self.num_treatments:
                dk = eval_d[:, k]
            else:
                dk = eval_d[:, k - 1]
            _, yk, _ = self.forward(x, ak, dk)
            preds.append(yk.squeeze(1))
        return torch.stack(preds, dim=1)

class TreatmentClassifier(nn.Module):
    def __init__(self, latent_dim, num_treatments, dropout=0.15):
        super().__init__()
        out_dim = 1 if num_treatments == 2 else num_treatments
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 64)), nn.GELU(), nn.Dropout(dropout),
            spectral_norm(nn.Linear(64, out_dim)),
        )
    def forward(self, z):
        return self.net(z)
