"""
noise_sensitivity_artemis.py
============================
Analisi della sensibilità al rumore nelle stime pseudo-ITE di ARTEMIS.

Domanda di ricerca: come si comporta ARTEMIS quando le stime pseudo-ITE
usate per la costruzione dei pair sono corrotte da rumore?

Questo script testa ARTEMIS iniettando rumore gaussiano sulle stime pseudo-ITE
a ogni step di aggiornamento del pair-sampler.
Il livello di rumore è parametrizzato come multiplo della std degli ITE stimati:
    tau_noisy = tau_clean + noise_factor * std(tau_clean) * N(0, I)

Livelli di rumore testati:
    0.0  → nessun rumore (baseline ARTEMIS)
    0.25 → rumore leggero
    0.5  → rumore moderato
    1.0  → rumore = std(ITE)  (segnale ≈ rumore)
    2.0  → rumore doppio della std
    4.0  → rumore molto elevato (pair quasi casuali)

Output:
    noise_sensitivity_outputs/
        noise_sensitivity_per_sim.csv
        noise_sensitivity_aggregate.csv
        noise_sensitivity_plot.pdf
        noise_sensitivity_plot.png

Dipendenze:
    pip install torch numpy pandas matplotlib requests scipy

Uso:
    python noise_sensitivity_artemis.py
    python noise_sensitivity_artemis.py --n_sims 50 --seed 0
"""

import os
import copy
import zipfile
import logging
import argparse
import requests
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("NoiseAnalysis")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER.info(f"Device: {DEVICE}")

# Livelli di rumore da testare (multipli della std degli ITE stimati)
NOISE_FACTORS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

OUT_DIR = "noise_sensitivity_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Configurazione ARTEMIS (dai parametri ottimali IHDP)
BEST_PARAMS = {
    'lr': 0.002706384665268061,
    'batch_size': 128,
    'latent_dim': 128,
    'alpha': 0.5323677007263332,
    'perc': 22,
    'ite_update_freq': 3,
    'lr_treat_clf': 0.0002874148000031135,
    'treat_clf_steps': 5,
    'outcome_clip_factor': 4.0,
    'use_output_clip': False,
    'margin': 0.7415498862697159,
    'epochs': 400,
    'patience': 40,
    'lambda_mi_pos': 0.07187797264363245,
    'mi_start_epoch': 30,
    'mi_pos_min_count': 12,
    'warmup_epochs': 20,
    'clip_norm': 2.0,
    'main_weight_decay': 0.0030582385526029317,
    'clf_weight_decay': 5.21853520111354e-05,
    'huber_beta': 0.5,
    'encoder_dropout': 0.2,
    'head_dropout': 0.05,
    'clf_dropout': 0.15,
    'use_contrastive': True,
    'use_local_mi': True,
    'use_dynamic_update': True,
    'pair_mode': 'dynamic_ite',
    'feature_k': 20,
    # Chiave specifica per questo esperimento
    'noise_factor': 0.0,
}


# ==============================================================================
# DATA LOADING
# ==============================================================================
def download_url(url: str, save_path: str, chunk_size: int = 128):
    LOGGER.info(f"Scaricando {url} -> {save_path}")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def load_ihdp(data_dir: str = "data") -> Tuple:
    """Scarica e carica il dataset IHDP (1000 simulazioni)."""
    os.makedirs(data_dir, exist_ok=True)
    base_url = "http://www.fredjo.com/files/"

    def _get(name):
        path = os.path.join(data_dir, name)
        zip_path = path + ".zip"
        if not os.path.exists(path):
            if not os.path.exists(zip_path):
                download_url(base_url + name + ".zip", zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
        return np.load(path)

    train = _get("ihdp_npci_1-1000.train.npz")
    test = _get("ihdp_npci_1-1000.test.npz")

    return (
        train['x'], train['t'], train['yf'], train['mu0'], train['mu1'],
        test['x'], test['t'], test['yf'], test['mu0'], test['mu1'],
    )


# ==============================================================================
# UTILITIES
# ==============================================================================
def get_continuous_indices(X: np.ndarray) -> List[int]:
    return [c for c in range(X.shape[1]) if len(np.unique(X[:, c])) > 2]


def sqrt_pehe(mu0: np.ndarray, mu1: np.ndarray, tau_hat: np.ndarray) -> float:
    tau_true = mu1 - mu0
    return float(np.sqrt(np.mean((tau_true - tau_hat) ** 2)))


def eps_ate(mu0: np.ndarray, mu1: np.ndarray, tau_hat: np.ndarray) -> float:
    return float(np.abs(np.mean(mu1 - mu0) - np.mean(tau_hat)))


def infer_num_treatments(t: np.ndarray) -> int:
    t_flat = np.round(t.reshape(-1)).astype(int)
    return int(len(np.unique(t_flat)))


# ==============================================================================
# MODELLI
# ==============================================================================
class CATEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, latent_dim)),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OutcomeHead(nn.Module):
    def __init__(self, latent_dim: int, num_treatments: int = 2,
                 use_output_clip: bool = True, clip_val: float = 5.0,
                 dropout: float = 0.1):
        super().__init__()
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h(z) for h in self.heads], dim=1)


class TreatmentClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_treatments: int = 2, dropout: float = 0.1):
        super().__init__()
        out_dim = 1 if num_treatments == 2 else num_treatments
        self.num_treatments = num_treatments
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 64)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(64, out_dim)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ==============================================================================
# PAIRING & DATASET
# ==============================================================================
def compute_tau_threshold(mu0_hat: np.ndarray, mu1_hat: np.ndarray,
                          perc: float = 20, sample: int = 100_000,
                          rng: np.random.Generator = None) -> float:
    tau = (mu1_hat - mu0_hat).reshape(-1)
    N = tau.size
    if N < 2:
        return 0.1
    if rng is None:
        rng = np.random.default_rng()
    m = min(sample, N)
    idx1 = rng.integers(0, N, size=m)
    idx2 = rng.integers(0, N, size=m)
    diffs = np.abs(tau[idx1] - tau[idx2])
    thr = float(np.percentile(diffs, perc))
    tau_std = max(float(np.std(tau)), 1e-6)
    thr = float(np.clip(thr if np.isfinite(thr) else 0.2 * tau_std,
                        0.05 * tau_std, tau_std))
    return thr


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    rng = np.random.default_rng(seed)
    tau = (mu1_hat - mu0_hat).reshape(-1)
    N = tau.shape[0]
    if N < 2:
        return _empty_pair_batch(X, T, Y)

    n_pairs = int(min(max(1, n_pairs), N))
    half = n_pairs // 2
    used = set()
    sim_pairs, dis_pairs = [], []
    max_attempts = max(50, n_pairs * 10)

    def add_pair(i, j, label, lst):
        if i == j:
            return
        key = (min(i, j), max(i, j))
        if key in used:
            return
        used.add(key)
        lst.append((i, j, label))

    attempts = 0
    while len(sim_pairs) < half and attempts < max_attempts:
        i = int(rng.integers(0, N))
        cand = np.where(np.abs(tau - tau[i]) < thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            add_pair(i, int(rng.choice(cand)), 1, sim_pairs)
        attempts += 1

    attempts = 0
    while len(dis_pairs) < (n_pairs - len(sim_pairs)) and attempts < max_attempts:
        i = int(rng.integers(0, N))
        cand = np.where(np.abs(tau - tau[i]) >= thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            add_pair(i, int(rng.choice(cand)), 0, dis_pairs)
        attempts += 1

    pairs = sim_pairs + dis_pairs
    if not pairs:
        return _empty_pair_batch(X, T, Y)

    rng.shuffle(pairs)
    idx_a, idx_b, labels = zip(*pairs)
    idx_a, idx_b = np.array(idx_a), np.array(idx_b)
    labels = np.array(labels, dtype=np.int64)
    return X[idx_a], Y[idx_a], T[idx_a], X[idx_b], Y[idx_b], T[idx_b], labels


def _empty_pair_batch(X, T, Y):
    empty = (0,) + X.shape[1:]
    return (np.zeros(empty, X.dtype), np.zeros((0,) + Y.shape[1:], Y.dtype),
            np.zeros((0,) + T.shape[1:], T.dtype), np.zeros(empty, X.dtype),
            np.zeros((0,) + Y.shape[1:], Y.dtype), np.zeros((0,) + T.shape[1:], T.dtype),
            np.array([], dtype=np.int64))


class NoisyDynamicDataset(Dataset):
    """
    Dataset che supporta l'iniezione controllata di rumore sulle stime pseudo-ITE.

    noise_factor: std del rumore = noise_factor * std(tau_estimated)
                  0.0 → nessun rumore (ARTEMIS standard)
    """

    def __init__(self, X, T, Y, mu0_hat, mu1_hat, bs=128, perc=20,
                 seed=0, noise_factor=0.0):
        self.X, self.T, self.Y = X, T, Y
        self.bs = int(bs)
        self.perc = float(perc)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.epoch = 0
        self.noise_factor = float(noise_factor)

        mu0 = mu0_hat if mu0_hat is not None else np.zeros(X.shape[0], np.float32)
        mu1 = mu1_hat if mu1_hat is not None else np.zeros(X.shape[0], np.float32)
        self._set_estimates(mu0, mu1)

    def _inject_noise(self, mu0: np.ndarray, mu1: np.ndarray):
        """Aggiunge rumore gaussiano alle stime pseudo-ITE."""
        if self.noise_factor == 0.0:
            return mu0, mu1
        tau = mu1 - mu0
        tau_std = max(float(np.std(tau)), 1e-6)
        noise = self.noise_factor * tau_std * self.rng.standard_normal(tau.shape)
        # Modifica mu1 in modo che tau_noisy = tau + noise
        mu1_noisy = mu0 + tau + noise
        return mu0, mu1_noisy.astype(np.float32)

    def _set_estimates(self, mu0: np.ndarray, mu1: np.ndarray):
        mu0_n, mu1_n = self._inject_noise(mu0, mu1)
        self.mu0_hat = mu0_n
        self.mu1_hat = mu1_n
        self.thr = compute_tau_threshold(mu0_n, mu1_n, perc=self.perc,
                                         rng=self.rng)

    def update_ite_estimates(self, mu0: np.ndarray, mu1: np.ndarray):
        self._set_estimates(mu0, mu1)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.bs))

    def __getitem__(self, idx: int):
        seed = (self.seed + 1000003 * self.epoch + 9176 * idx) & 0xFFFFFFFF
        x1, y1, t1, x2, y2, t2, lab = make_pairs_from_hat(
            self.X, self.T, self.Y, self.mu0_hat, self.mu1_hat,
            self.thr, self.bs, seed=seed
        )
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.float32),
            torch.tensor(t1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y2, dtype=torch.float32),
            torch.tensor(t2, dtype=torch.float32),
            torch.tensor(lab, dtype=torch.long),
        )


# ==============================================================================
# UTILITY MI
# ==============================================================================
def empirical_entropy(t_idx: torch.Tensor, num_treatments: int,
                      eps: float = 1e-8) -> torch.Tensor:
    t_idx = t_idx.view(-1).long()
    counts = torch.bincount(t_idx, minlength=num_treatments).float()
    probs = counts / counts.sum().clamp_min(1.0)
    return -(probs * torch.log(probs + eps)).sum()


def treatment_log_prob(clf: nn.Module, z: torch.Tensor,
                       t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    logits = clf(z)
    if num_treatments == 2:
        t_f = t_idx.view(-1, 1).float()
        return (t_f * F.logsigmoid(logits) +
                (1.0 - t_f) * F.logsigmoid(-logits)).mean()
    return F.log_softmax(logits, dim=1).gather(1, t_idx.view(-1, 1).long()).mean()


def clf_loss(clf: nn.Module, z: torch.Tensor,
             t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    logits = clf(z)
    if num_treatments == 2:
        return F.binary_cross_entropy_with_logits(logits, t_idx.view(-1, 1).float())
    return F.cross_entropy(logits, t_idx.view(-1).long())


def variational_mi(clf: nn.Module, z: torch.Tensor,
                   t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    h_t = empirical_entropy(t_idx.detach(), num_treatments)
    return h_t + treatment_log_prob(clf, z, t_idx, num_treatments)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                     label: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    dist_sq = (z1 - z2).pow(2).sum(dim=1)
    loss_sim = label * dist_sq
    loss_dis = (1 - label) * torch.clamp(margin - (dist_sq + 1e-8).sqrt(), min=0.0).pow(2)
    return (loss_sim + loss_dis).mean() / 2


# ==============================================================================
# EARLY STOPPING
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 40):
        self.patience = patience
        self.best = np.inf
        self.counter = 0
        self.early_stop = False
        self._enc_state = None
        self._pred_state = None

    def __call__(self, val_pehe: float, encoder: nn.Module, predictor: nn.Module):
        if val_pehe < self.best - 1e-6:
            self.best = val_pehe
            self.counter = 0
            self._enc_state = copy.deepcopy(encoder.state_dict())
            self._pred_state = copy.deepcopy(predictor.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, encoder: nn.Module, predictor: nn.Module):
        if self._enc_state is not None:
            encoder.load_state_dict(self._enc_state)
            predictor.load_state_dict(self._pred_state)


# ==============================================================================
# TRAINING CON NOISE_FACTOR
# ==============================================================================
def train_with_noise(
        sim_idx: int,
        data_train: Tuple,
        data_test: Tuple,
        device: str,
        hyperparams: Dict[str, Any],
) -> Dict[str, float]:
    """
    Allena ARTEMIS iniettando rumore sulle stime pseudo-ITE usate per il pairing.

    Il parametro chiave è hyperparams['noise_factor']:
        0.0  → ARTEMIS standard (nessun rumore)
        k>0  → std(rumore) = k * std(tau_stimato)
    """
    torch.manual_seed(sim_idx)
    np.random.seed(sim_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sim_idx)

    # Hyperparametri
    LR = hyperparams.get('lr', 1e-3)
    BS = hyperparams.get('batch_size', 128)
    LATENT_DIM = hyperparams.get('latent_dim', 128)
    ALPHA = hyperparams.get('alpha', 0.5)
    PERC = hyperparams.get('perc', 22)
    UPD_FREQ = hyperparams.get('ite_update_freq', 3)
    LR_CLF = hyperparams.get('lr_treat_clf', 1e-3)
    CLF_STEPS = hyperparams.get('treat_clf_steps', 5)
    MARGIN = hyperparams.get('margin', 0.74)
    LAMBDA_MI = hyperparams.get('lambda_mi_pos', 0.07)
    MI_START = hyperparams.get('mi_start_epoch', 30)
    POS_MIN = hyperparams.get('mi_pos_min_count', 12)
    EPOCHS = hyperparams.get('epochs', 400)
    PATIENCE = hyperparams.get('patience', 40)
    WARMUP = hyperparams.get('warmup_epochs', 20)
    CLIP = hyperparams.get('clip_norm', 2.0)
    MAIN_WD = hyperparams.get('main_weight_decay', 3e-3)
    CLF_WD = hyperparams.get('clf_weight_decay', 5e-5)
    HUBER = hyperparams.get('huber_beta', 0.5)
    ENC_DROP = hyperparams.get('encoder_dropout', 0.2)
    HEAD_DROP = hyperparams.get('head_dropout', 0.05)
    CLF_DROP = hyperparams.get('clf_dropout', 0.15)
    NOISE_FACTOR = hyperparams.get('noise_factor', 0.0)
    USE_CONTRASTIVE = hyperparams.get('use_contrastive', True)
    USE_MI = hyperparams.get('use_local_mi', True)

    X_tr, T_tr, Y_tr, mu0_tr, mu1_tr = data_train
    X_te, T_te, Y_te, mu0_te, mu1_te = data_test
    num_t = infer_num_treatments(T_tr)

    # Split train/val
    rng = np.random.default_rng(sim_idx)
    perm = rng.permutation(X_tr.shape[0])
    n_val = int(0.2 * X_tr.shape[0])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Normalizzazione
    cont_idx = get_continuous_indices(X_tr)
    X_train = X_tr[train_idx].copy()
    X_val = X_tr[val_idx].copy()
    X_test = X_te.copy()
    if cont_idx:
        mu_x = X_train[:, cont_idx].mean(0, keepdims=True)
        sd_x = np.maximum(X_train[:, cont_idx].std(0, keepdims=True), 1e-6)
        X_train[:, cont_idx] = (X_train[:, cont_idx] - mu_x) / sd_x
        X_val[:, cont_idx] = (X_val[:, cont_idx] - mu_x) / sd_x
        X_test[:, cont_idx] = (X_test[:, cont_idx] - mu_x) / sd_x

    mu_y = float(Y_tr[train_idx].mean())
    sd_y = max(float(Y_tr[train_idx].std()), 1e-6)
    Y_norm = (Y_tr - mu_y) / sd_y

    ds = NoisyDynamicDataset(
        X_train, T_tr[train_idx], Y_norm[train_idx],
        None, None, bs=BS, perc=PERC, seed=sim_idx,
        noise_factor=NOISE_FACTOR
    )
    dl = DataLoader(ds, batch_size=None, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    gt_val = mu1_tr[val_idx] - mu0_tr[val_idx]

    input_dim = X_tr.shape[1]
    encoder = CATEEncoder(input_dim, LATENT_DIM, ENC_DROP).to(device)
    predictor = OutcomeHead(LATENT_DIM, num_t, use_output_clip=False,
                            dropout=HEAD_DROP).to(device)
    clf = TreatmentClassifier(LATENT_DIM, num_t, CLF_DROP).to(device)

    opt_main = optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR, weight_decay=MAIN_WD
    )
    opt_clf = optim.AdamW(clf.parameters(), lr=LR_CLF, weight_decay=CLF_WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS, eta_min=1e-6)
    stopper = EarlyStopping(PATIENCE)

    static_done = False
    last_epoch = 0

    for epoch in range(EPOCHS):
        ds.set_epoch(epoch)
        RAMP = 80
        lam_ctr = (0.0 if (epoch < WARMUP or not USE_CONTRASTIVE)
                   else min(ALPHA, (epoch - WARMUP) / RAMP * ALPHA))
        lam_mi = (0.0 if (epoch < MI_START or not USE_MI) else LAMBDA_MI)

        encoder.train();
        predictor.train();
        clf.train()

        for batch in dl:
            x1, y1, t1, x2, y2, t2, lab = [b.to(device) for b in batch]

            # Fallback supervised se nessun pair
            if x1.shape[0] == 0:
                bidx = np.random.choice(len(train_idx), size=min(BS, len(train_idx)), replace=False)
                xb = torch.tensor(X_train[bidx], dtype=torch.float32, device=device)
                tb = torch.tensor(T_tr[train_idx][bidx], device=device).view(-1).long()
                yb = torch.tensor(Y_norm[train_idx][bidx], dtype=torch.float32, device=device).view(-1, 1)
                opt_main.zero_grad()
                zb = encoder(xb)
                mu_b = predictor(zb)
                loss = F.smooth_l1_loss(mu_b.gather(1, tb.unsqueeze(1)), yb, beta=HUBER)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), CLIP)
                opt_main.step()
                continue

            lab = lab.float()
            t1_idx = t1.view(-1).long()
            t2_idx = t2.view(-1).long()
            pos_mask = (lab == 1)

            z1 = encoder(x1)
            z2 = encoder(x2)

            # Update classificatore (detached)
            if lam_mi > 0 and pos_mask.sum().item() > POS_MIN:
                z_pos_det = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0).detach()
                t_pos_det = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
                for _ in range(CLF_STEPS):
                    opt_clf.zero_grad()
                    clf_loss(clf, z_pos_det, t_pos_det, num_t).backward()
                    torch.nn.utils.clip_grad_norm_(clf.parameters(), CLIP)
                    opt_clf.step()

            for p in clf.parameters():
                p.requires_grad = False

            opt_main.zero_grad()
            mu1_pred = predictor(z1)
            mu2_pred = predictor(z2)
            loss_sup = 0.5 * (
                    F.smooth_l1_loss(mu1_pred.gather(1, t1_idx.unsqueeze(1)), y1.view(-1, 1), beta=HUBER) +
                    F.smooth_l1_loss(mu2_pred.gather(1, t2_idx.unsqueeze(1)), y2.view(-1, 1), beta=HUBER)
            )
            loss_ctr = contrastive_loss(z1, z2, lab, MARGIN) if lam_ctr > 0 else torch.tensor(0., device=device)
            loss_mi = torch.tensor(0., device=device)
            if lam_mi > 0 and pos_mask.sum().item() > POS_MIN:
                z_pos = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)
                t_pos = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
                loss_mi = torch.clamp(variational_mi(clf, z_pos, t_pos, num_t), -5., 5.)

            total = loss_sup + lam_ctr * loss_ctr + lam_mi * loss_mi
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), CLIP)
            opt_main.step()

            for p in clf.parameters():
                p.requires_grad = True

        scheduler.step()

        encoder.eval();
        predictor.eval()
        with torch.no_grad():
            z_val = encoder(X_val_t)
            mu_val = predictor(z_val)
            tau_pred_val = (mu_val[:, 1] - mu_val[:, 0]).cpu().numpy() * sd_y
        val_pehe = np.sqrt(np.mean((gt_val - tau_pred_val) ** 2))
        if not np.isfinite(val_pehe):
            val_pehe = 999.0
        stopper(val_pehe, encoder, predictor)
        last_epoch = epoch + 1
        if stopper.early_stop:
            break

        # Update pseudo-ITE estimates con noise injection
        should_upd = (epoch >= WARMUP and UPD_FREQ > 0 and epoch % UPD_FREQ == 0)
        if should_upd:
            with torch.no_grad():
                X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
                mu_full = predictor(encoder(X_tr_t))
                new_mu0 = mu_full[:, 0].cpu().numpy()
                new_mu1 = mu_full[:, 1].cpu().numpy()
            # Il noise viene iniettato internamente da NoisyDynamicDataset
            ds.update_ite_estimates(new_mu0, new_mu1)

    stopper.restore(encoder, predictor)
    encoder.eval();
    predictor.eval()

    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu_te = predictor(encoder(X_te_t))
        tau_hat = (mu_te[:, 1] - mu_te[:, 0]).cpu().numpy() * sd_y

    return {
        'test_pehe': sqrt_pehe(mu0_te, mu1_te, tau_hat),
        'ate_err': eps_ate(mu0_te, mu1_te, tau_hat),
        'epochs': last_epoch,
    }


# ==============================================================================
# RUNNER ESPERIMENTO
# ==============================================================================
def run_noise_experiment(
        noise_factor: float,
        X_tr, T_tr, YF_tr, mu0_tr, mu1_tr,
        X_te, T_te, YF_te, mu0_te, mu1_te,
        n_sims: int,
) -> pd.DataFrame:
    params = copy.deepcopy(BEST_PARAMS)
    params['noise_factor'] = noise_factor
    label = f"noise={noise_factor:.2f}"
    LOGGER.info(f"\n{'=' * 70}\nRUNNING: {label}\n{'=' * 70}")

    rows = []
    total_sims = X_tr.shape[-1] if X_tr.ndim == 3 else 1
    n_sims = min(n_sims, total_sims)

    for i in range(n_sims):
        if X_tr.ndim == 3:
            train_data = (X_tr[:, :, i], T_tr[:, i], YF_tr[:, i], mu0_tr[:, i], mu1_tr[:, i])
            test_data = (X_te[:, :, i], T_te[:, i], YF_te[:, i], mu0_te[:, i], mu1_te[:, i])
        else:
            train_data = (X_tr, T_tr, YF_tr, mu0_tr, mu1_tr)
            test_data = (X_te, T_te, YF_te, mu0_te, mu1_te)

        res = train_with_noise(i, train_data, test_data, DEVICE, params)
        rows.append({
            'noise_factor': noise_factor,
            'sim_id': i,
            'test_pehe': res['test_pehe'],
            'ate_err': res['ate_err'],
            'epochs': res['epochs'],
        })
        LOGGER.info(
            f"  [{label}] Sim {i + 1}/{n_sims} | "
            f"PEHE={res['test_pehe']:.4f} | ATE={res['ate_err']:.4f} | Ep={res['epochs']}"
        )

    return pd.DataFrame(rows)


# ==============================================================================
# VISUALIZZAZIONE
# ==============================================================================
def plot_noise_sensitivity(df_agg: pd.DataFrame, out_dir: str):
    """
    Crea tre grafici:
    1. PEHE (mean ± std) vs noise_factor
    2. ATE error vs noise_factor
    3. Degradazione relativa rispetto al baseline (noise=0)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("ARTEMIS: Sensitivity to Pseudo-ITE Noise During Pairing",
                 fontsize=13, fontweight='bold', y=1.01)

    noise_vals = df_agg['noise_factor'].values
    pehe_mean = df_agg['mean_pehe'].values
    pehe_std = df_agg['std_pehe'].values
    ate_mean = df_agg['mean_ate'].values
    ate_std = df_agg['std_ate'].values

    x_labels = [str(v) for v in noise_vals]

    # ---- Plot 1: PEHE ----
    ax = axes[0]
    ax.errorbar(noise_vals, pehe_mean, yerr=pehe_std, fmt='o-',
                color='#2563eb', linewidth=2, markersize=7, capsize=5,
                label=r'$\sqrt{\epsilon_{\mathrm{PEHE}}}$')
    ax.fill_between(noise_vals,
                    np.maximum(pehe_mean - pehe_std, 0),
                    pehe_mean + pehe_std,
                    alpha=0.15, color='#2563eb')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='No noise')
    ax.set_xlabel("Noise Factor (× ITE std)", fontsize=11)
    ax.set_ylabel(r"$\sqrt{\epsilon_{\mathrm{PEHE}}}$", fontsize=11)
    ax.set_title("PEHE vs Noise Level", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(noise_vals)
    ax.set_xticklabels(x_labels, rotation=30)

    # ---- Plot 2: ATE ----
    ax = axes[1]
    ax.errorbar(noise_vals, ate_mean, yerr=ate_std, fmt='s-',
                color='#dc2626', linewidth=2, markersize=7, capsize=5,
                label=r'$\epsilon_{\mathrm{ATE}}$')
    ax.fill_between(noise_vals,
                    np.maximum(ate_mean - ate_std, 0),
                    ate_mean + ate_std,
                    alpha=0.15, color='#dc2626')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel("Noise Factor (× ITE std)", fontsize=11)
    ax.set_ylabel(r"$\epsilon_{\mathrm{ATE}}$", fontsize=11)
    ax.set_title("ATE Error vs Noise Level", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(noise_vals)
    ax.set_xticklabels(x_labels, rotation=30)

    # ---- Plot 3: Degradazione relativa ----
    ax = axes[2]
    baseline_pehe = pehe_mean[0] if pehe_mean[0] > 1e-8 else 1.0
    baseline_ate = ate_mean[0] if ate_mean[0] > 1e-8 else 1.0
    rel_pehe = (pehe_mean / baseline_pehe - 1.0) * 100
    rel_ate = (ate_mean / baseline_ate - 1.0) * 100
    ax.plot(noise_vals, rel_pehe, 'o-', color='#2563eb', linewidth=2,
            markersize=7, label='PEHE degradation')
    ax.plot(noise_vals, rel_ate, 's--', color='#dc2626', linewidth=2,
            markersize=7, label='ATE degradation')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel("Noise Factor (× ITE std)", fontsize=11)
    ax.set_ylabel("Relative Degradation (%)", fontsize=11)
    ax.set_title("Relative Degradation vs Baseline", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xticks(noise_vals)
    ax.set_xticklabels(x_labels, rotation=30)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f"noise_sensitivity_plot.{ext}")
        fig.savefig(path, bbox_inches='tight', dpi=150)
        LOGGER.info(f"Salvato: {path}")
    plt.close(fig)


def print_latex_table(df_agg: pd.DataFrame):
    """Stampa una tabella LaTeX pronta per il paper."""
    print("\n" + "=" * 70)
    print("TABELLA LATEX (noise sensitivity)")
    print("=" * 70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Sensitivity of \texttt{ARTEMIS} to pseudo-ITE noise during pairing.")
    print(r"Noise factor $\sigma$ denotes the standard deviation of injected noise")
    print(r"as a multiple of the estimated ITE standard deviation.")
    print(r"Results on IHDP over " + str(int(df_agg['n_sims'].iloc[0])) + r" simulations.}")
    print(r"\label{tab:noise_sensitivity}")
    print(r"\small")
    print(r"\begin{tabular}{ccc}")
    print(r"\toprule")
    print(r"Noise Factor $\sigma$ & $\sqrt{\epsilon_{\text{PEHE}}}$ & $\epsilon_{\text{ATE}}$ \\")
    print(r"\midrule")
    for _, row in df_agg.iterrows():
        nf = row['noise_factor']
        label = r"\textbf{0.00 (baseline)}" if nf == 0.0 else f"{nf:.2f}"
        p = f"{row['mean_pehe']:.4f} $\\pm$ {row['std_pehe']:.4f}"
        a = f"{row['mean_ate']:.4f} $\\pm$ {row['std_ate']:.4f}"
        print(f"{label} & {p} & {a} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ARTEMIS noise sensitivity analysis")
    parser.add_argument("--n_sims", type=int, default=100,
                        help="Simulazioni per livello di rumore (default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed globale (non usato nel training, solo per info)")
    parser.add_argument("--noise_factors", type=float, nargs='+',
                        default=NOISE_FACTORS,
                        help="Lista di noise factors da testare")
    args = parser.parse_args()

    LOGGER.info(f"Caricamento IHDP...")
    (X_tr, T_tr, YF_tr, mu0_tr, mu1_tr,
     X_te, T_te, YF_te, mu0_te, mu1_te) = load_ihdp()

    n_avail = X_tr.shape[-1] if X_tr.ndim == 3 else 1
    n_sims = min(args.n_sims, n_avail)
    LOGGER.info(f"Simulazioni per livello: {n_sims} / {n_avail}")
    LOGGER.info(f"Noise factors: {args.noise_factors}")

    all_dfs = []
    for nf in args.noise_factors:
        df = run_noise_experiment(
            noise_factor=nf,
            X_tr=X_tr, T_tr=T_tr, YF_tr=YF_tr, mu0_tr=mu0_tr, mu1_tr=mu1_tr,
            X_te=X_te, T_te=T_te, YF_te=YF_te, mu0_te=mu0_te, mu1_te=mu1_te,
            n_sims=n_sims,
        )
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Aggregazione
    agg_rows = []
    for nf, grp in df_all.groupby('noise_factor', sort=False):
        agg_rows.append({
            'noise_factor': nf,
            'n_sims': len(grp),
            'mean_pehe': grp['test_pehe'].mean(),
            'std_pehe': grp['test_pehe'].std(),
            'mean_ate': grp['ate_err'].mean(),
            'std_ate': grp['ate_err'].std(),
            'mean_epochs': grp['epochs'].mean(),
        })
    df_agg = pd.DataFrame(agg_rows)

    # Salvataggio CSV
    per_sim_path = os.path.join(OUT_DIR, "noise_sensitivity_per_sim.csv")
    agg_path = os.path.join(OUT_DIR, "noise_sensitivity_aggregate.csv")
    df_all.to_csv(per_sim_path, index=False, sep=';')
    df_agg.to_csv(agg_path, index=False, sep=';')
    LOGGER.info(f"Risultati per sim: {per_sim_path}")
    LOGGER.info(f"Risultati aggregati: {agg_path}")

    # Stampa summary
    print("\n" + "=" * 70)
    print("SUMMARY NOISE SENSITIVITY")
    print("=" * 70)
    print(df_agg.to_string(index=False))

    # Grafici
    plot_noise_sensitivity(df_agg, OUT_DIR)

    # Tabella LaTeX
    print_latex_table(df_agg)

    LOGGER.info("Completato.")


if __name__ == "__main__":
    main()