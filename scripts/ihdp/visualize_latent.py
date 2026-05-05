"""
latent_viz_artemis.py
=====================
Quantitative visualization of the latent space: ARTEMIS vs CFRNet.

This script:
1. Trains ARTEMIS and CFRNet on simulation 0 of IHDP
2. Extracts latent representations on the test set
3. Applies t-SNE and UMAP for 2D dimensionality reduction
4. Generates publication-quality figures with 4 panels:
   - ARTEMIS embeddings colored by true ITE
   - ARTEMIS embeddings colored by treatment group
   - CFRNet embeddings colored by true ITE
   - CFRNet embeddings colored by treatment group
5. Computes quantitative metrics on the latent space:
   - Linear MMD between treated and control
   - Spearman correlation between latent distances and |ITE_i - ITE_j|
   - Mean distance to the nearest neighbor of opposite treatment (NOTD)
   - AUC of the linear treatment classifier (treatment probe)

Output:
    latent_viz_outputs/
        latent_tsne_umap.pdf / .png     (main figure 2×4)
        latent_metrics.csv              (quantitative metrics)
        latent_metrics_latex.txt        (ready LaTeX table)

Dependencies:
    pip install torch numpy pandas matplotlib scikit-learn umap-learn requests

Usage:
    python latent_viz_artemis.py
    python latent_viz_artemis.py --sim_id 0 --n_epochs 400 --umap      (with UMAP)
    python latent_viz_artemis.py --no_umap                              (t-SNE only)
"""

import os
import copy
import zipfile
import logging
import argparse
import requests
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils import spectral_norm

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("LatentViz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER.info(f"Device: {DEVICE}")

OUT_DIR = "latent_viz_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Parametri ARTEMIS ottimali su IHDP
ARTEMIS_PARAMS = {
    'lr': 0.002706384665268061, 'batch_size': 128, 'latent_dim': 128,
    'alpha': 0.5323677007263332, 'perc': 22, 'ite_update_freq': 3,
    'lr_treat_clf': 0.0002874148000031135, 'treat_clf_steps': 5,
    'outcome_clip_factor': 4.0, 'use_output_clip': False,
    'margin': 0.7415498862697159, 'epochs': 400, 'patience': 40,
    'lambda_mi_pos': 0.07187797264363245, 'mi_start_epoch': 30,
    'mi_pos_min_count': 12, 'warmup_epochs': 20, 'clip_norm': 2.0,
    'main_weight_decay': 0.0030582385526029317,
    'clf_weight_decay': 5.21853520111354e-05,
    'huber_beta': 0.5, 'encoder_dropout': 0.2, 'head_dropout': 0.05,
    'clf_dropout': 0.15, 'use_contrastive': True, 'use_local_mi': True,
    'use_dynamic_update': True, 'pair_mode': 'dynamic_ite', 'feature_k': 20,
}

# Parametri CFRNet (stesso encoder, stesse head, + MMD penalty)
CFRNET_PARAMS = {
    'lr': 0.002706384665268061, 'batch_size': 128, 'latent_dim': 128,
    'mmd_lambda': 1.0, 'mmd_sigma': 1.0,
    'epochs': 400, 'patience': 40, 'clip_norm': 2.0,
    'main_weight_decay': 0.0030582385526029317,
    'huber_beta': 0.5, 'encoder_dropout': 0.2, 'head_dropout': 0.05,
    'use_output_clip': False,
}


# ==============================================================================
# DATA LOADING
# ==============================================================================
def download_url(url: str, save_path: str, chunk_size: int = 128):
    LOGGER.info(f"Downloading {url} -> {save_path}")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def load_ihdp(data_dir: str = "data") -> Tuple:
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
    test  = _get("ihdp_npci_1-1000.test.npz")

    return (
        train['x'], train['t'], train['yf'], train['mu0'], train['mu1'],
        test['x'],  test['t'],  test['yf'],  test['mu0'], test['mu1'],
    )


def get_sim(X, T, Y, mu0, mu1, sim_id: int):
    if X.ndim == 3:
        return X[:, :, sim_id], T[:, sim_id], Y[:, sim_id], mu0[:, sim_id], mu1[:, sim_id]
    return X, T, Y, mu0, mu1


# ==============================================================================
# PREPROCESSING
# ==============================================================================
def get_continuous_indices(X: np.ndarray):
    return [c for c in range(X.shape[1]) if len(np.unique(X[:, c])) > 2]


def preprocess(X_tr, Y_tr, T_tr, X_te, train_idx, val_idx):
    cont_idx = get_continuous_indices(X_tr)
    X_train = X_tr[train_idx].copy()
    X_val   = X_tr[val_idx].copy()
    X_test  = X_te.copy()
    if cont_idx:
        mu_x = X_train[:, cont_idx].mean(0, keepdims=True)
        sd_x = np.maximum(X_train[:, cont_idx].std(0, keepdims=True), 1e-6)
        X_train[:, cont_idx] = (X_train[:, cont_idx] - mu_x) / sd_x
        X_val[:,   cont_idx] = (X_val[:,   cont_idx] - mu_x) / sd_x
        X_test[:,  cont_idx] = (X_test[:,  cont_idx] - mu_x) / sd_x
    mu_y = float(Y_tr[train_idx].mean())
    sd_y = max(float(Y_tr[train_idx].std()), 1e-6)
    Y_norm = (Y_tr - mu_y) / sd_y
    return X_train, X_val, X_test, Y_norm, mu_y, sd_y


# ==============================================================================
# MODELLI COMUNI
# ==============================================================================
class CATEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, latent_dim)),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.network(x)


class OutcomeHead(nn.Module):
    def __init__(self, latent_dim: int, num_treatments: int = 2,
                 use_output_clip: bool = False, clip_val: float = 5.0,
                 dropout: float = 0.05):
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_treatments):
            layers = [
                spectral_norm(nn.Linear(latent_dim, 64)), nn.GELU(),
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(64, 1)),
            ]
            if use_output_clip:
                layers.append(nn.Hardtanh(-clip_val, clip_val))
            self.heads.append(nn.Sequential(*layers))

    def forward(self, z):
        return torch.cat([h(z) for h in self.heads], dim=1)


class TreatmentClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_treatments: int = 2, dropout: float = 0.15):
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

    def forward(self, z):
        return self.net(z)


# ==============================================================================
# PAIRING DATASET (per ARTEMIS)
# ==============================================================================
def compute_tau_threshold(mu0_hat, mu1_hat, perc=22, sample=100_000, rng=None):
    tau = (mu1_hat - mu0_hat).reshape(-1)
    N = tau.size
    if N < 2:
        return 0.1
    if rng is None:
        rng = np.random.default_rng()
    m = min(sample, N)
    i1 = rng.integers(0, N, size=m)
    i2 = rng.integers(0, N, size=m)
    diffs = np.abs(tau[i1] - tau[i2])
    thr = float(np.percentile(diffs, perc))
    sd = max(float(np.std(tau)), 1e-6)
    return float(np.clip(thr if np.isfinite(thr) else 0.2 * sd, 0.05 * sd, sd))


def _empty(X, T, Y):
    e = (0,) + X.shape[1:]
    return (np.zeros(e, X.dtype), np.zeros((0,) + Y.shape[1:], Y.dtype),
            np.zeros((0,) + T.shape[1:], T.dtype), np.zeros(e, X.dtype),
            np.zeros((0,) + Y.shape[1:], Y.dtype), np.zeros((0,) + T.shape[1:], T.dtype),
            np.array([], dtype=np.int64))


def make_pairs(X, T, Y, mu0, mu1, thr, n_pairs, seed=None):
    rng = np.random.default_rng(seed)
    tau = (mu1 - mu0).reshape(-1)
    N = tau.shape[0]
    if N < 2:
        return _empty(X, T, Y)
    n_pairs = int(min(max(1, n_pairs), N))
    half = n_pairs // 2
    used, sim_p, dis_p = set(), [], []
    mx = max(50, n_pairs * 10)

    def add(i, j, lbl, lst):
        if i == j: return
        k = (min(i, j), max(i, j))
        if k in used: return
        used.add(k); lst.append((i, j, lbl))

    att = 0
    while len(sim_p) < half and att < mx:
        i = int(rng.integers(0, N))
        cand = np.where(np.abs(tau - tau[i]) < thr)[0]
        cand = cand[cand != i]
        if cand.size:
            add(i, int(rng.choice(cand)), 1, sim_p)
        att += 1
    att = 0
    while len(dis_p) < n_pairs - len(sim_p) and att < mx:
        i = int(rng.integers(0, N))
        cand = np.where(np.abs(tau - tau[i]) >= thr)[0]
        cand = cand[cand != i]
        if cand.size:
            add(i, int(rng.choice(cand)), 0, dis_p)
        att += 1

    pairs = sim_p + dis_p
    if not pairs:
        return _empty(X, T, Y)
    rng.shuffle(pairs)
    ia, ib, lb = zip(*pairs)
    ia, ib = np.array(ia), np.array(ib)
    lb = np.array(lb, dtype=np.int64)
    return X[ia], Y[ia], T[ia], X[ib], Y[ib], T[ib], lb


class ARTEMISDataset(Dataset):
    def __init__(self, X, T, Y, bs=128, perc=22, seed=0):
        self.X, self.T, self.Y, self.bs, self.perc = X, T, Y, bs, perc
        self.seed = seed; self.epoch = 0; self.rng = np.random.default_rng(seed)
        self.mu0 = np.zeros(X.shape[0], np.float32)
        self.mu1 = np.zeros(X.shape[0], np.float32)
        self._update_thr()

    def _update_thr(self):
        self.thr = compute_tau_threshold(self.mu0, self.mu1, self.perc, rng=self.rng)

    def update(self, mu0, mu1):
        self.mu0, self.mu1 = mu0, mu1
        self._update_thr()

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.bs))

    def __getitem__(self, idx):
        seed = (self.seed + 1000003 * self.epoch + 9176 * idx) & 0xFFFFFFFF
        x1, y1, t1, x2, y2, t2, lb = make_pairs(
            self.X, self.T, self.Y, self.mu0, self.mu1, self.thr, self.bs, seed)
        return (torch.tensor(x1, dtype=torch.float32),
                torch.tensor(y1, dtype=torch.float32),
                torch.tensor(t1, dtype=torch.float32),
                torch.tensor(x2, dtype=torch.float32),
                torch.tensor(y2, dtype=torch.float32),
                torch.tensor(t2, dtype=torch.float32),
                torch.tensor(lb, dtype=torch.long))


# ==============================================================================
# UTILS MI E CONTRASTIVE
# ==============================================================================
def emp_entropy(t_idx, num_t, eps=1e-8):
    t_idx = t_idx.view(-1).long()
    cnt = torch.bincount(t_idx, minlength=num_t).float()
    p = cnt / cnt.sum().clamp_min(1.)
    return -(p * torch.log(p + eps)).sum()


def log_prob_mean(clf, z, t_idx, num_t):
    logits = clf(z)
    if num_t == 2:
        tf = t_idx.view(-1, 1).float()
        return (tf * F.logsigmoid(logits) + (1 - tf) * F.logsigmoid(-logits)).mean()
    return F.log_softmax(logits, dim=1).gather(1, t_idx.view(-1, 1).long()).mean()


def clf_bce(clf, z, t_idx, num_t):
    logits = clf(z)
    if num_t == 2:
        return F.binary_cross_entropy_with_logits(logits, t_idx.view(-1, 1).float())
    return F.cross_entropy(logits, t_idx.view(-1).long())


def var_mi(clf, z, t_idx, num_t):
    return emp_entropy(t_idx.detach(), num_t) + log_prob_mean(clf, z, t_idx, num_t)


def ctr_loss(z1, z2, label, margin=1.0):
    d2 = (z1 - z2).pow(2).sum(1)
    return (label * d2 + (1 - label) * torch.clamp(margin - (d2 + 1e-8).sqrt(), min=0).pow(2)).mean() / 2


# ==============================================================================
# EARLY STOPPING
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=40):
        self.patience = patience
        self.best = np.inf; self.counter = 0; self.early_stop = False
        self._enc = None; self._pred = None

    def __call__(self, val_pehe, enc, pred):
        if val_pehe < self.best - 1e-6:
            self.best = val_pehe; self.counter = 0
            self._enc = copy.deepcopy(enc.state_dict())
            self._pred = copy.deepcopy(pred.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, enc, pred):
        if self._enc: enc.load_state_dict(self._enc); pred.load_state_dict(self._pred)


# ==============================================================================
# TRAINING ARTEMIS
# ==============================================================================
def train_artemis(X_train, X_val, X_test, Y_norm, T_tr, mu0_tr, mu1_tr,
                  mu0_te, mu1_te, train_idx, val_idx, sd_y, device, params):
    LOGGER.info("Training ARTEMIS...")
    torch.manual_seed(0); np.random.seed(0)

    LR = params['lr']; BS = params['batch_size']; LD = params['latent_dim']
    ALPHA = params['alpha']; PERC = params['perc']
    UPD = params['ite_update_freq']; LR_CLF = params['lr_treat_clf']
    CLF_S = params['treat_clf_steps']; MAR = params['margin']
    LMI = params['lambda_mi_pos']; MIS = params['mi_start_epoch']
    PMN = params['mi_pos_min_count']; EPOCHS = params['epochs']
    PAT = params['patience']; WARM = params['warmup_epochs']
    CLIP = params['clip_norm']; WD = params['main_weight_decay']
    CWD = params['clf_weight_decay']; HUB = params['huber_beta']
    ENC_D = params['encoder_dropout']; HEAD_D = params['head_dropout']
    CLF_D = params['clf_dropout']

    ds = ARTEMISDataset(X_train, T_tr[train_idx], Y_norm[train_idx], BS, PERC, 0)
    dl = DataLoader(ds, batch_size=None, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    gt_val = mu1_tr[val_idx] - mu0_tr[val_idx]

    enc  = CATEEncoder(X_train.shape[1], LD, ENC_D).to(device)
    pred = OutcomeHead(LD, 2, False, 5.0, HEAD_D).to(device)
    clf  = TreatmentClassifier(LD, 2, CLF_D).to(device)

    opt_main = optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR, weight_decay=WD)
    opt_clf  = optim.AdamW(clf.parameters(), lr=LR_CLF, weight_decay=CWD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS, eta_min=1e-6)
    stopper = EarlyStopping(PAT)

    for epoch in range(EPOCHS):
        ds.set_epoch(epoch)
        RAMP = 80
        lc = (0. if epoch < WARM else min(ALPHA, (epoch - WARM) / RAMP * ALPHA))
        lm = (0. if epoch < MIS else LMI)
        enc.train(); pred.train(); clf.train()

        for batch in dl:
            x1, y1, t1, x2, y2, t2, lab = [b.to(device) for b in batch]
            if x1.shape[0] == 0:
                continue
            lab = lab.float()
            t1i = t1.view(-1).long(); t2i = t2.view(-1).long()
            pos = (lab == 1)
            z1 = enc(x1); z2 = enc(x2)

            if lm > 0 and pos.sum().item() > PMN:
                zp = torch.cat([z1[pos], z2[pos]], 0).detach()
                tp = torch.cat([t1i[pos], t2i[pos]], 0)
                for _ in range(CLF_S):
                    opt_clf.zero_grad()
                    clf_bce(clf, zp, tp, 2).backward()
                    torch.nn.utils.clip_grad_norm_(clf.parameters(), CLIP)
                    opt_clf.step()

            for p in clf.parameters(): p.requires_grad = False
            opt_main.zero_grad()
            m1 = pred(z1); m2 = pred(z2)
            ls = 0.5 * (
                F.smooth_l1_loss(m1.gather(1, t1i.unsqueeze(1)), y1.view(-1, 1), beta=HUB) +
                F.smooth_l1_loss(m2.gather(1, t2i.unsqueeze(1)), y2.view(-1, 1), beta=HUB)
            )
            lc_val = ctr_loss(z1, z2, lab, MAR) if lc > 0 else torch.tensor(0., device=device)
            lm_val = torch.tensor(0., device=device)
            if lm > 0 and pos.sum().item() > PMN:
                zp2 = torch.cat([z1[pos], z2[pos]], 0)
                tp2 = torch.cat([t1i[pos], t2i[pos]], 0)
                lm_val = torch.clamp(var_mi(clf, zp2, tp2, 2), -5., 5.)
            (ls + lc * lc_val + lm * lm_val).backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), CLIP)
            opt_main.step()
            for p in clf.parameters(): p.requires_grad = True

        sched.step()
        enc.eval(); pred.eval()
        with torch.no_grad():
            zv = enc(X_val_t)
            mv = pred(zv)
            tv = (mv[:, 1] - mv[:, 0]).cpu().numpy() * sd_y
        vp = float(np.sqrt(np.mean((gt_val - tv) ** 2)))
        if not np.isfinite(vp): vp = 999.
        stopper(vp, enc, pred)
        if stopper.early_stop: break

        if epoch >= WARM and UPD > 0 and epoch % UPD == 0:
            with torch.no_grad():
                xt = torch.tensor(X_train, dtype=torch.float32, device=device)
                mf = pred(enc(xt))
                ds.update(mf[:, 0].cpu().numpy(), mf[:, 1].cpu().numpy())

    stopper.restore(enc, pred)
    enc.eval(); pred.eval()

    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_te = enc(X_te_t)
        m_te = pred(z_te)
        tau_hat = (m_te[:, 1] - m_te[:, 0]).cpu().numpy() * sd_y
    pehe = float(np.sqrt(np.mean((mu1_te - mu0_te - tau_hat) ** 2)))
    LOGGER.info(f"ARTEMIS test PEHE: {pehe:.4f}")
    return enc, pred, z_te.cpu().numpy(), tau_hat, pehe


# ==============================================================================
# TRAINING CFRNet (TARNet + MMD penalty)
# ==============================================================================
def gaussian_mmd(z1: torch.Tensor, z2: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """MMD con kernel RBF tra due set di rappresentazioni."""
    def rbf_kernel(a, b, s):
        d2 = torch.cdist(a, b).pow(2)
        return torch.exp(-d2 / (2 * s ** 2))

    K_11 = rbf_kernel(z1, z1, sigma)
    K_22 = rbf_kernel(z2, z2, sigma)
    K_12 = rbf_kernel(z1, z2, sigma)
    n1, n2 = float(z1.shape[0]), float(z2.shape[0])
    return (K_11.sum() / (n1 * n1) + K_22.sum() / (n2 * n2)
            - 2.0 * K_12.sum() / (n1 * n2))


def train_cfrnet(X_train, X_val, X_test, Y_norm, T_tr, mu0_tr, mu1_tr,
                 mu0_te, mu1_te, train_idx, val_idx, sd_y, device, params):
    """
    CFRNet: TARNet + MMD penalty tra treated e control representations.
    Implementazione fedele a Shalit et al. 2017.
    """
    LOGGER.info("Training CFRNet...")
    torch.manual_seed(0); np.random.seed(0)

    LR = params['lr']; BS = params['batch_size']; LD = params['latent_dim']
    LMD = params['mmd_lambda']; SIG = params['mmd_sigma']
    EPOCHS = params['epochs']; PAT = params['patience']
    CLIP = params['clip_norm']; WD = params['main_weight_decay']
    HUB = params['huber_beta']; ENC_D = params['encoder_dropout']
    HEAD_D = params['head_dropout']

    T_train = T_tr[train_idx]
    Y_train = Y_norm[train_idx]

    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    T_tr_t = torch.tensor(T_train, dtype=torch.float32, device=device).view(-1).long()
    Y_tr_t = torch.tensor(Y_train, dtype=torch.float32, device=device).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    gt_val = mu1_tr[val_idx] - mu0_tr[val_idx]

    ds = TensorDataset(X_tr_t, T_tr_t, Y_tr_t)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=False)

    enc  = CATEEncoder(X_train.shape[1], LD, ENC_D).to(device)
    pred = OutcomeHead(LD, 2, False, 5.0, HEAD_D).to(device)

    opt = optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    stopper = EarlyStopping(PAT)

    for epoch in range(EPOCHS):
        enc.train(); pred.train()
        for xb, tb, yb in dl:
            opt.zero_grad()
            zb = enc(xb)
            mu_b = pred(zb)
            loss_sup = F.smooth_l1_loss(mu_b.gather(1, tb.unsqueeze(1)), yb, beta=HUB)

            # MMD tra treated e control
            mask1 = (tb == 1)
            mask0 = (tb == 0)
            if mask1.sum() > 1 and mask0.sum() > 1:
                z1_t = zb[mask1]
                z0_t = zb[mask0]
                mmd = gaussian_mmd(z1_t, z0_t, SIG)
            else:
                mmd = torch.tensor(0., device=device)

            loss = loss_sup + LMD * mmd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), CLIP)
            opt.step()

        sched.step()
        enc.eval(); pred.eval()
        with torch.no_grad():
            zv = enc(X_val_t)
            mv = pred(zv)
            tv = (mv[:, 1] - mv[:, 0]).cpu().numpy() * sd_y
        vp = float(np.sqrt(np.mean((gt_val - tv) ** 2)))
        if not np.isfinite(vp): vp = 999.
        stopper(vp, enc, pred)
        if stopper.early_stop: break

    stopper.restore(enc, pred)
    enc.eval(); pred.eval()

    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_te = enc(X_te_t)
        m_te = pred(z_te)
        tau_hat = (m_te[:, 1] - m_te[:, 0]).cpu().numpy() * sd_y
    pehe = float(np.sqrt(np.mean((mu1_te - mu0_te - tau_hat) ** 2)))
    LOGGER.info(f"CFRNet test PEHE: {pehe:.4f}")
    return enc, pred, z_te.cpu().numpy(), tau_hat, pehe


# ==============================================================================
# METRICHE LATENT SPACE
# ==============================================================================
def linear_mmd(Z: np.ndarray, T: np.ndarray) -> float:
    """MMD lineare tra treated e control (mean embedding difference)."""
    z1 = Z[T == 1]; z0 = Z[T == 0]
    if len(z1) == 0 or len(z0) == 0:
        return float('nan')
    diff = z1.mean(0) - z0.mean(0)
    return float(np.sqrt((diff ** 2).sum()))


def semantic_alignment(Z: np.ndarray, tau_true: np.ndarray,
                        n_sample: int = 5000, seed: int = 42) -> float:
    """Spearman corr tra distanze latenti e |tau_i - tau_j| su coppie campionate."""
    rng = np.random.default_rng(seed)
    N = Z.shape[0]
    n_pairs = min(n_sample, N * (N - 1) // 2)
    i_idx = rng.integers(0, N, size=n_pairs)
    j_idx = rng.integers(0, N, size=n_pairs)
    mask = (i_idx != j_idx)
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    if len(i_idx) < 10:
        return float('nan')
    lat_dist = np.sqrt(((Z[i_idx] - Z[j_idx]) ** 2).sum(1))
    ite_dist  = np.abs(tau_true[i_idx] - tau_true[j_idx])
    corr, _ = spearmanr(lat_dist, ite_dist)
    return float(corr)


def nearest_opposite_treatment_dist(Z: np.ndarray, T: np.ndarray) -> float:
    """Distanza media al nearest neighbor di trattamento opposto."""
    z1 = Z[T == 1]; z0 = Z[T == 0]
    if len(z1) == 0 or len(z0) == 0:
        return float('nan')
    # Per ogni unità treated → NN in control
    from sklearn.neighbors import NearestNeighbors
    nn0 = NearestNeighbors(n_neighbors=1).fit(z0)
    d0, _ = nn0.kneighbors(z1)
    nn1 = NearestNeighbors(n_neighbors=1).fit(z1)
    d1, _ = nn1.kneighbors(z0)
    return float(np.concatenate([d0.ravel(), d1.ravel()]).mean())


def treatment_probe_auc(Z: np.ndarray, T: np.ndarray) -> float:
    """AUC di un classificatore lineare che predice T da Z (con CV)."""
    scaler = StandardScaler()
    Z_sc = scaler.fit_transform(Z)
    lr = LogisticRegression(max_iter=500, C=1.0)
    scores = cross_val_score(lr, Z_sc, T.astype(int), cv=5, scoring='roc_auc')
    return float(scores.mean())


def compute_latent_metrics(Z: np.ndarray, T: np.ndarray,
                           tau_true: np.ndarray) -> Dict[str, float]:
    return {
        'Lin-MMD':       linear_mmd(Z, T),
        'Sem-Align':     semantic_alignment(Z, tau_true),
        'NOTD':          nearest_opposite_treatment_dist(Z, T),
        'Probe-AUC':     treatment_probe_auc(Z, T),
    }


# ==============================================================================
# RIDUZIONE DIMENSIONALE
# ==============================================================================
def tsne_embed(Z: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    LOGGER.info(f"t-SNE su {Z.shape[0]} punti ({Z.shape[1]}D → 2D)...")
    import sklearn
    from packaging import version as pkg_version
    # n_iter → max_iter in scikit-learn >= 1.2
    sk_ver = pkg_version.parse(sklearn.__version__)
    iter_kwarg = "max_iter" if sk_ver >= pkg_version.parse("1.2") else "n_iter"
    ts = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
              verbose=0, **{iter_kwarg: 1000})
    return ts.fit_transform(Z)


def umap_embed(Z: np.ndarray, n_neighbors: int = 15, seed: int = 42) -> Optional[np.ndarray]:
    try:
        import umap
        LOGGER.info(f"UMAP on {Z.shape[0]} points ({Z.shape[1]}D → 2D)...")
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            random_state=seed, verbose=False)
        return reducer.fit_transform(Z)
    except ImportError:
        LOGGER.warning("umap-learn not installed. Skipping UMAP.")
        return None


# ==============================================================================
# VISUALIZZAZIONE
# ==============================================================================
CMAP_ITE = "RdYlBu_r"   # divergente per ITE
CMAP_TRT = "bwr"         # rosso/blu per trattamento

def _scatter_panel(ax, xy, color_vals, cmap, vmin, vmax, title, xlabel, ylabel,
                   colorbar=True, cb_label="", s=6, alpha=0.6):
    sc = ax.scatter(xy[:, 0], xy[:, 1],
                    c=color_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=s, alpha=alpha, linewidths=0)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        cb = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label(cb_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return sc


def make_figure(
    tsne_artemis, tsne_cfrnet,
    umap_artemis, umap_cfrnet,
    tau_true, T_te,
    pehe_artemis, pehe_cfrnet,
    metrics_artemis, metrics_cfrnet,
    out_dir: str,
    use_umap: bool = True,
):
    """
    Layout finale per publication:

    Row 0  : [ARTEMIS t-SNE / ITE] [ARTEMIS t-SNE / T] [CFRNet t-SNE / ITE] [CFRNet t-SNE / T]
    Row 1  : [ARTEMIS UMAP / ITE ] [ARTEMIS UMAP / T ] [CFRNet UMAP / ITE ] [CFRNet UMAP / T ]
             (solo se use_umap=True)
    Bottom : metrics bar chart (full width, sotto le scatter)

    Each column has a border color to identify the model:
      blue = ARTEMIS
      red  = CFRNet
    """
    has_umap = use_umap and umap_artemis is not None
    scatter_rows = 2 if has_umap else 1
    n_rows_total = scatter_rows + 1   # +1 per la riga delle metriche

    fig = plt.figure(figsize=(15, 4.0 * scatter_rows + 3.5))

    # ---- Grid ----
    gs = gridspec.GridSpec(
        n_rows_total, 4,
        figure=fig,
        hspace=0.55, wspace=0.35,
        height_ratios=[4.0] * scatter_rows + [3.2],
    )

    fig.suptitle(
        "Latent Space Comparison: ARTEMIS vs CFRNet on IHDP (Simulation 0)",
        fontsize=12, fontweight='bold', y=1.01,
    )

    tau_min, tau_max = float(tau_true.min()), float(tau_true.max())

    # Panel border colors by model
    BORDER_ART = '#2563eb'  # blu ARTEMIS
    BORDER_CFR = '#dc2626'  # rosso CFRNet

    def add_scatter_row(gs_row, xy_art, xy_cfr, method_name):
        # Col 0 – ARTEMIS / ITE
        ax = fig.add_subplot(gs[gs_row, 0])
        _scatter_panel(ax, xy_art, tau_true, CMAP_ITE, tau_min, tau_max,
                       f"ARTEMIS – {method_name}\n(true ITE)",
                       f"{method_name}-1", f"{method_name}-2",
                       colorbar=True, cb_label="ITE", s=18, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_ART); spine.set_linewidth(1.8)

        # Col 1 – ARTEMIS / Treatment
        ax = fig.add_subplot(gs[gs_row, 1])
        sc = _scatter_panel(ax, xy_art, T_te.astype(float), CMAP_TRT, -0.1, 1.1,
                            f"ARTEMIS – {method_name}\n(treatment group)",
                            f"{method_name}-1", "",
                            colorbar=False, s=18, alpha=0.75)
        # Manual binary legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#1a54c7', label='Control (T=0)'),
            Patch(facecolor='#c72424', label='Treated  (T=1)'),
        ], fontsize=6.5, loc='lower right', framealpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_ART); spine.set_linewidth(1.8)

        # Col 2 – CFRNet / ITE
        ax = fig.add_subplot(gs[gs_row, 2])
        _scatter_panel(ax, xy_cfr, tau_true, CMAP_ITE, tau_min, tau_max,
                       f"CFRNet – {method_name}\n(true ITE)",
                       f"{method_name}-1", "",
                       colorbar=True, cb_label="ITE", s=18, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_CFR); spine.set_linewidth(1.8)

        # Col 3 – CFRNet / Treatment
        ax = fig.add_subplot(gs[gs_row, 3])
        _scatter_panel(ax, xy_cfr, T_te.astype(float), CMAP_TRT, -0.1, 1.1,
                       f"CFRNet – {method_name}\n(treatment group)",
                       f"{method_name}-1", "",
                       colorbar=False, s=18, alpha=0.75)
        ax.legend(handles=[
            Patch(facecolor='#1a54c7', label='Control (T=0)'),
            Patch(facecolor='#c72424', label='Treated  (T=1)'),
        ], fontsize=6.5, loc='lower right', framealpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_CFR); spine.set_linewidth(1.8)

    # ---- Scatter rows ----
    add_scatter_row(0, tsne_artemis, tsne_cfrnet, "t-SNE")
    if has_umap:
        add_scatter_row(1, umap_artemis, umap_cfrnet, "UMAP")

    # ---- Metrics row (full width, split 3:1) ----
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[scatter_rows, :],
        width_ratios=[3, 1], wspace=0.35,
    )

    # Bar chart metriche
    ax_m = fig.add_subplot(gs_bot[0])
    metric_names = list(metrics_artemis.keys())
    arrow_map = {'Lin-MMD': '↓', 'Sem-Align': '↑', 'NOTD': '↓', 'Probe-AUC': '→0.5'}
    x_labels = [f"{k}\n{arrow_map.get(k,'')}" for k in metric_names]
    v_art = np.array([metrics_artemis[k] for k in metric_names])
    v_cfr = np.array([metrics_cfrnet[k]  for k in metric_names])
    x = np.arange(len(metric_names))
    w = 0.32
    b1 = ax_m.bar(x - w/2, v_art, w, label=f'ARTEMIS (PEHE={pehe_artemis:.3f})',
                  color=BORDER_ART, alpha=0.85)
    b2 = ax_m.bar(x + w/2, v_cfr, w, label=f'CFRNet  (PEHE={pehe_cfrnet:.3f})',
                  color=BORDER_CFR, alpha=0.85)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(x_labels, fontsize=9)
    ax_m.set_title("Latent Space Diagnostics", fontsize=10, fontweight='bold')
    ax_m.legend(fontsize=8.5, loc='upper right')
    ax_m.grid(axis='y', alpha=0.3, linestyle='--')
    ax_m.set_ylim(0, max(v_art.max(), v_cfr.max()) * 1.22)
    for bar, v in zip(b1, v_art):
        ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                  f"{v:.3f}", ha='center', va='bottom', fontsize=7.5, color=BORDER_ART, fontweight='bold')
    for bar, v in zip(b2, v_cfr):
        ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                  f"{v:.3f}", ha='center', va='bottom', fontsize=7.5, color=BORDER_CFR, fontweight='bold')

    # Pannello note interpretative
    ax_note = fig.add_subplot(gs_bot[1])
    ax_note.axis('off')
    note_lines = (
        "Metric guide\n"
        "─────────────────────\n"
        "Lin-MMD ↓\n"
        "  treated vs control discrepancy\n\n"
        "Sem-Align ↑\n"
        "  Spearman(latent dist, |ITE dist|)\n"
        "  → ITE-organized geometry\n\n"
        "NOTD ↓\n"
        "  nearest opposite-treatment dist\n"
        "  → counterfactual support\n\n"
        "Probe-AUC → 0.5\n"
        "  treatment not decodable from Z"
    )
    ax_note.text(0.04, 0.97, note_lines, transform=ax_note.transAxes,
                 fontsize=7.2, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff',
                           edgecolor='#aaaacc', linewidth=1.2))

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f"latent_tsne_umap.{ext}")
        fig.savefig(path, bbox_inches='tight', dpi=150)
        LOGGER.info(f"Salvato: {path}")
    plt.close(fig)


# ==============================================================================
# TABELLA LATEX METRICHE
# ==============================================================================
def save_metrics_latex(metrics_a, metrics_c, pehe_a, pehe_c, out_dir):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Latent-space diagnostics on IHDP (simulation~0) comparing",
        r"\texttt{ARTEMIS} and \texttt{CFRNet}. Lin-MMD: distributional",
        r"discrepancy between treated and control (lower is better).",
        r"Sem-Align: Spearman correlation between latent distances and true",
        r"$|\tau_i - \tau_j|$ (higher is better). NOTD: mean nearest",
        r"opposite-treatment distance (lower is better). Probe-AUC: AUC of",
        r"linear treatment classifier (closer to 0.5 is better).}",
        r"\label{tab:latent_metrics_viz}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & \texttt{ARTEMIS} & \texttt{CFRNet} \\",
        r"\midrule",
    ]
    lines.append(rf"$\sqrt{{\epsilon_{{\text{{PEHE}}}}}}$ & {pehe_a:.4f} & {pehe_c:.4f} \\")
    lines.append(r"\midrule")
    arrow_map = {
        'Lin-MMD':   r'$\downarrow$',
        'Sem-Align': r'$\uparrow$',
        'NOTD':      r'$\downarrow$',
        'Probe-AUC': r'$\rightarrow 0.5$',
    }
    for k in metrics_a:
        va, vc = metrics_a[k], metrics_c[k]
        arrow = arrow_map.get(k, '')
        lines.append(rf"{k} {arrow} & {va:.4f} & {vc:.4f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_str = "\n".join(lines)

    path = os.path.join(out_dir, "latent_metrics_latex.txt")
    with open(path, 'w') as f:
        f.write(table_str)
    LOGGER.info(f"Tabella LaTeX salvata: {path}")
    print("\n" + "=" * 70)
    print("TABELLA LATEX (latent metrics)")
    print("=" * 70)
    print(table_str)
    return table_str


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ARTEMIS vs CFRNet latent space visualization")
    parser.add_argument("--sim_id", type=int, default=0, help="Indice simulazione IHDP (default: 0)")
    parser.add_argument("--no_umap", action="store_true", help="Salta UMAP (usa solo t-SNE)")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--mmd_lambda", type=float, default=1.0,
                        help="Peso MMD in CFRNet (default: 1.0)")
    args = parser.parse_args()

    LOGGER.info("Caricamento IHDP...")
    (X_tr, T_tr, YF_tr, mu0_tr, mu1_tr,
     X_te, T_te, YF_te, mu0_te, mu1_te) = load_ihdp()

    # Estrai simulazione
    X_tr_s, T_tr_s, YF_tr_s, mu0_tr_s, mu1_tr_s = get_sim(X_tr, T_tr, YF_tr, mu0_tr, mu1_tr, args.sim_id)
    X_te_s, T_te_s, YF_te_s, mu0_te_s, mu1_te_s = get_sim(X_te, T_te, YF_te, mu0_te, mu1_te, args.sim_id)
    tau_true_te = mu1_te_s - mu0_te_s

    LOGGER.info(f"Simulazione {args.sim_id}: "
                f"train={X_tr_s.shape[0]}, test={X_te_s.shape[0]}, "
                f"covariates={X_tr_s.shape[1]}")

    # Preprocessing
    rng = np.random.default_rng(42)
    perm = rng.permutation(X_tr_s.shape[0])
    n_val = int(0.2 * X_tr_s.shape[0])
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, X_val, X_test, Y_norm, mu_y, sd_y = preprocess(
        X_tr_s, YF_tr_s, T_tr_s, X_te_s, train_idx, val_idx)

    # Aggiorna params CFRNet
    cfrnet_params = copy.deepcopy(CFRNET_PARAMS)
    cfrnet_params['mmd_lambda'] = args.mmd_lambda

    # Training ARTEMIS
    enc_a, pred_a, Z_art, tau_hat_a, pehe_a = train_artemis(
        X_train, X_val, X_test, Y_norm, T_tr_s, mu0_tr_s, mu1_tr_s,
        mu0_te_s, mu1_te_s, train_idx, val_idx, sd_y, DEVICE, ARTEMIS_PARAMS)

    # Training CFRNet
    enc_c, pred_c, Z_cfr, tau_hat_c, pehe_c = train_cfrnet(
        X_train, X_val, X_test, Y_norm, T_tr_s, mu0_tr_s, mu1_tr_s,
        mu0_te_s, mu1_te_s, train_idx, val_idx, sd_y, DEVICE, cfrnet_params)

    # Metriche latent space
    LOGGER.info("Calcolo metriche spazio latente...")
    metrics_a = compute_latent_metrics(Z_art, T_te_s, tau_true_te)
    metrics_c = compute_latent_metrics(Z_cfr, T_te_s, tau_true_te)

    print("\n" + "=" * 60)
    print("LATENT SPACE METRICS (test set)")
    print("=" * 60)
    print(f"{'Metric':<15} {'ARTEMIS':>10} {'CFRNet':>10}")
    print("-" * 40)
    print(f"{'PEHE':<15} {pehe_a:>10.4f} {pehe_c:>10.4f}")
    for k in metrics_a:
        print(f"{k:<15} {metrics_a[k]:>10.4f} {metrics_c[k]:>10.4f}")

    # Salva CSV metriche
    df_m = pd.DataFrame([
        {'model': 'ARTEMIS', 'pehe': pehe_a, **metrics_a},
        {'model': 'CFRNet',  'pehe': pehe_c, **metrics_c},
    ])
    csv_path = os.path.join(OUT_DIR, "latent_metrics.csv")
    df_m.to_csv(csv_path, index=False, sep=';')
    LOGGER.info(f"Metriche CSV: {csv_path}")

    # t-SNE — calcolato SEPARATAMENTE per ciascun modello
    # (l'embedding congiunto non è corretto: mescolerebbe spazi latenti diversi)
    LOGGER.info("Calcolo t-SNE (separato per ARTEMIS e CFRNet)...")
    tsne_art = tsne_embed(Z_art, args.tsne_perplexity, seed=42)
    tsne_cfr = tsne_embed(Z_cfr, args.tsne_perplexity, seed=42)

    # UMAP — calcolato SEPARATAMENTE
    umap_art, umap_cfr = None, None
    if not args.no_umap:
        LOGGER.info("Calcolo UMAP (separato per ARTEMIS e CFRNet)...")
        umap_art = umap_embed(Z_art, args.umap_neighbors, seed=42)
        umap_cfr = umap_embed(Z_cfr, args.umap_neighbors, seed=42)

    # Figura
    LOGGER.info("Generazione figura...")
    make_figure(
        tsne_art, tsne_cfr,
        umap_art, umap_cfr,
        tau_true_te, T_te_s,
        pehe_a, pehe_c,
        metrics_a, metrics_c,
        OUT_DIR,
        use_umap=(not args.no_umap),
    )

    # Tabella LaTeX
    save_metrics_latex(metrics_a, metrics_c, pehe_a, pehe_c, OUT_DIR)

    LOGGER.info("Completato. Output in: " + OUT_DIR)


if __name__ == "__main__":
    main()