#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("ARTEMIS_IHDP_STRESS")


# ==============================================================================
# Helpers
# ==============================================================================

def ensure_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_single_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    keys = set(raw.files)

    def pick(*cands: str, required: bool = True):
        for c in cands:
            if c in keys:
                return np.asarray(raw[c])
        if required:
            raise KeyError(f"Missing keys among {cands} in {path}; found {sorted(keys)}")
        return None

    x = pick("x", "X", "feature", "features")
    t = pick("t", "T", "treatment", "w")
    y = pick("yf", "y", "Y", "outcome")
    mu0 = pick("mu0", "y0", "mu_0")
    mu1 = pick("mu1", "y1", "mu_1")

    # Stress-suite should already hand us single-rep files, but keep this safe.
    if x.ndim == 3:
        if x.shape[-1] == 1:
            x = x[:, :, 0]
        elif x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError(f"Expected single-rep X in {path}, got shape {x.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D X in {path}, got shape {x.shape}")

    def squeeze_target(a: np.ndarray, name: str) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 2:
            if a.shape[1] == 1:
                a = a[:, 0]
            elif a.shape[0] == 1:
                a = a[0]
            else:
                raise ValueError(f"Expected single-rep {name} in {path}, got shape {a.shape}")
        return ensure_1d(a)

    t = squeeze_target(t, "t").astype(int)
    y = squeeze_target(y, "y").astype(float)
    mu0 = squeeze_target(mu0, "mu0").astype(float)
    mu1 = squeeze_target(mu1, "mu1").astype(float)

    n = x.shape[0]
    for name, arr in [("t", t), ("y", y), ("mu0", mu0), ("mu1", mu1)]:
        if arr.shape[0] != n:
            raise ValueError(f"Mismatched length for {name}: {arr.shape[0]} vs X rows {n}")

    return x.astype(np.float32), t, y, mu0, mu1


# ==============================================================================
# Dataset / Pairing
# ==============================================================================

class EarlyStoppingPEHE:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_pehe = np.inf
        self.best_epoch = -1
        self.early_stop = False
        self.best_encoder_state = None
        self.best_predictor_state = None

    def __call__(self, val_pehe: float, encoder: nn.Module, predictor: nn.Module, epoch: int):
        if val_pehe < self.best_pehe - self.min_delta:
            self.best_pehe = float(val_pehe)
            self.best_epoch = int(epoch)
            self.counter = 0
            self.best_encoder_state = copy.deepcopy(encoder.state_dict())
            self.best_predictor_state = copy.deepcopy(predictor.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, encoder: nn.Module, predictor: nn.Module) -> None:
        if self.best_encoder_state is not None:
            encoder.load_state_dict(self.best_encoder_state)
            predictor.load_state_dict(self.best_predictor_state)


def compute_tau_threshold(mu0_hat, mu1_hat, perc=20, sample=100_000, rng=None) -> float:
    tau = (mu1_hat - mu0_hat).reshape(-1)
    n = tau.size
    if n < 2:
        return 0.1
    if rng is None:
        rng = np.random.default_rng()

    m = min(sample, n)
    idx1 = rng.integers(0, n, size=m)
    idx2 = rng.integers(0, n, size=m)
    diffs = np.abs(tau[idx1] - tau[idx2])
    thr = float(np.percentile(diffs, perc))

    tau_std = float(np.std(tau))
    tau_std = max(tau_std, 1e-6)
    thr_min = max(0.05 * tau_std, 1e-3)
    thr_max = max(1.00 * tau_std, thr_min)

    if not np.isfinite(thr):
        thr = 0.2 * tau_std
    return float(np.clip(thr, thr_min, thr_max))


def _empty_pair_batch(X, T, Y):
    empty_shape = (0,) + X.shape[1:]
    return (
        np.zeros(empty_shape, dtype=X.dtype), np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
        np.zeros((0,) + T.shape[1:], dtype=T.dtype), np.zeros(empty_shape, dtype=X.dtype),
        np.zeros((0,) + Y.shape[1:], dtype=Y.dtype), np.zeros((0,) + T.shape[1:], dtype=T.dtype),
        np.array([], dtype=np.int64),
    )


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    rng = np.random.default_rng(seed)
    tau = (mu1_hat - mu0_hat).reshape(-1)
    N = tau.shape[0]
    if N < 2:
        return _empty_pair_batch(X, T, Y)

    n_pairs = int(min(max(1, n_pairs), N))
    half = n_pairs // 2
    used = set()
    sim_pairs = []
    dis_pairs = []

    def add_pair(i, j, label, container):
        if i == j:
            return
        key = (min(i, j), max(i, j))
        if key in used:
            return
        used.add(key)
        container.append((i, j, label))

    attempts = 0
    max_attempts = max(50, n_pairs * 10)

    while len(sim_pairs) < half and attempts < max_attempts:
        i = int(rng.integers(0, N))
        diffs = np.abs(tau - tau[i])
        cand = np.where(diffs < thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            j = int(rng.choice(cand))
            add_pair(i, j, 1, sim_pairs)
        attempts += 1

    attempts = 0
    while len(dis_pairs) < (n_pairs - len(sim_pairs)) and attempts < max_attempts:
        i = int(rng.integers(0, N))
        diffs = np.abs(tau - tau[i])
        cand = np.where(diffs >= thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            j = int(rng.choice(cand))
            add_pair(i, j, 0, dis_pairs)
        attempts += 1

    pairs = sim_pairs + dis_pairs

    if len(pairs) < max(1, n_pairs // 2):
        needed = n_pairs - len(pairs)
        for _ in range(needed):
            i = int(rng.integers(0, N))
            j = int(rng.integers(0, N - 1))
            if j >= i:
                j += 1
            label = 1 if np.abs(tau[i] - tau[j]) < thr else 0
            add_pair(i, j, label, pairs)

    if not pairs:
        return _empty_pair_batch(X, T, Y)

    rng.shuffle(pairs)
    idx_a, idx_b, labels = zip(*pairs)
    return (
        X[np.array(idx_a)], Y[np.array(idx_a)], T[np.array(idx_a)],
        X[np.array(idx_b)], Y[np.array(idx_b)], T[np.array(idx_b)],
        np.array(labels, dtype=np.int64)
    )


class DynamicContrastiveCausalDS(Dataset):
    def __init__(self, X_all, T_all, Y_all, mu0_hat, mu1_hat, bs=256, perc=20, sample_for_thr_calc=100_000, seed=0):
        self.X_all = X_all
        self.T_all = T_all
        self.Y_all = Y_all
        self.bs = int(bs)
        self.perc = float(perc)
        self.sample_for_thr_calc = int(sample_for_thr_calc)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.epoch = 0

        if mu0_hat is None or mu1_hat is None:
            self.current_mu0_hat = np.zeros(X_all.shape[0], dtype=np.float32)
            self.current_mu1_hat = np.zeros(X_all.shape[0], dtype=np.float32)
        else:
            self.current_mu0_hat = mu0_hat
            self.current_mu1_hat = mu1_hat

        self.update_threshold()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def update_threshold(self):
        self.thr = compute_tau_threshold(
            self.current_mu0_hat,
            self.current_mu1_hat,
            perc=self.perc,
            sample=self.sample_for_thr_calc,
            rng=self.rng,
        )

    def update_ite_estimates(self, mu0_hat, mu1_hat):
        if mu0_hat is not None and mu1_hat is not None:
            self.current_mu0_hat = mu0_hat
            self.current_mu1_hat = mu1_hat
            self.update_threshold()

    def __len__(self):
        return int(np.ceil(self.X_all.shape[0] / self.bs))

    def __getitem__(self, idx: int):
        seed = (self.seed + 1000003 * self.epoch + 9176 * int(idx)) & 0xFFFFFFFF
        x1, y1, t1, x2, y2, t2, lab = make_pairs_from_hat(
            self.X_all, self.T_all, self.Y_all, self.current_mu0_hat, self.current_mu1_hat,
            self.thr, self.bs, seed=seed
        )
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.float32),
            torch.tensor(t1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y2, dtype=torch.float32),
            torch.tensor(t2, dtype=torch.float32),
            torch.tensor(lab, dtype=torch.long)
        )


# ==============================================================================
# Model
# ==============================================================================

class MineStatisticsNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim, 1))
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def mine_lower_bound_stable(t_network, x, y, clamp_val=10.0):
    t_joint = t_network(x, y).view(-1)
    y_shuffle = y[torch.randperm(y.shape[0], device=y.device)]
    t_marg = t_network(x, y_shuffle).view(-1)
    t_marg = torch.clamp(t_marg, -clamp_val, clamp_val)
    m = torch.max(t_marg)
    log_mean_exp = m + torch.log(torch.mean(torch.exp(t_marg - m)) + 1e-8)
    return t_joint.mean() - log_mean_exp


class CATEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)), nn.GELU(),
            nn.Dropout(0.15),
            spectral_norm(nn.Linear(128, 128)), nn.GELU(),
            nn.Dropout(0.15),
            spectral_norm(nn.Linear(128, latent_dim)), nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        return self.network(x)


class OutcomeHead(nn.Module):
    def __init__(self, latent_dim, clip_val=5.0):
        super().__init__()
        self.head0 = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 64)), nn.GELU(),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(64, 1)),
            nn.Hardtanh(min_val=-clip_val, max_val=clip_val)
        )
        self.head1 = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 64)), nn.GELU(),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(64, 1)),
            nn.Hardtanh(min_val=-clip_val, max_val=clip_val)
        )

    def forward(self, z):
        return self.head0(z), self.head1(z)


def contrastive_loss(z1, z2, label, margin=1.0):
    dist_sq = torch.sum(torch.pow(z1 - z2, 2), dim=1)
    loss_sim = label * dist_sq
    loss_dissim = (1 - label) * torch.pow(torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0), 2)
    return torch.mean(loss_sim + loss_dissim) / 2


# ==============================================================================
# Metrics
# ==============================================================================

def sqrt_PEHE_with_diff(y: np.ndarray, hat_tau: np.ndarray) -> float:
    tau = (y[:, 1] - y[:, 0])
    return float(np.sqrt(np.mean((tau - hat_tau) ** 2)))


def eps_ATE_diff(ite: np.ndarray, hat_ite: np.ndarray) -> float:
    return float(np.abs(np.mean(ite) - np.mean(hat_ite)))


# ==============================================================================
# Training
# ==============================================================================

DEFAULT_PARAMS: Dict[str, Any] = {
    'lr': 0.0013734160981144544,
    'batch_size': 128,
    'latent_dim': 64,
    'alpha': 0.6521571285476828,
    'perc': 31,
    'ite_update_freq': 5,
    'critic_steps': 6,
    'lr_critic': 0.00036468471047333836,
    'outcome_clip_factor': 3.0,
    'margin': 1.0,
    'epochs': 3000,
    'patience': 50,
    'warmup_epochs': 30,
    'clip_norm': 1.0,
    'mi_bias_weight': 0.5,
    'mi_outcome_weight': 0.1,
}


def train_single_run(
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: str,
    hyperparams: Dict[str, Any],
    seed: int,
) -> Dict[str, float]:
    set_all_seeds(seed)

    LR_MAIN = hyperparams.get('lr', 1e-3)
    BATCH_SIZE = hyperparams.get('batch_size', 64)
    LATENT_DIM = hyperparams.get('latent_dim', 32)
    ALPHA = hyperparams.get('alpha', 0.5)
    PERC_THR = hyperparams.get('perc', 20)
    ITE_UPDATE_FREQ = hyperparams.get('ite_update_freq', 5)
    MINE_CRITIC_STEPS = hyperparams.get('critic_steps', 5)
    LR_CRITIC = hyperparams.get('lr_critic', 1e-3)
    MARGIN = hyperparams.get('margin', 1.0)
    EPOCHS = hyperparams.get('epochs', 3000)
    PATIENCE = hyperparams.get('patience', 50)
    WARMUP_EPOCHS = hyperparams.get('warmup_epochs', 30)
    CLIP_NORM = hyperparams.get('clip_norm', 1.0)

    MI_BIAS_WEIGHT = hyperparams.get('mi_bias_weight', 0.5)
    MI_OUTCOME_WEIGHT = hyperparams.get('mi_outcome_weight', 0.1)

    variant = hyperparams.get("variant", "full")
    use_mi = variant not in {"no_mi", "factual_only"}
    use_contrastive = variant not in {"no_contrastive", "factual_only"}

    LOGGER.info(
        "variant=%s | use_mi=%s | use_contrastive=%s | seed=%d",
        variant, use_mi, use_contrastive, seed,
    )

    X_tr_s, T_tr_s, Y_tr_s, mu0_tr_s, mu1_tr_s = train_data
    X_te_s, T_te_s, Y_te_s, mu0_te_s, mu1_te_s = test_data

    n_total = X_tr_s.shape[0]
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    def get_continuous_indices(X):
        is_cont = []
        for c in range(X.shape[1]):
            unique_vals = np.unique(X[:, c])
            if len(unique_vals) > 2:
                is_cont.append(c)
        return is_cont

    cont_indices = get_continuous_indices(X_tr_s)

    X_train_processed = X_tr_s[train_idx].copy()
    X_val_processed = X_tr_s[val_idx].copy()
    X_test_processed = X_te_s.copy()

    if cont_indices:
        x_train_cont = X_train_processed[:, cont_indices]
        x_mean = np.mean(x_train_cont, axis=0, keepdims=True)
        x_std = np.std(x_train_cont, axis=0, keepdims=True)
        x_std = np.maximum(x_std, 1e-6)

        X_train_processed[:, cont_indices] = (X_train_processed[:, cont_indices] - x_mean) / x_std
        X_val_processed[:, cont_indices] = (X_val_processed[:, cont_indices] - x_mean) / x_std
        X_test_processed[:, cont_indices] = (X_test_processed[:, cont_indices] - x_mean) / x_std

    y_train_raw = Y_tr_s[train_idx]
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if y_std < 1e-6:
        y_std = 1.0

    Y_tr_norm_all = (Y_tr_s - y_mean) / y_std

    max_y_obs_std = float(np.max(np.abs(Y_tr_norm_all)))
    clip_factor = hyperparams.get('outcome_clip_factor', 1.5)
    calculated_clip_val = max(max_y_obs_std * clip_factor, 3.0)

    ds_train = DynamicContrastiveCausalDS(
        X_all=X_train_processed,
        T_all=T_tr_s[train_idx],
        Y_all=Y_tr_norm_all[train_idx],
        mu0_hat=None,
        mu1_hat=None,
        bs=BATCH_SIZE,
        perc=PERC_THR,
        seed=seed,
    )
    dl_train = DataLoader(ds_train, batch_size=None, shuffle=True)

    X_val_t = torch.tensor(X_val_processed, dtype=torch.float32).to(device)
    gt_val_ite = mu1_tr_s[val_idx] - mu0_tr_s[val_idx]

    input_dim = X_tr_s.shape[1]
    encoder = CATEEncoder(input_dim, LATENT_DIM).to(device)
    predictor = OutcomeHead(LATENT_DIM, clip_val=calculated_clip_val).to(device)

    mine_bias = MineStatisticsNetwork(LATENT_DIM + 1).to(device)
    mine_outcome = MineStatisticsNetwork(LATENT_DIM + 1).to(device)

    opt_main = optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR_MAIN,
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS, eta_min=1e-6)

    opt_mine_bias = optim.AdamW(mine_bias.parameters(), lr=LR_CRITIC, weight_decay=1e-3)
    opt_mine_outcome = optim.AdamW(mine_outcome.parameters(), lr=LR_CRITIC, weight_decay=1e-3)

    early_stopper = EarlyStoppingPEHE(patience=PATIENCE)
    final_val_pehe = 999.0
    best_epoch = -1

    for epoch in range(EPOCHS):
        ds_train.set_epoch(epoch)
        RAMP = 80
        alpha_val = 0.0 if epoch < WARMUP_EPOCHS else min(ALPHA, (epoch - WARMUP_EPOCHS) / RAMP * ALPHA)

        encoder.train()
        predictor.train()

        for batch in dl_train:
            x1, y1, t1, x2, y2, t2, label = [b.to(device) for b in batch]
            if x1.shape[0] == 0:
                continue

            label = label.float()
            t1_r, y1_r = t1.view(-1, 1), y1.view(-1, 1)

            with torch.no_grad():
                z1_detached = encoder(x1).detach()

            if alpha_val > 0 and use_mi:
                for _ in range(MINE_CRITIC_STEPS):
                    opt_mine_bias.zero_grad()
                    mi_b = mine_lower_bound_stable(mine_bias, z1_detached, t1_r)
                    (-mi_b).backward()
                    torch.nn.utils.clip_grad_norm_(mine_bias.parameters(), CLIP_NORM)
                    opt_mine_bias.step()

                for _ in range(MINE_CRITIC_STEPS):
                    opt_mine_outcome.zero_grad()
                    mi_o = mine_lower_bound_stable(mine_outcome, z1_detached, y1_r)
                    (-mi_o).backward()
                    torch.nn.utils.clip_grad_norm_(mine_outcome.parameters(), CLIP_NORM)
                    opt_mine_outcome.step()

            for p in mine_bias.parameters():
                p.requires_grad = False
            for p in mine_outcome.parameters():
                p.requires_grad = False

            opt_main.zero_grad()
            z1 = encoder(x1)
            mu0_p, mu1_p = predictor(z1)

            y_pred = torch.where(t1_r == 1, mu1_p, mu0_p)
            loss_sup = F.smooth_l1_loss(y_pred, y1_r, beta=1.0)

            loss_extra = torch.tensor(0.0, device=device)
            if alpha_val > 0:
                if use_contrastive:
                    z2 = encoder(x2)
                    l_cont = contrastive_loss(z1, z2, label, margin=MARGIN)
                    loss_extra = loss_extra + l_cont

                if use_mi:
                    mi_b = mine_lower_bound_stable(mine_bias, z1, t1_r)
                    mi_o = mine_lower_bound_stable(mine_outcome, z1, y1_r)
                    mi_b_clamped = torch.clamp(mi_b, -5.0, 5.0)
                    mi_o_clamped = torch.clamp(mi_o, -5.0, 5.0)
                    loss_extra = loss_extra + MI_BIAS_WEIGHT * mi_b_clamped - MI_OUTCOME_WEIGHT * mi_o_clamped

            loss_main = loss_sup + alpha_val * loss_extra
            loss_main.backward()

            for p in mine_bias.parameters():
                p.requires_grad = True
            for p in mine_outcome.parameters():
                p.requires_grad = True

            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), CLIP_NORM)
            opt_main.step()

        scheduler.step()

        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            z_val = encoder(X_val_t)
            m0_val_p, m1_val_p = predictor(z_val)
            ite_val_pred_norm = (m1_val_p - m0_val_p).cpu().numpy().ravel()
            ite_val_pred = ite_val_pred_norm * y_std
            val_pehe = np.sqrt(np.mean((gt_val_ite - ite_val_pred) ** 2))

        if not np.isfinite(val_pehe):
            val_pehe = 999.0

        early_stopper(val_pehe, encoder, predictor, epoch)
        if early_stopper.best_epoch >= 0:
            best_epoch = early_stopper.best_epoch
        final_val_pehe = early_stopper.best_pehe

        if early_stopper.early_stop:
            break

        if use_contrastive and epoch >= WARMUP_EPOCHS and ITE_UPDATE_FREQ > 0 and (epoch % ITE_UPDATE_FREQ == 0):
            with torch.no_grad():
                x_tr_curr = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
                z_full = encoder(x_tr_curr)
                fm0, fm1 = predictor(z_full)
                ds_train.update_ite_estimates(fm0.cpu().numpy().ravel(), fm1.cpu().numpy().ravel())

    early_stopper.restore_best_weights(encoder, predictor)
    encoder.eval()
    predictor.eval()

    X_te_t = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_te = encoder(X_te_t)
        m0_te, m1_te = predictor(z_te)
        ite_pred = (m1_te - m0_te).cpu().numpy().ravel() * y_std

    y_true_te = np.stack([mu0_te_s, mu1_te_s], axis=1)
    pehe = sqrt_PEHE_with_diff(y_true_te, ite_pred)
    ate_err = eps_ATE_diff(mu1_te_s - mu0_te_s, ite_pred)

    return {
        'val_pehe': float(final_val_pehe),
        'test_pehe': float(pehe),
        'test_ate': float(ate_err),
        'epochs': int(best_epoch + 1 if best_epoch >= 0 else epoch + 1),
        'z_te': z_te.cpu().numpy().astype(np.float32),
        't_te': np.asarray(T_te_s).astype(np.int64),
        'tau_true_te': (np.asarray(mu1_te_s) - np.asarray(mu0_te_s)).astype(np.float32),
        'tau_hat_te': np.asarray(ite_pred).astype(np.float32),
    }


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone IHDP ARTEMIS runner for stress-suite")
    p.add_argument("--train_path", type=Path, required=True)
    p.add_argument("--test_path", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--variant", type=str, default="full", choices=["full", "no_mi", "no_contrastive", "factual_only"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=DEFAULT_PARAMS['epochs'])
    p.add_argument("--patience", type=int, default=DEFAULT_PARAMS['patience'])
    p.add_argument("--warmup_epochs", type=int, default=DEFAULT_PARAMS['warmup_epochs'])
    p.add_argument("--batch_size", type=int, default=DEFAULT_PARAMS['batch_size'])
    p.add_argument("--mi_bias_weight", type=float, default=DEFAULT_PARAMS['mi_bias_weight'])
    p.add_argument("--mi_outcome_weight", type=float, default=DEFAULT_PARAMS['mi_outcome_weight'])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info("Using device=%s", device)
    LOGGER.info("Reading train=%s | test=%s", args.train_path, args.test_path)

    X_tr, T_tr, Y_tr, mu0_tr, mu1_tr = load_single_npz(args.train_path)
    X_te, T_te, Y_te, mu0_te, mu1_te = load_single_npz(args.test_path)

    params = dict(DEFAULT_PARAMS)
    params['variant'] = args.variant
    params['epochs'] = int(args.epochs)
    params['patience'] = int(args.patience)
    params['warmup_epochs'] = int(args.warmup_epochs)
    params['batch_size'] = int(args.batch_size)
    params['mi_bias_weight'] = float(args.mi_bias_weight)
    params['mi_outcome_weight'] = float(args.mi_outcome_weight)

    result = train_single_run(
        train_data=(X_tr, T_tr, Y_tr, mu0_tr, mu1_tr),
        test_data=(X_te, T_te, Y_te, mu0_te, mu1_te),
        device=device,
        hyperparams=params,
        seed=int(args.seed),
    )

    embeddings_path = args.out_dir / "test_embeddings.npz"
    np.savez(
        embeddings_path,
        z=result["z_te"],
        t=result["t_te"],
        tau_true=result["tau_true_te"],
        tau_hat=result["tau_hat_te"],
    )

    metrics = {
        'variant': args.variant,
        'seed': int(args.seed),
        'val_pehe': result['val_pehe'],
        'test_pehe': result['test_pehe'],
        'test_ate': result['test_ate'],
        'epochs': result['epochs'],
        'train_path': str(args.train_path),
        'test_path': str(args.test_path),
        'device': device,
        'artifact_paths': {
            'test_embeddings': str(embeddings_path),
        },
    }

    with (args.out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(args.out_dir / 'metrics.csv', index=False)
    LOGGER.info("Done. metrics.json written to %s", args.out_dir / 'metrics.json')


if __name__ == '__main__':
    main()
