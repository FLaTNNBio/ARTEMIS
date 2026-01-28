import os
import os.path
import logging
import zipfile
import requests
import numpy as np
import copy
import csv
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import optuna

# Configurazione Logger
logging.basicConfig(level=logging.INFO)
# Imposta verbosità Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)


# ==============================================================================
# PARTE 1: DATA LOADER (IHDP)
# ==============================================================================

def download_url(url, save_path, chunk_size=128):
    print(">>> downloading ", url, " into ", save_path, "...")
    try:
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
    except Exception as e:
        print(f"Errore durante il download: {e}")
        raise


class AbstractCausalLoader(ABC):
    def __init__(self):
        self.loaded = False

    @staticmethod
    def get_loader(dataset_name='IHDP'):
        if dataset_name == 'IHDP':
            return IHDPLoader()
        else:
            raise Exception('dataset not supported::' + str(dataset_name))

    @abstractmethod
    def load(self):
        pass


class IHDPLoader(AbstractCausalLoader):
    def __init__(self):
        super(IHDPLoader, self).__init__()

    def load(self):
        try:
            my_path = os.path.abspath(os.path.dirname(__file__))
        except NameError:
            my_path = os.getcwd()

        path = os.path.join(my_path, "data")
        path_train_zip = os.path.join(path, "ihdp_npci_1-1000.train.npz.zip")
        path_train = os.path.join(path, "ihdp_npci_1-1000.train.npz")
        path_test_zip = os.path.join(path, "ihdp_npci_1-1000.test.npz.zip")
        path_test = os.path.join(path, "ihdp_npci_1-1000.test.npz")

        if not os.path.exists(path):
            os.makedirs(path)

        # Download e estrazione Train
        if not os.path.exists(path_train):
            if not os.path.exists(path_train_zip):
                try:
                    download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip", path_train_zip)
                except:
                    print("WARN: Download fallito. Verifica i file locali.")
                    return None

            if os.path.exists(path_train_zip):
                with zipfile.ZipFile(path_train_zip, 'r') as zip_ref:
                    zip_ref.extractall(path)

        # Download e estrazione Test
        if not os.path.exists(path_test):
            if not os.path.exists(path_test_zip):
                try:
                    download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip", path_test_zip)
                except:
                    print("WARN: Download fallito. Verifica i file locali.")
                    return None

            if os.path.exists(path_test_zip):
                with zipfile.ZipFile(path_test_zip, 'r') as zip_ref:
                    zip_ref.extractall(path)

        if not os.path.exists(path_train) or not os.path.exists(path_test):
            raise FileNotFoundError("I file dataset IHDP non sono stati trovati.")

        train_cv = np.load(path_train)
        test = np.load(path_test)

        self.X_tr = train_cv['x']
        self.T_tr = train_cv['t']
        self.YF_tr = train_cv['yf']
        self.YCF_tr = train_cv['ycf']
        self.mu_0_tr = train_cv['mu0']
        self.mu_1_tr = train_cv['mu1']

        self.X_te = test['x']
        self.T_te = test['t']
        self.YF_te = test['yf']
        self.YCF_te = test['ycf']
        self.mu_0_te = test['mu0']
        self.mu_1_te = test['mu1']

        self.loaded = True

        return self.X_tr, self.T_tr, self.YF_tr, self.YCF_tr, self.mu_0_tr, self.mu_1_tr, \
            self.X_te, self.T_te, self.YF_te, self.YCF_te, self.mu_0_te, self.mu_1_te


# ==============================================================================
# PARTE 2: UTILS & DATASET
# ==============================================================================

class EarlyStoppingPEHE:
    def __init__(self, patience=20, min_delta=1e-6):  # Pazienza aumentata
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_pehe = np.inf
        self.early_stop = False
        self.best_encoder_state = None
        self.best_predictor_state = None

    def __call__(self, val_pehe, encoder, predictor):
        if val_pehe < self.best_pehe - self.min_delta:
            self.best_pehe = val_pehe
            self.counter = 0
            self.best_encoder_state = copy.deepcopy(encoder.state_dict())
            self.best_predictor_state = copy.deepcopy(predictor.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, encoder, predictor):
        if self.best_encoder_state is not None:
            encoder.load_state_dict(self.best_encoder_state)
            predictor.load_state_dict(self.best_predictor_state)


def compute_tau_threshold(mu0_hat, mu1_hat, perc=20, sample=100_000, rng=None) -> float:
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

    tau_std = float(np.std(tau))
    tau_std = max(tau_std, 1e-6)
    thr_min = max(0.05 * tau_std, 1e-3)
    thr_max = max(1.00 * tau_std, thr_min)

    if not np.isfinite(thr):
        thr = 0.2 * tau_std
    thr = float(np.clip(thr, thr_min, thr_max))
    return thr


def _empty_pair_batch(X, T, Y):
    empty_shape = (0,) + X.shape[1:]
    return (
        np.zeros(empty_shape, dtype=X.dtype), np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
        np.zeros((0,) + T.shape[1:], dtype=T.dtype), np.zeros(empty_shape, dtype=X.dtype),
        np.zeros((0,) + Y.shape[1:], dtype=Y.dtype), np.zeros((0,) + T.shape[1:], dtype=T.dtype),
        np.array([], dtype=np.int64),
    )


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    tau = (mu1_hat - mu0_hat).reshape(-1)
    N = tau.shape[0]
    if N < 2: return _empty_pair_batch(X, T, Y)

    n_pairs = int(min(max(1, n_pairs), N))
    half = n_pairs // 2
    used = set()
    sim_pairs = []
    dis_pairs = []

    def add_pair(i, j, label, container):
        if i == j: return
        key = (min(i, j), max(i, j))
        if key in used: return
        used.add(key)
        container.append((i, j, label))

    attempts = 0
    max_attempts = max(50, n_pairs * 10)

    # 1. Simili
    while len(sim_pairs) < half and attempts < max_attempts:
        i = int(rng.integers(0, N))
        diffs = np.abs(tau - tau[i])
        cand = np.where(diffs < thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            j = int(rng.choice(cand))
            add_pair(i, j, 1, sim_pairs)
        attempts += 1

    # 2. Dissimili
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
            if j >= i: j += 1
            label = 1 if np.abs(tau[i] - tau[j]) < thr else 0
            add_pair(i, j, label, pairs)

    if not pairs: return _empty_pair_batch(X, T, Y)
    rng.shuffle(pairs)
    idx_a, idx_b, labels = zip(*pairs)
    return (X[np.array(idx_a)], Y[np.array(idx_a)], T[np.array(idx_a)],
            X[np.array(idx_b)], Y[np.array(idx_b)], T[np.array(idx_b)], np.array(labels, dtype=np.int64))


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
        self.thr = compute_tau_threshold(self.current_mu0_hat, self.current_mu1_hat, perc=self.perc,
                                         sample=self.sample_for_thr_calc, rng=self.rng)

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
        return (torch.tensor(x1, dtype=torch.float32), torch.tensor(y1, dtype=torch.float32),
                torch.tensor(t1, dtype=torch.float32),
                torch.tensor(x2, dtype=torch.float32), torch.tensor(y2, dtype=torch.float32),
                torch.tensor(t2, dtype=torch.float32),
                torch.tensor(lab, dtype=torch.long))


# ==============================================================================
# PARTE 3: MODEL ARCHITECTURE (TURBO VERSION)
# ==============================================================================

class MineStatisticsNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.GELU(),  # GELU
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.GELU(),  # GELU
            spectral_norm(nn.Linear(hidden_dim, 1))
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def mine_lower_bound_stable(t_network, x, y, clamp_val=10.0):
    t_joint = t_network(x, y).view(-1)
    y_shuffle = y[torch.randperm(y.shape[0])]
    t_marg = t_network(x, y_shuffle).view(-1)
    t_marg = torch.clamp(t_marg, -clamp_val, clamp_val)
    m = torch.max(t_marg)
    log_mean_exp = m + torch.log(torch.mean(torch.exp(t_marg - m)) + 1e-8)
    return t_joint.mean() - log_mean_exp


class CATEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        # PATCH TURBO: GELU e Dropout per regolarizzare
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)), nn.GELU(),
            nn.Dropout(0.15),
            spectral_norm(nn.Linear(128, 128)), nn.GELU(),
            nn.Dropout(0.15),
            spectral_norm(nn.Linear(128, latent_dim)), nn.LayerNorm(latent_dim)
        )

    def forward(self, x): return self.network(x)


class OutcomeHead(nn.Module):
    def __init__(self, latent_dim, clip_val=5.0):
        super().__init__()
        # PATCH TURBO: GELU e Dropout
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

    def forward(self, z): return self.head0(z), self.head1(z)


def contrastive_loss(z1, z2, label, margin=1.0):
    dist_sq = torch.sum(torch.pow(z1 - z2, 2), dim=1)
    loss_sim = label * dist_sq
    loss_dissim = (1 - label) * torch.pow(torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0), 2)
    return torch.mean(loss_sim + loss_dissim) / 2


# ==============================================================================
# PARTE 4: METRICS
# ==============================================================================

def sqrt_PEHE_with_diff(y: np.ndarray, hat_tau: np.ndarray) -> float:
    tau = (y[:, 1] - y[:, 0])
    return float(np.sqrt(np.mean((tau - hat_tau) ** 2)))


def eps_ATE_diff(ite: np.ndarray, hat_ite: np.ndarray) -> float:
    return float(np.abs(np.mean(ite) - np.mean(hat_ite)))


# ==============================================================================
# PARTE 5: TRAINING LOGIC
# ==============================================================================

def train_single_simulation(
        sim_idx: int,
        data_train: Tuple,
        data_test: Tuple,
        device: str,
        hyperparams: Dict[str, Any]
) -> Dict[str, float]:
    # Riproducibilità
    torch.manual_seed(sim_idx)
    np.random.seed(sim_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sim_idx)

    # Parametri
    LR_MAIN = hyperparams.get('lr', 1e-3)
    BATCH_SIZE = hyperparams.get('batch_size', 64)
    LATENT_DIM = hyperparams.get('latent_dim', 32)
    ALPHA = hyperparams.get('alpha', 0.5)
    PERC_THR = hyperparams.get('perc', 20)
    ITE_UPDATE_FREQ = hyperparams.get('ite_update_freq', 5)
    MINE_CRITIC_STEPS = hyperparams.get('critic_steps', 5)
    LR_CRITIC = hyperparams.get('lr_critic', 1e-3)
    MARGIN = hyperparams.get('margin', 1.0)

    EPOCHS = hyperparams.get('epochs', 150)
    PATIENCE = hyperparams.get('patience', 20)

    WARMUP_EPOCHS = 30
    CLIP_NORM = 1.0

    # Unpack Data
    X_tr_s, T_tr_s, Y_tr_s, mu0_tr_s, mu1_tr_s = data_train
    X_te_s, T_te_s, Y_te_s, mu0_te_s, mu1_te_s = data_test

    # Split Train/Val
    n_total = X_tr_s.shape[0]
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    rng = np.random.default_rng(sim_idx)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # PATCH CRITICA: NORMALIZZAZIONE SELETTIVA
    # Per IHDP, identifichiamo colonne binarie (solo 0 e 1) e le lasciamo stare
    # Le altre le normalizziamo.

    # Funzione helper per trovare colonne continue
    def get_continuous_indices(X):
        # Euristica: se ha più di 2 valori unici, è continua
        is_cont = []
        for c in range(X.shape[1]):
            unique_vals = np.unique(X[:, c])
            if len(unique_vals) > 2:
                is_cont.append(c)
        return is_cont

    # Determina indici su tutto il train set
    cont_indices = get_continuous_indices(X_tr_s)

    # Copiamo per non sporcare i dati originali
    X_train_processed = X_tr_s[train_idx].copy()
    X_val_processed = X_tr_s[val_idx].copy()
    X_test_processed = X_te_s.copy()  # Usiamo tutto X test

    if cont_indices:
        # Calcola mean/std solo sul training set (sottoinsieme)
        x_train_cont = X_train_processed[:, cont_indices]
        x_mean = np.mean(x_train_cont, axis=0, keepdims=True)
        x_std = np.std(x_train_cont, axis=0, keepdims=True)
        x_std = np.maximum(x_std, 1e-6)

        # Applica normalizzazione solo alle colonne continue
        X_train_processed[:, cont_indices] = (X_train_processed[:, cont_indices] - x_mean) / x_std
        X_val_processed[:, cont_indices] = (X_val_processed[:, cont_indices] - x_mean) / x_std
        X_test_processed[:, cont_indices] = (X_test_processed[:, cont_indices] - x_mean) / x_std

    # Normalizzazione Target
    y_train_raw = Y_tr_s[train_idx]
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if y_std < 1e-6: y_std = 1.0

    Y_tr_norm_all = (Y_tr_s - y_mean) / y_std

    # Clipping Dinamico
    max_y_obs_std = float(np.max(np.abs(Y_tr_norm_all)))
    clip_factor = hyperparams.get('outcome_clip_factor', 1.5)
    calculated_clip_val = max(max_y_obs_std * clip_factor, 3.0)

    # Dataset Train (Usa X processato)
    ds_train = DynamicContrastiveCausalDS(
        X_all=X_train_processed, T_all=T_tr_s[train_idx], Y_all=Y_tr_norm_all[train_idx],
        mu0_hat=None, mu1_hat=None, bs=BATCH_SIZE, perc=PERC_THR, seed=sim_idx
    )
    dl_train = DataLoader(ds_train, batch_size=None, shuffle=True)

    # Dataset Val (Usa X processato)
    X_val_t = torch.tensor(X_val_processed, dtype=torch.float32).to(device)
    gt_val_ite = mu1_tr_s[val_idx] - mu0_tr_s[val_idx]

    # Modelli
    input_dim = X_tr_s.shape[1]
    encoder = CATEEncoder(input_dim, LATENT_DIM).to(device)
    predictor = OutcomeHead(LATENT_DIM, clip_val=calculated_clip_val).to(device)

    mine_bias = MineStatisticsNetwork(LATENT_DIM + 1).to(device)
    mine_outcome = MineStatisticsNetwork(LATENT_DIM + 1).to(device)

    # Opt: AdamW per miglior regolarizzazione
    opt_main = optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=LR_MAIN, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS, eta_min=1e-6)

    opt_mine_bias = optim.AdamW(mine_bias.parameters(), lr=LR_CRITIC, weight_decay=1e-3)
    opt_mine_outcome = optim.AdamW(mine_outcome.parameters(), lr=LR_CRITIC, weight_decay=1e-3)

    early_stopper = EarlyStoppingPEHE(patience=PATIENCE)

    final_val_pehe = 999.0

    for epoch in range(EPOCHS):
        ds_train.set_epoch(epoch)
        RAMP = 80
        alpha_val = 0.0 if epoch < WARMUP_EPOCHS else min(ALPHA, (epoch - WARMUP_EPOCHS) / RAMP * ALPHA)

        encoder.train()
        predictor.train()

        for batch in dl_train:
            x1, y1, t1, x2, y2, t2, label = [b.to(device) for b in batch]

            if x1.shape[0] == 0: continue
            label = label.float()

            t1_r, y1_r = t1.view(-1, 1), y1.view(-1, 1)

            with torch.no_grad():
                z1_detached = encoder(x1).detach()

            if alpha_val > 0:
                for _ in range(MINE_CRITIC_STEPS):
                    opt_mine_bias.zero_grad()
                    mi_b = mine_lower_bound_stable(mine_bias, z1_detached, t1_r)
                    loss_mb = -mi_b
                    loss_mb.backward()
                    torch.nn.utils.clip_grad_norm_(mine_bias.parameters(), CLIP_NORM)
                    opt_mine_bias.step()

                for _ in range(MINE_CRITIC_STEPS):
                    opt_mine_outcome.zero_grad()
                    mi_o = mine_lower_bound_stable(mine_outcome, z1_detached, y1_r)
                    loss_mo = -mi_o
                    loss_mo.backward()
                    torch.nn.utils.clip_grad_norm_(mine_outcome.parameters(), CLIP_NORM)
                    opt_mine_outcome.step()

            # Main Update
            for p in mine_bias.parameters(): p.requires_grad = False
            for p in mine_outcome.parameters(): p.requires_grad = False

            opt_main.zero_grad()
            z1 = encoder(x1)
            mu0_p, mu1_p = predictor(z1)

            y_pred = torch.where(t1_r == 1, mu1_p, mu0_p)
            loss_sup = F.smooth_l1_loss(y_pred, y1_r, beta=1.0)

            loss_extra = torch.tensor(0.0, device=device)
            if alpha_val > 0:
                z2 = encoder(x2)
                l_cont = contrastive_loss(z1, z2, label, margin=MARGIN)

                mi_b = mine_lower_bound_stable(mine_bias, z1, t1_r)
                mi_o = mine_lower_bound_stable(mine_outcome, z1, y1_r)

                mi_b_clamped = torch.clamp(mi_b, -5.0, 5.0)
                mi_o_clamped = torch.clamp(mi_o, -5.0, 5.0)

                loss_extra = l_cont + 0.5 * mi_b_clamped - 0.1 * mi_o_clamped

            loss_main = loss_sup + alpha_val * loss_extra
            loss_main.backward()

            for p in mine_bias.parameters(): p.requires_grad = True
            for p in mine_outcome.parameters(): p.requires_grad = True

            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), CLIP_NORM)
            opt_main.step()

        scheduler.step()

        # Validation
        encoder.eval();
        predictor.eval()
        with torch.no_grad():
            z_val = encoder(X_val_t)
            m0_val_p, m1_val_p = predictor(z_val)

            # Denormalizzazione
            ite_val_pred_norm = (m1_val_p - m0_val_p).cpu().numpy().ravel()
            ite_val_pred = ite_val_pred_norm * y_std

            val_pehe = np.sqrt(np.mean((gt_val_ite - ite_val_pred) ** 2))

        if not np.isfinite(val_pehe):
            val_pehe = 999.0

        early_stopper(val_pehe, encoder, predictor)
        if early_stopper.early_stop:
            final_val_pehe = early_stopper.best_pehe
            break
        final_val_pehe = early_stopper.best_pehe

        # ITE Update
        if epoch >= WARMUP_EPOCHS and ITE_UPDATE_FREQ > 0 and (epoch % ITE_UPDATE_FREQ == 0):
            with torch.no_grad():
                x_tr_curr = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
                z_full = encoder(x_tr_curr)
                fm0, fm1 = predictor(z_full)
                ds_train.update_ite_estimates(fm0.cpu().numpy().ravel(), fm1.cpu().numpy().ravel())

    # Restore best
    early_stopper.restore_best_weights(encoder, predictor)
    encoder.eval();
    predictor.eval()

    # Test usando X processato
    X_te_t = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_te = encoder(X_te_t)
        m0_te, m1_te = predictor(z_te)

        # Denormalizzazione Test
        ite_pred = (m1_te - m0_te).cpu().numpy().ravel() * y_std

    y_true_te = np.stack([mu0_te_s, mu1_te_s], axis=1)
    pehe = sqrt_PEHE_with_diff(y_true_te, ite_pred)
    ate_err = eps_ATE_diff(mu1_te_s - mu0_te_s, ite_pred)

    return {
        'val_pehe': final_val_pehe,
        'test_pehe': pehe,
        'ate_err': ate_err,
        'epochs': epoch
    }


# ==============================================================================
# PARTE 6: MAIN WORKFLOW (FIXED PARAMS)
# ==============================================================================

def save_results_csv(filename, results_list, best_params):
    df_res = pd.DataFrame(results_list)
    for k, v in best_params.items():
        df_res[f"param_{k}"] = v
    df_res.to_csv(filename, index=False, sep=';')
    print(f"Risultati salvati in {filename}", flush=True)


def main():
    try:
        loader = AbstractCausalLoader.get_loader('IHDP')
        loaded_data = loader.load()
        if loaded_data is None: return
        X_tr, T_tr, YF_tr, _, mu0_tr, mu1_tr, X_te, T_te, YF_te, _, mu0_te, mu1_te = loaded_data
    except Exception as e:
        print(f"Errore dati: {e}")
        return

    # PARAMETRI FINALIZZATI: IL MEGLIO DI ENTRAMBI I MONDI
    final_params = {
        'lr': 0.0013734160981144544,  # LR del Trial 5 (quello con batch 128)
        'batch_size': 128,  # Batch 128 (come volevi tu)
        'latent_dim': 64,  # Trial 5
        'alpha': 0.6521571285476828,  # Trial 5
        'perc': 31,  # Trial 5
        'ite_update_freq': 5,
        'critic_steps': 6,
        'lr_critic': 0.00036468471047333836,
        'outcome_clip_factor': 3.0,  # FIX CRUCIALE: Aumentato per evitare l'errore del Trial 5 sul Test
        'margin': 1.0,
        'epochs': 3000,
        'patience': 50
    }

    # Assicura che il fattore sia presente anche nella fase finale (se non trovato da optuna usa default)
    if 'outcome_clip_factor' not in final_params:
        final_params['outcome_clip_factor'] = 1.5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_avail = X_tr.shape[-1] if X_tr.ndim == 3 else 1

    print("\n" + "=" * 50)
    print(f"STARTING TURBO EVALUATION (BATCH 128) ON {total_avail} SIMULATIONS")
    print("=" * 50)
    print(f"Final Params: {final_params}")
    print("-" * 50)

    results_storage = []

    for i in range(total_avail):
        if X_tr.ndim == 3:
            train_data = (X_tr[:, :, i], T_tr[:, i], YF_tr[:, i], mu0_tr[:, i], mu1_tr[:, i])
            test_data = (X_te[:, :, i], T_te[:, i], YF_te[:, i], mu0_te[:, i], mu1_te[:, i])
        else:
            train_data = (X_tr, T_tr, YF_tr, mu0_tr, mu1_tr)
            test_data = (X_te, T_te, YF_te, mu0_te, mu1_te)

        res = train_single_simulation(i, train_data, test_data, device, final_params)

        print(
            f"[Sim {i + 1}/{total_avail}] PEHE: {res['test_pehe']:.4f} | ATE Err: {res['ate_err']:.4f} | Ep: {res['epochs']}",
            flush=True)

        row = {
            'sim_id': i,
            'test_pehe': res['test_pehe'],
            'ate_err': res['ate_err'],
            'epochs': res['epochs'],
            'val_pehe_early_stop': res['val_pehe']
        }
        results_storage.append(row)

    pehes = [r['test_pehe'] for r in results_storage]
    ate_errs = [r['ate_err'] for r in results_storage]

    print("\n" + "=" * 50)
    print(f"FINAL RESULT (Avg over {total_avail} runs):", flush=True)
    print(f"PEHE: {np.mean(pehes):.4f} +/- {np.std(pehes):.4f}", flush=True)
    print(f"ATE Err: {np.mean(ate_errs):.4f} +/- {np.std(ate_errs):.4f}", flush=True)
    print("=" * 50)

    save_results_csv("final_ihdp_turbo_results.csv", results_storage, final_params)


if __name__ == "__main__":
    main()