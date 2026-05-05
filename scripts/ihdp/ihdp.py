import os
import os.path
import logging
import zipfile
import requests
import numpy as np
import copy
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("IHDP_GMI_ABLATION")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# USER CONFIG
# ==============================================================================
# Numero di simulazioni per tutti gli esperimenti di ablation.
# Cambia solo questo valore per scalare velocità vs affidabilità:
#   10  → test rapido (~minuti)
#   50  → risultati orientativi (~ore)
#   1000 → full benchmark (tutte le simulazioni disponibili)
N_SIMS = 1000

SAVE_RESULTS = True
OUT_DIR = "ablation_outputs_ihdp_gmi"
os.makedirs(OUT_DIR, exist_ok=True)

BEST_PARAMS = {
    'lr': 0.002706384665268061, 'batch_size': 128, 'latent_dim': 128, 'alpha': 0.5323677007263332, 'perc': 22,
    'ite_update_freq': 3, 'lr_treat_clf': 0.0002874148000031135, 'treat_clf_steps': 5, 'outcome_clip_factor': 4.0,
    'use_output_clip': False, 'margin': 0.7415498862697159, 'epochs': 400, 'patience': 40,
    'lambda_mi_pos': 0.07187797264363245, 'mi_start_epoch': 30, 'mi_pos_min_count': 12, 'ema_beta': 0.8912246135582452,
    'warmup_epochs': 20, 'clip_norm': 2.0, 'main_weight_decay': 0.0030582385526029317,
    'clf_weight_decay': 5.21853520111354e-05, 'huber_beta': 0.5, 'encoder_dropout': 0.2, 'head_dropout': 0.05,
    'clf_dropout': 0.15,

    # nuovi switch per ablation
    'use_contrastive': True,
    'use_local_mi': True,
    'use_dynamic_update': True,
    'pair_mode': 'dynamic_ite',  # dynamic_ite | static_ite | random | feature_knn | none
    'feature_k': 20
}


# ==============================================================================
# DATA LOADER
# ==============================================================================
def download_url(url, save_path, chunk_size=128):
    print(">>> downloading ", url, " into ", save_path, "...")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


class AbstractCausalLoader(ABC):
    def __init__(self):
        self.loaded = False

    @staticmethod
    def get_loader(dataset_name='IHDP'):
        if dataset_name == 'IHDP':
            return IHDPLoader()
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

        if not os.path.exists(path_train):
            if not os.path.exists(path_train_zip):
                download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip", path_train_zip)
            with zipfile.ZipFile(path_train_zip, 'r') as zip_ref:
                zip_ref.extractall(path)

        if not os.path.exists(path_test):
            if not os.path.exists(path_test_zip):
                download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip", path_test_zip)
            with zipfile.ZipFile(path_test_zip, 'r') as zip_ref:
                zip_ref.extractall(path)

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

        return (
            self.X_tr, self.T_tr, self.YF_tr, self.YCF_tr, self.mu_0_tr, self.mu_1_tr,
            self.X_te, self.T_te, self.YF_te, self.YCF_te, self.mu_0_te, self.mu_1_te
        )


# ==============================================================================
# UTILS
# ==============================================================================
class EarlyStoppingPEHE:
    def __init__(self, patience=20, min_delta=1e-6):
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


def sqrt_PEHE_with_diff(y: np.ndarray, hat_tau: np.ndarray) -> float:
    tau = (y[:, 1] - y[:, 0])
    return float(np.sqrt(np.mean((tau - hat_tau) ** 2)))


def eps_ATE_diff(ite: np.ndarray, hat_ite: np.ndarray) -> float:
    return float(np.abs(np.mean(ite) - np.mean(hat_ite)))


def infer_num_treatments(t_array: np.ndarray) -> int:
    t_flat = np.asarray(t_array).reshape(-1)
    if np.issubdtype(t_flat.dtype, np.floating):
        t_flat = np.round(t_flat).astype(int)
    else:
        t_flat = t_flat.astype(int)
    return int(len(np.unique(t_flat)))


def empirical_entropy_from_labels(t_idx: torch.Tensor, num_treatments: int, eps: float = 1e-8) -> torch.Tensor:
    t_idx = t_idx.view(-1).long()
    counts = torch.bincount(t_idx, minlength=num_treatments).float()
    probs = counts / counts.sum().clamp_min(1.0)
    ent = -(probs * torch.log(probs + eps)).sum()
    return ent


def treatment_log_prob_mean(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor,
                            num_treatments: int) -> torch.Tensor:
    logits = classifier(z)
    if num_treatments == 2:
        t_float = t_idx.view(-1, 1).float()
        log_p1 = F.logsigmoid(logits)
        log_p0 = F.logsigmoid(-logits)
        log_prob = t_float * log_p1 + (1.0 - t_float) * log_p0
        return log_prob.mean()
    log_probs = F.log_softmax(logits, dim=1)
    chosen = log_probs.gather(1, t_idx.view(-1, 1).long())
    return chosen.mean()


def treatment_classifier_loss(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor,
                              num_treatments: int) -> torch.Tensor:
    logits = classifier(z)
    if num_treatments == 2:
        return F.binary_cross_entropy_with_logits(logits, t_idx.view(-1, 1).float())
    return F.cross_entropy(logits, t_idx.view(-1).long())


def variational_mi_lower_bound(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor,
                               num_treatments: int) -> torch.Tensor:
    h_t = empirical_entropy_from_labels(t_idx.detach(), num_treatments)
    mean_log_q = treatment_log_prob_mean(classifier, z, t_idx, num_treatments)
    return h_t + mean_log_q


def contrastive_loss(z1, z2, label, margin=1.0):
    dist_sq = torch.sum(torch.pow(z1 - z2, 2), dim=1)
    loss_sim = label * dist_sq
    loss_dissim = (1 - label) * torch.pow(torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0), 2)
    return torch.mean(loss_sim + loss_dissim) / 2


def get_continuous_indices(X):
    is_cont = []
    for c in range(X.shape[1]):
        unique_vals = np.unique(X[:, c])
        if len(unique_vals) > 2:
            is_cont.append(c)
    return is_cont


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


# ==============================================================================
# PAIRING GENERATORS
# ==============================================================================
def make_pairs_random(X, T, Y, n_pairs, seed=None):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 2:
        return _empty_pair_batch(X, T, Y)

    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N - 1, size=n_pairs)
    idx_b = np.where(idx_b >= idx_a, idx_b + 1, idx_b)

    labels = rng.integers(0, 2, size=n_pairs).astype(np.int64)
    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels
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
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    labels = np.array(labels, dtype=np.int64)

    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels
    )


def make_pairs_feature_knn(X, T, Y, n_pairs, k=20, seed=None):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 2:
        return _empty_pair_batch(X, T, Y)

    n_pairs = int(min(max(1, n_pairs), N))
    idx_a = rng.integers(0, N, size=n_pairs)

    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)

    idx_b = np.zeros(n_pairs, dtype=np.int64)
    labels = np.zeros(n_pairs, dtype=np.int64)

    half = n_pairs // 2
    for p in range(n_pairs):
        i = idx_a[p]
        sorted_idx = np.argsort(d2[i])

        if p < half:
            # similar pair from nearest neighbors
            topk = sorted_idx[:min(k, len(sorted_idx))]
            j = int(rng.choice(topk))
            labels[p] = 1
        else:
            # dissimilar pair from far points
            far = sorted_idx[max(1, len(sorted_idx) - min(k, len(sorted_idx))):]
            j = int(rng.choice(far))
            labels[p] = 0

        idx_b[p] = j

    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels
    )


class DynamicContrastiveCausalDS(Dataset):
    def __init__(
            self,
            X_all,
            T_all,
            Y_all,
            mu0_hat,
            mu1_hat,
            bs=256,
            perc=20,
            sample_for_thr_calc=100_000,
            seed=0,
            pair_mode='dynamic_ite',
            feature_k=20
    ):
        self.X_all = X_all
        self.T_all = T_all
        self.Y_all = Y_all
        self.bs = int(bs)
        self.perc = float(perc)
        self.sample_for_thr_calc = int(sample_for_thr_calc)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.epoch = 0
        self.pair_mode = pair_mode
        self.feature_k = feature_k

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
            rng=self.rng
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

        if self.pair_mode == 'none':
            return (
                torch.zeros((0, self.X_all.shape[1]), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0, self.X_all.shape[1]), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        if self.pair_mode == 'dynamic_ite' or self.pair_mode == 'static_ite':
            x1, y1, t1, x2, y2, t2, lab = make_pairs_from_hat(
                self.X_all, self.T_all, self.Y_all,
                self.current_mu0_hat, self.current_mu1_hat,
                self.thr, self.bs, seed=seed
            )
        elif self.pair_mode == 'random':
            x1, y1, t1, x2, y2, t2, lab = make_pairs_random(
                self.X_all, self.T_all, self.Y_all,
                self.bs, seed=seed
            )
        elif self.pair_mode == 'feature_knn':
            x1, y1, t1, x2, y2, t2, lab = make_pairs_feature_knn(
                self.X_all, self.T_all, self.Y_all,
                self.bs, k=self.feature_k, seed=seed
            )
        else:
            raise ValueError(f"Unknown pair_mode: {self.pair_mode}")

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
# MODEL
# ==============================================================================
class CATEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.15):
        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, latent_dim)), nn.LayerNorm(latent_dim)
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
                spectral_norm(nn.Linear(64, 1))
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
        self.num_treatments = num_treatments
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(128, 64)), nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(64, out_dim))
        )

    def forward(self, z):
        return self.net(z)


# ==============================================================================
# TRAIN
# ==============================================================================
def train_single_simulation(
        sim_idx: int,
        data_train: Tuple,
        data_test: Tuple,
        device: str,
        hyperparams: Dict[str, Any]
) -> Dict[str, float]:
    torch.manual_seed(sim_idx)
    np.random.seed(sim_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sim_idx)

    LR_MAIN = hyperparams.get('lr', 1e-3)
    BATCH_SIZE = hyperparams.get('batch_size', 64)
    LATENT_DIM = hyperparams.get('latent_dim', 64)
    ALPHA = hyperparams.get('alpha', 0.5)
    PERC_THR = hyperparams.get('perc', 20)
    ITE_UPDATE_FREQ = hyperparams.get('ite_update_freq', 5)
    LR_TREAT_CLF = hyperparams.get('lr_treat_clf', 1e-3)
    TREAT_CLF_STEPS = hyperparams.get('treat_clf_steps', 3)
    MARGIN = hyperparams.get('margin', 1.0)

    LAMBDA_MI_POS = hyperparams.get('lambda_mi_pos', 0.05)
    MI_START_EPOCH = hyperparams.get('mi_start_epoch', 60)
    POS_MIN_COUNT = hyperparams.get('mi_pos_min_count', 4)

    EPOCHS = hyperparams.get('epochs', 400)
    PATIENCE = hyperparams.get('patience', 40)

    WARMUP_EPOCHS = hyperparams.get('warmup_epochs', 30)
    CLIP_NORM = hyperparams.get('clip_norm', 1.0)
    MAIN_WD = hyperparams.get('main_weight_decay', 1e-2)
    CLF_WD = hyperparams.get('clf_weight_decay', 1e-3)
    HUBER_BETA = hyperparams.get('huber_beta', 1.0)
    USE_OUTPUT_CLIP = hyperparams.get('use_output_clip', True)
    ENCODER_DROPOUT = hyperparams.get('encoder_dropout', 0.15)
    HEAD_DROPOUT = hyperparams.get('head_dropout', 0.1)
    CLF_DROPOUT = hyperparams.get('clf_dropout', 0.1)

    USE_CONTRASTIVE = hyperparams.get('use_contrastive', True)
    USE_LOCAL_MI = hyperparams.get('use_local_mi', True)
    USE_DYNAMIC_UPDATE = hyperparams.get('use_dynamic_update', True)
    PAIR_MODE = hyperparams.get('pair_mode', 'dynamic_ite')
    FEATURE_K = hyperparams.get('feature_k', 20)

    X_tr_s, T_tr_s, Y_tr_s, mu0_tr_s, mu1_tr_s = data_train
    X_te_s, T_te_s, Y_te_s, mu0_te_s, mu1_te_s = data_test

    num_treatments = infer_num_treatments(T_tr_s)

    n_total = X_tr_s.shape[0]
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    rng = np.random.default_rng(sim_idx)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

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
    clip_factor = hyperparams.get('outcome_clip_factor', 3.0)
    calculated_clip_val = max(max_y_obs_std * clip_factor, 3.0)

    # initial ITE estimates for static pairing: use true mu on train only for initialization? NO.
    # Better: keep zeros and freeze, so static_ite means fixed after first refresh from warmup.
    ds_train = DynamicContrastiveCausalDS(
        X_all=X_train_processed,
        T_all=T_tr_s[train_idx],
        Y_all=Y_tr_norm_all[train_idx],
        mu0_hat=None,
        mu1_hat=None,
        bs=BATCH_SIZE,
        perc=PERC_THR,
        seed=sim_idx,
        pair_mode=PAIR_MODE,
        feature_k=FEATURE_K
    )
    dl_train = DataLoader(ds_train, batch_size=None, shuffle=True)

    X_val_t = torch.tensor(X_val_processed, dtype=torch.float32).to(device)
    gt_val_ite = mu1_tr_s[val_idx] - mu0_tr_s[val_idx]

    input_dim = X_tr_s.shape[1]
    encoder = CATEEncoder(input_dim, LATENT_DIM, dropout=ENCODER_DROPOUT).to(device)
    predictor = OutcomeHead(
        LATENT_DIM,
        num_treatments=num_treatments,
        use_output_clip=USE_OUTPUT_CLIP,
        clip_val=calculated_clip_val,
        dropout=HEAD_DROPOUT
    ).to(device)
    treat_clf = TreatmentClassifier(LATENT_DIM, num_treatments=num_treatments, dropout=CLF_DROPOUT).to(device)

    opt_main = optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR_MAIN,
        weight_decay=MAIN_WD
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS, eta_min=1e-6)

    opt_treat_clf = optim.AdamW(
        treat_clf.parameters(),
        lr=LR_TREAT_CLF,
        weight_decay=CLF_WD
    )

    early_stopper = EarlyStoppingPEHE(patience=PATIENCE)

    final_val_pehe = 999.0
    epoch = 0
    static_pair_initialized = False

    for epoch in range(EPOCHS):
        ds_train.set_epoch(epoch)

        RAMP = 80
        lambda_ctr = 0.0 if (epoch < WARMUP_EPOCHS or not USE_CONTRASTIVE or PAIR_MODE == 'none') else min(ALPHA, (
                    epoch - WARMUP_EPOCHS) / RAMP * ALPHA)
        lambda_mi = 0.0 if (epoch < MI_START_EPOCH or not USE_LOCAL_MI or PAIR_MODE == 'none') else LAMBDA_MI_POS

        encoder.train()
        predictor.train()
        treat_clf.train()

        for batch in dl_train:
            x1, y1, t1, x2, y2, t2, label = [b.to(device) for b in batch]

            if x1.shape[0] == 0:
                # Pair mode none -> fallback supervised mini-batch sampled manually
                batch_idx = np.random.choice(len(train_idx), size=min(BATCH_SIZE, len(train_idx)), replace=False)
                xb = torch.tensor(X_train_processed[batch_idx], dtype=torch.float32, device=device)
                tb = torch.tensor(T_tr_s[train_idx][batch_idx], dtype=torch.float32, device=device).view(-1).long()
                yb = torch.tensor(Y_tr_norm_all[train_idx][batch_idx], dtype=torch.float32, device=device).view(-1, 1)

                opt_main.zero_grad()
                zb = encoder(xb)
                mu_all_b = predictor(zb)
                y_pred_b = mu_all_b.gather(1, tb.unsqueeze(1))
                loss_sup = F.smooth_l1_loss(y_pred_b, yb, beta=HUBER_BETA)
                loss_sup.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), CLIP_NORM)
                opt_main.step()
                continue

            label = label.float()
            t1_idx = t1.view(-1).long()
            t2_idx = t2.view(-1).long()
            y1_r = y1.view(-1, 1)
            y2_r = y2.view(-1, 1)

            z1 = encoder(x1)
            z2 = encoder(x2)
            pos_mask = (label == 1)

            if lambda_mi > 0 and pos_mask.sum().item() > POS_MIN_COUNT:
                with torch.no_grad():
                    z_pos_det = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0).detach()
                    t_pos_det = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)

                for _ in range(TREAT_CLF_STEPS):
                    opt_treat_clf.zero_grad()
                    clf_loss = treatment_classifier_loss(treat_clf, z_pos_det, t_pos_det, num_treatments)
                    clf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(treat_clf.parameters(), CLIP_NORM)
                    opt_treat_clf.step()

            for p in treat_clf.parameters():
                p.requires_grad = False

            opt_main.zero_grad()

            mu_all_1 = predictor(z1)
            mu_all_2 = predictor(z2)

            y_pred1 = mu_all_1.gather(1, t1_idx.unsqueeze(1))
            y_pred2 = mu_all_2.gather(1, t2_idx.unsqueeze(1))

            loss_sup = 0.5 * (
                    F.smooth_l1_loss(y_pred1, y1_r, beta=HUBER_BETA) +
                    F.smooth_l1_loss(y_pred2, y2_r, beta=HUBER_BETA)
            )

            loss_ctr = torch.tensor(0.0, device=device)
            if lambda_ctr > 0:
                loss_ctr = contrastive_loss(z1, z2, label, margin=MARGIN)

            loss_mi = torch.tensor(0.0, device=device)
            if lambda_mi > 0 and pos_mask.sum().item() > POS_MIN_COUNT:
                z_pos = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)
                t_pos = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
                mi_lb = variational_mi_lower_bound(treat_clf, z_pos, t_pos, num_treatments)
                loss_mi = torch.clamp(mi_lb, -5.0, 5.0)

            loss_main = loss_sup + lambda_ctr * loss_ctr + lambda_mi * loss_mi
            loss_main.backward()

            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), CLIP_NORM)
            opt_main.step()

            for p in treat_clf.parameters():
                p.requires_grad = True

        scheduler.step()

        encoder.eval()
        predictor.eval()

        with torch.no_grad():
            z_val = encoder(X_val_t)
            mu_val_all = predictor(z_val)
            ite_val_pred_norm = (mu_val_all[:, 1] - mu_val_all[:, 0]).cpu().numpy().ravel()
            ite_val_pred = ite_val_pred_norm * y_std
            val_pehe = np.sqrt(np.mean((gt_val_ite - ite_val_pred) ** 2))

        if not np.isfinite(val_pehe):
            val_pehe = 999.0

        early_stopper(val_pehe, encoder, predictor)
        final_val_pehe = early_stopper.best_pehe
        if early_stopper.early_stop:
            break

        # update dynamic/static ITE estimates for pairing
        should_update = (epoch >= WARMUP_EPOCHS and ITE_UPDATE_FREQ > 0 and (epoch % ITE_UPDATE_FREQ == 0))
        if should_update and PAIR_MODE in ['dynamic_ite', 'static_ite']:
            with torch.no_grad():
                x_tr_curr = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
                z_full = encoder(x_tr_curr)
                mu_full = predictor(z_full)
                new_mu0 = mu_full[:, 0].cpu().numpy().ravel()
                new_mu1 = mu_full[:, 1].cpu().numpy().ravel()

                if PAIR_MODE == 'dynamic_ite' and USE_DYNAMIC_UPDATE:
                    ds_train.update_ite_estimates(new_mu0, new_mu1)

                elif PAIR_MODE == 'static_ite' and not static_pair_initialized:
                    ds_train.update_ite_estimates(new_mu0, new_mu1)
                    static_pair_initialized = True

    early_stopper.restore_best_weights(encoder, predictor)
    encoder.eval()
    predictor.eval()

    X_te_t = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_te = encoder(X_te_t)
        mu_te_all = predictor(z_te)
        ite_pred = (mu_te_all[:, 1] - mu_te_all[:, 0]).cpu().numpy().ravel() * y_std

    y_true_te = np.stack([mu0_te_s, mu1_te_s], axis=1)
    pehe = sqrt_PEHE_with_diff(y_true_te, ite_pred)
    ate_err = eps_ATE_diff(mu1_te_s - mu0_te_s, ite_pred)

    return {
        'val_pehe': final_val_pehe,
        'test_pehe': pehe,
        'ate_err': ate_err,
        'epochs': epoch + 1
    }


# ==============================================================================
# EXPERIMENT RUNNERS
# ==============================================================================
def subset_num_sims(X_tr):
    total = X_tr.shape[-1] if X_tr.ndim == 3 else 1
    return min(N_SIMS, total)


def run_setting(
        setting_name: str,
        params: Dict[str, Any],
        X_tr, T_tr, YF_tr, mu0_tr, mu1_tr,
        X_te, T_te, YF_te, mu0_te, mu1_te,
        n_sims: int
):
    total_avail = X_tr.shape[-1] if X_tr.ndim == 3 else 1
    n_sims = min(n_sims, total_avail)

    print("\n" + "=" * 80)
    print(f"RUNNING SETTING: {setting_name}")
    print("=" * 80)

    results_storage = []
    for i in range(n_sims):
        if X_tr.ndim == 3:
            train_data = (X_tr[:, :, i], T_tr[:, i], YF_tr[:, i], mu0_tr[:, i], mu1_tr[:, i])
            test_data = (X_te[:, :, i], T_te[:, i], YF_te[:, i], mu0_te[:, i], mu1_te[:, i])
        else:
            train_data = (X_tr, T_tr, YF_tr, mu0_tr, mu1_tr)
            test_data = (X_te, T_te, YF_te, mu0_te, mu1_te)

        res = train_single_simulation(i, train_data, test_data, DEVICE, params)
        row = {
            'setting': setting_name,
            'sim_id': i,
            'test_pehe': res['test_pehe'],
            'ate_err': res['ate_err'],
            'epochs': res['epochs'],
            'val_pehe_early_stop': res['val_pehe']
        }
        for k, v in params.items():
            row[f'param_{k}'] = v
        results_storage.append(row)

        print(
            f"[{setting_name}] [Sim {i + 1}/{n_sims}] "
            f"PEHE: {res['test_pehe']:.4f} | ATE: {res['ate_err']:.4f} | Ep: {res['epochs']}",
            flush=True
        )

    df = pd.DataFrame(results_storage)
    agg = pd.DataFrame([{
        'setting': setting_name,
        'n_sims': n_sims,
        'mean_pehe': df['test_pehe'].mean(),
        'std_pehe': df['test_pehe'].std(),
        'mean_ate_err': df['ate_err'].mean(),
        'std_ate_err': df['ate_err'].std(),
        'mean_epochs': df['epochs'].mean()
    }])

    return df, agg


def build_component_ablation_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs = {}

    configs['full_model'] = copy.deepcopy(best_params)

    p = copy.deepcopy(best_params)
    p['use_contrastive'] = False
    p['pair_mode'] = 'none'
    configs['no_contrastive'] = p

    p = copy.deepcopy(best_params)
    p['use_local_mi'] = False
    configs['no_local_mi'] = p

    p = copy.deepcopy(best_params)
    p['use_dynamic_update'] = False
    p['pair_mode'] = 'static_ite'
    configs['no_dynamic_update'] = p

    p = copy.deepcopy(best_params)
    p['use_contrastive'] = False
    p['use_local_mi'] = False
    p['pair_mode'] = 'none'
    configs['supervised_only'] = p

    return configs


def build_pairing_ablation_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs = {}

    p = copy.deepcopy(best_params)
    p['pair_mode'] = 'dynamic_ite'
    configs['pair_dynamic_ite'] = p

    p = copy.deepcopy(best_params)
    p['pair_mode'] = 'static_ite'
    configs['pair_static_ite'] = p

    p = copy.deepcopy(best_params)
    p['pair_mode'] = 'random'
    configs['pair_random'] = p

    p = copy.deepcopy(best_params)
    p['pair_mode'] = 'feature_knn'
    configs['pair_feature_knn'] = p

    p = copy.deepcopy(best_params)
    p['pair_mode'] = 'none'
    p['use_contrastive'] = False
    p['use_local_mi'] = False
    configs['pair_none_supervised'] = p

    return configs


def build_sensitivity_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs = {}
    configs['best_reference'] = copy.deepcopy(best_params)

    sensitivity_grid = {
        'alpha': [0.35, best_params['alpha'], 0.70],
        'margin': [0.50, best_params['margin'], 0.90],
        'perc': [18, best_params['perc'], 26],
        'lambda_mi_pos': [0.03, best_params['lambda_mi_pos'], 0.12],
        'ite_update_freq': [1, best_params['ite_update_freq'], 5],
        'mi_start_epoch': [10, best_params['mi_start_epoch'], 60],
        'warmup_epochs': [10, best_params['warmup_epochs'], 40],
    }

    for param_name, values in sensitivity_grid.items():
        for v in values:
            cfg = copy.deepcopy(best_params)
            cfg[param_name] = v
            name = f"sens_{param_name}_{str(v).replace('.', 'p')}"
            configs[name] = cfg

    return configs


def save_experiment_results(experiment_name: str, df_all: pd.DataFrame, df_agg: pd.DataFrame):
    per_sim_path = os.path.join(OUT_DIR, f"{experiment_name}_per_sim.csv")
    agg_path = os.path.join(OUT_DIR, f"{experiment_name}_aggregate.csv")
    rank_path = os.path.join(OUT_DIR, f"{experiment_name}_ranking.csv")

    df_all.to_csv(per_sim_path, index=False, sep=';')
    df_agg = df_agg.sort_values(by=['mean_pehe', 'mean_ate_err'], ascending=[True, True]).reset_index(drop=True)
    df_agg.to_csv(agg_path, index=False, sep=';')
    df_agg.to_csv(rank_path, index=False, sep=';')

    print(f"\nSaved per-sim results to: {per_sim_path}")
    print(f"Saved aggregate results to: {agg_path}")
    print(f"Saved ranking to: {rank_path}")


def run_experiment_group(
        experiment_name: str,
        configs: Dict[str, Dict[str, Any]],
        X_tr, T_tr, YF_tr, mu0_tr, mu1_tr,
        X_te, T_te, YF_te, mu0_te, mu1_te,
        n_sims: int
):
    all_rows = []
    all_agg = []

    for setting_name, params in configs.items():
        df, agg = run_setting(
            setting_name=setting_name,
            params=params,
            X_tr=X_tr, T_tr=T_tr, YF_tr=YF_tr, mu0_tr=mu0_tr, mu1_tr=mu1_tr,
            X_te=X_te, T_te=T_te, YF_te=YF_te, mu0_te=mu0_te, mu1_te=mu1_te,
            n_sims=n_sims
        )
        all_rows.append(df)
        all_agg.append(agg)

    df_all = pd.concat(all_rows, axis=0, ignore_index=True)
    df_agg = pd.concat(all_agg, axis=0, ignore_index=True)

    save_experiment_results(experiment_name, df_all, df_agg)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {experiment_name}")
    print("=" * 80)
    print(df_agg.sort_values(by=['mean_pehe', 'mean_ate_err']).to_string(index=False))


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    loader = AbstractCausalLoader.get_loader('IHDP')
    loaded_data = loader.load()

    X_tr, T_tr, YF_tr, _, mu0_tr, mu1_tr, X_te, T_te, YF_te, _, mu0_te, mu1_te = loaded_data

    n_sims = subset_num_sims(X_tr)
    print(f"Simulazioni per ablation: {n_sims} / {X_tr.shape[-1] if X_tr.ndim == 3 else 1}")

    component_configs = build_component_ablation_configs(BEST_PARAMS)
    pairing_configs = build_pairing_ablation_configs(BEST_PARAMS)
    sensitivity_configs = build_sensitivity_configs(BEST_PARAMS)

    run_experiment_group(
        experiment_name="component_ablation",
        configs=component_configs,
        X_tr=X_tr, T_tr=T_tr, YF_tr=YF_tr, mu0_tr=mu0_tr, mu1_tr=mu1_tr,
        X_te=X_te, T_te=T_te, YF_te=YF_te, mu0_te=mu0_te, mu1_te=mu1_te,
        n_sims=n_sims
    )

    run_experiment_group(
        experiment_name="pairing_ablation",
        configs=pairing_configs,
        X_tr=X_tr, T_tr=T_tr, YF_tr=YF_tr, mu0_tr=mu0_tr, mu1_tr=mu1_tr,
        X_te=X_te, T_te=T_te, YF_te=YF_te, mu0_te=mu0_te, mu1_te=mu1_te,
        n_sims=n_sims
    )

    run_experiment_group(
        experiment_name="sensitivity_ablation",
        configs=sensitivity_configs,
        X_tr=X_tr, T_tr=T_tr, YF_tr=YF_tr, mu0_tr=mu0_tr, mu1_tr=mu1_tr,
        X_te=X_te, T_te=T_te, YF_te=YF_te, mu0_te=mu0_te, mu1_te=mu1_te,
        n_sims=n_sims
    )


if __name__ == "__main__":
    main()
