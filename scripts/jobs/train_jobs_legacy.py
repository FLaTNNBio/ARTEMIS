import os
import copy
import json
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import optuna
except ImportError:
    optuna = None


# ==============================================================================
# LOGGING / DEVICE
# ==============================================================================
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("JOBS_GMI_OPTUNA_ABLATION")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# USER CONFIG
# ==============================================================================
JOBS_PATH = "../../datasets/jobs/jobs_DW_bin.new.10.train.npz"

# Available modes:
#   "optuna"   -> hyperparameter search on full model
#   "final"    -> final evaluation on full model using BEST_PARAMS
#   "ablation" -> ablation study using BEST_PARAMS
EXPERIMENT_MODE = "ablation"

# jobs_DW_bin.new.10.train.npz usually contains 10 replications.
# For final/ablation: 10 = all replications; 1/2 = quick test.
N_SIMS = 10

# For Optuna: use fewer replications per trial to speed up.
OPTUNA_TRIALS = 50
OPTUNA_N_SIMS = 10

# Print training progress every PRINT_EVERY epochs.
PRINT_EVERY = 25

SAVE_RESULTS = True
OUT_DIR = "ablation_outputs_jobs_gmi"
os.makedirs(OUT_DIR, exist_ok=True)

BEST_PARAMS_JSON = os.path.join(OUT_DIR, "jobs_optuna_best_params.json")
BEST_PARAMS_CSV = os.path.join(OUT_DIR, "jobs_optuna_best_params.csv")
OPTUNA_TRIALS_CSV = os.path.join(OUT_DIR, "jobs_optuna_trials.csv")


# ==============================================================================
# INITIAL BEST PARAMS
# ==============================================================================
AUTO_LOAD_BEST_PARAMS = True

BEST_PARAMS = {
    'lr': 0.00043698548003515706,
    'batch_size': 64,
    'latent_dim': 64,
    'alpha': 0.39885537029882545,
    'perc': 35,
    'ite_update_freq': 2,           # was 5 — updates mu_hat more often
    'lr_treat_clf': 3e-05,
    'treat_clf_steps': 1,
    'outcome_clip_factor': 4.0,
    'use_output_clip': True,
    'margin': 1.15935272501416,
    'epochs': 400,
    'patience': 50,                 # was 40 — give more time after warmup
    'lambda_mi_pos': 0.025,
    'mi_ramp_epochs': 80,          # gradual MI activation; avoids early over-regularization
    'mi_start_epoch': 60,           # delayed safe MI for JOBS
    'mi_pos_min_count': 32,
    'warmup_epochs': 15,            # was 46 — contrastive loss must activate early
    'clip_norm': 5.0,
    'main_weight_decay': 0.0012428049792646263,
    'clf_weight_decay': 3.2143995784148505e-05,
    'huber_beta': 1.0,
    'encoder_dropout': 0.17821215957183187,
    'head_dropout': 0.09737601742270623,
    'clf_dropout': 0.25,
    'use_contrastive': True,
    'use_local_mi': True,
    'use_dynamic_update': True,
    'pair_mode': 'dynamic_ite',     # dynamic_ite | static_ite | random | feature_knn | none
    'feature_k': 20,

    # MI application mode:
    #   'local'  -> MI penalty only on positive/effect-homogeneous pairs
    #   'global' -> MI penalty on all latent representations in the pair batch
    'mi_mode': 'local',
}


# ==============================================================================
# METRICS FOR JOBS
# ==============================================================================
def RPol(t: np.ndarray, y: np.ndarray, e: np.ndarray, hat_y: np.ndarray) -> float:
    """
    Jobs policy risk evaluated on the randomized/experimental subset only.

    Deterministic policy:
        pi(x)=1 if hat_y(1)-hat_y(0)>0, else 0.

    If the induced policy has no evaluable treated or control cell inside the
    randomized subset, returns NaN instead of silently creating a biased value.
    """
    t = np.asarray(t).reshape(-1).astype(float)
    y = np.asarray(y).reshape(-1).astype(float)
    e = np.asarray(e).reshape(-1).astype(float)
    hat_y = np.asarray(hat_y)

    exp_mask = (e == 1)
    if exp_mask.sum() == 0:
        return np.nan

    t = t[exp_mask]
    y = y[exp_mask]
    hat_y = hat_y[exp_mask]

    # Deterministic treatment policy. Avoid np.sign, which gives pi=0.5 on ties.
    pi = (hat_y[:, 1] > hat_y[:, 0]).astype(float)

    idx1 = (pi == 1) & (t == 1)
    idx0 = (pi == 0) & (t == 0)

    # The policy value cannot be estimated if one matching cell is empty.
    if idx1.sum() == 0 or idx0.sum() == 0:
        return np.nan

    p1 = float(pi.mean())
    p0 = 1.0 - p1

    v1 = float(y[idx1].mean())
    v0 = float(y[idx0].mean())
    policy_value = p1 * v1 + p0 * v0

    return float(1.0 - policy_value)


def compute_jobs_att_reference(t: np.ndarray, y: np.ndarray, e: np.ndarray) -> float:
    """
    Jobs ATT reference on the randomized component:

        ATT_JOBS = E[Y | T=1, E=1] - E[Y | T=0, E=1].

    This avoids mixing experimental and non-experimental units when computing
    the reference causal effect.
    """
    t = np.asarray(t).reshape(-1).astype(float)
    y = np.asarray(y).reshape(-1).astype(float)
    e = np.asarray(e).reshape(-1).astype(float)

    randomized_treated = (t == 1) & (e == 1)
    randomized_controls = (t == 0) & (e == 1)

    if randomized_treated.sum() == 0 or randomized_controls.sum() == 0:
        return np.nan

    return float(y[randomized_treated].mean() - y[randomized_controls].mean())


def jobs_att_error(t: np.ndarray, hat_y: np.ndarray, jobs_att_ref: float) -> float:
    """
    ATT error:

        | ATT_ref - mean_{i:T_i=1}(hatY_i(1)-hatY_i(0)) |.

    The function can be used on either the test split or the full dataset,
    depending on the supplied arrays.
    """
    t = np.asarray(t).reshape(-1).astype(float)
    hat_y = np.asarray(hat_y)

    treated = (t == 1)
    if treated.sum() == 0 or jobs_att_ref is None or not np.isfinite(jobs_att_ref):
        return np.nan

    hat_tau = hat_y[:, 1] - hat_y[:, 0]
    att_hat = float(hat_tau[treated].mean())
    return float(abs(att_hat - float(jobs_att_ref)))


def jobs_policy_rate(hat_y: np.ndarray) -> float:
    """Fraction of units assigned to treatment by the deterministic policy."""
    hat_y = np.asarray(hat_y)
    return float(np.mean((hat_y[:, 1] - hat_y[:, 0]) > 0))


# ==============================================================================
# EXPERIMENT RUNNER HELPERS USED BY LOADER
# ==============================================================================
def get_num_sims(X, T, Y):
    if X.ndim == 3:
        return X.shape[-1]
    if T.ndim == 2:
        return T.shape[-1]
    if Y.ndim == 2:
        return Y.shape[-1]
    return 1


def slice_sim(X, T, Y, E, sim_idx):
    if X.ndim == 3:
        X_s = X[:, :, sim_idx]
    else:
        X_s = X

    if T.ndim == 2:
        T_s = T[:, sim_idx]
    else:
        T_s = T

    if Y.ndim == 2:
        Y_s = Y[:, sim_idx]
    else:
        Y_s = Y

    if E.ndim == 2:
        E_s = E[:, sim_idx]
    else:
        E_s = E

    return (
        X_s.astype(np.float32),
        T_s.astype(np.float32),
        Y_s.astype(np.float32),
        E_s.astype(np.float32),
    )


# ==============================================================================
# DATA LOADER
# ==============================================================================
class AbstractCausalLoader(ABC):
    @staticmethod
    def get_loader(dataset_name='JOBS'):
        if dataset_name.upper() == 'JOBS':
            return JobsLoader(JOBS_PATH)
        raise ValueError(f"Dataset not supported: {dataset_name}")

    @abstractmethod
    def load(self):
        pass


class JobsLoader(AbstractCausalLoader):
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def _pick_key(data, candidates):
        keys = list(data.keys())

        for c in candidates:
            if c in keys:
                return c

        lower_map = {k.lower(): k for k in keys}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]

        raise KeyError(f"None of {candidates} found. Available keys: {keys}")

    @staticmethod
    def _normalize_shapes(X, T, Y, E):
        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)
        E = np.asarray(E)

        # X supported shapes:
        #   (n, d)       -> one simulation
        #   (n, d, r)    -> r simulations
        #   (r, n, d)    -> converted to (n, d, r)
        if X.ndim == 2:
            pass
        elif X.ndim == 3:
            # Heuristic: if first axis is tiny and second is large, assume (r, n, d)
            if X.shape[0] <= 100 and X.shape[1] > X.shape[0] and X.shape[2] > 1:
                X = np.transpose(X, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported X shape: {X.shape}")

        T = np.squeeze(T)
        Y = np.squeeze(Y)
        E = np.squeeze(E)

        return (
            X.astype(np.float32),
            T.astype(np.float32),
            Y.astype(np.float32),
            E.astype(np.float32),
        )

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Jobs file not found at:\n{self.path}\n"
                "Check that the path is correct or change JOBS_PATH at the top of the script."
            )

        data = np.load(self.path)

        print("[LOAD] JOBS keys and shapes:")
        for k in data.keys():
            print(f"  {k}: {data[k].shape} {data[k].dtype}")

        x_key = self._pick_key(data, ['x', 'X', 'feature', 'features'])
        t_key = self._pick_key(data, ['t', 'T', 'treatment', 'treat'])
        y_key = self._pick_key(data, ['yf', 'y', 'Y', 'outcome'])
        e_key = self._pick_key(data, ['e', 'E', 'experiment', 'experimental'])

        # Diagnostic only. We do NOT use this scalar as an oracle ATE/ATT target.
        jobs_file_ate = None
        if 'ate' in data.keys():
            jobs_file_ate = float(np.asarray(data['ate']).squeeze())
            print(f"[LOAD] Diagnostic scalar stored as 'ate': {jobs_file_ate:.8f}")
            print("[LOAD] This scalar will NOT be used as oracle ATE/ATT.")
            print("[LOAD] Jobs ATT will be computed as E[Y|T=1] - E[Y|T=0,E=1].")

        X, T, Y, E = self._normalize_shapes(
            data[x_key],
            data[t_key],
            data[y_key],
            data[e_key]
        )

        print(f"[LOAD] Selected keys: X='{x_key}', T='{t_key}', Y='{y_key}', E='{e_key}'")
        print(f"[LOAD] Normalized shapes: X={X.shape}, T={T.shape}, Y={Y.shape}, E={E.shape}")

        total_sims = get_num_sims(X, T, Y)
        print("[LOAD] Jobs ATT reference diagnostic by replication:")
        for sim in range(total_sims):
            _, T_s, Y_s, E_s = slice_sim(X, T, Y, E, sim)
            att_ref_s = compute_jobs_att_reference(T_s, Y_s, E_s)

            if jobs_file_ate is not None:
                print(
                    f"  sim={sim:02d} | "
                    f"ATT_jobs={att_ref_s:.6f} | "
                    f"file_ate={jobs_file_ate:.6f} | "
                    f"abs_diff={abs(att_ref_s - jobs_file_ate):.6f}"
                )
            else:
                print(f"  sim={sim:02d} | ATT_jobs={att_ref_s:.6f}")

        return X, T, Y, E, jobs_file_ate


# ==============================================================================
# UTILS
# ==============================================================================
class EarlyStoppingMetric:
    def __init__(self, patience=40, min_delta=1e-6):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_metric = np.inf
        self.early_stop = False
        self.best_encoder_state = None
        self.best_predictor_state = None

    def __call__(self, metric, encoder, predictor):
        if metric < self.best_metric - self.min_delta:
            self.best_metric = float(metric)
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


def infer_num_treatments(t_array: np.ndarray) -> int:
    t_flat = np.asarray(t_array).reshape(-1)
    t_flat = np.round(t_flat).astype(int)
    return int(len(np.unique(t_flat)))


def empirical_entropy_from_labels(t_idx: torch.Tensor, num_treatments: int, eps: float = 1e-8) -> torch.Tensor:
    counts = torch.bincount(t_idx.view(-1).long(), minlength=num_treatments).float()
    probs = counts / counts.sum().clamp_min(1.0)
    return -(probs * torch.log(probs + eps)).sum()


def treatment_log_prob_mean(
        classifier: nn.Module,
        z: torch.Tensor,
        t_idx: torch.Tensor,
        num_treatments: int
) -> torch.Tensor:
    logits = classifier(z)

    if num_treatments == 2:
        t_float = t_idx.view(-1, 1).float()
        return (t_float * F.logsigmoid(logits) + (1.0 - t_float) * F.logsigmoid(-logits)).mean()

    log_probs = F.log_softmax(logits, dim=1)
    return log_probs.gather(1, t_idx.view(-1, 1).long()).mean()


def treatment_classifier_loss(
        classifier: nn.Module,
        z: torch.Tensor,
        t_idx: torch.Tensor,
        num_treatments: int
) -> torch.Tensor:
    logits = classifier(z)

    if num_treatments == 2:
        return F.binary_cross_entropy_with_logits(logits, t_idx.view(-1, 1).float())

    return F.cross_entropy(logits, t_idx.view(-1).long())


def variational_mi_lower_bound(
        classifier: nn.Module,
        z: torch.Tensor,
        t_idx: torch.Tensor,
        num_treatments: int
) -> torch.Tensor:
    return (
        empirical_entropy_from_labels(t_idx.detach(), num_treatments)
        + treatment_log_prob_mean(classifier, z, t_idx, num_treatments)
    )


def contrastive_loss(z1, z2, label, margin=1.0):
    dist_sq = torch.sum((z1 - z2) ** 2, dim=1)
    loss_sim = label * dist_sq
    loss_dissim = (1.0 - label) * torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0) ** 2
    return torch.mean(loss_sim + loss_dissim) / 2.0


def get_continuous_indices(X):
    is_cont = []
    for c in range(X.shape[1]):
        unique_vals = np.unique(X[:, c])
        if len(unique_vals) > 2:
            is_cont.append(c)
    return is_cont


def compute_tau_threshold(mu0_hat, mu1_hat, perc=20, sample=100_000, rng=None) -> float:
    tau = (mu1_hat - mu0_hat).reshape(-1)
    n = tau.size

    if n < 2:
        return 0.1

    rng = np.random.default_rng() if rng is None else rng

    m = min(sample, n)
    idx1 = rng.integers(0, n, size=m)
    idx2 = rng.integers(0, n, size=m)

    diffs = np.abs(tau[idx1] - tau[idx2])
    thr = float(np.percentile(diffs, perc))

    tau_std = max(float(np.std(tau)), 1e-6)
    thr_min = max(0.05 * tau_std, 1e-3)
    thr_max = max(1.0 * tau_std, thr_min)

    if not np.isfinite(thr):
        thr = 0.2 * tau_std

    return float(np.clip(thr, thr_min, thr_max))


def _empty_pair_batch(X, T, Y):
    return (
        np.zeros((0,) + X.shape[1:], dtype=X.dtype),
        np.zeros((0,), dtype=Y.dtype),
        np.zeros((0,), dtype=T.dtype),
        np.zeros((0,) + X.shape[1:], dtype=X.dtype),
        np.zeros((0,), dtype=Y.dtype),
        np.zeros((0,), dtype=T.dtype),
        np.array([], dtype=np.int64),
    )


# ==============================================================================
# PAIRING
# ==============================================================================
def make_pairs_random(X, T, Y, n_pairs, seed=None):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if n < 2:
        return _empty_pair_batch(X, T, Y)

    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n - 1, size=n_pairs)
    idx_b = np.where(idx_b >= idx_a, idx_b + 1, idx_b)

    labels = rng.integers(0, 2, size=n_pairs).astype(np.int64)

    return X[idx_a], Y[idx_a], T[idx_a], X[idx_b], Y[idx_b], T[idx_b], labels


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    rng = np.random.default_rng(seed)
    tau = (mu1_hat - mu0_hat).reshape(-1)
    n = tau.shape[0]

    if n < 2:
        return _empty_pair_batch(X, T, Y)

    n_pairs = int(min(max(1, n_pairs), n))
    half = n_pairs // 2

    used = set()
    sim_pairs = []
    dis_pairs = []

    def add_pair(i, j, label, container):
        if i == j:
            return
        key = (min(i, j), max(i, j))
        if key not in used:
            used.add(key)
            container.append((i, j, label))

    max_attempts = max(50, n_pairs * 10)

    attempts = 0
    while len(sim_pairs) < half and attempts < max_attempts:
        i = int(rng.integers(0, n))
        cand = np.where(np.abs(tau - tau[i]) < thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            add_pair(i, int(rng.choice(cand)), 1, sim_pairs)
        attempts += 1

    attempts = 0
    while len(dis_pairs) < (n_pairs - len(sim_pairs)) and attempts < max_attempts:
        i = int(rng.integers(0, n))
        cand = np.where(np.abs(tau - tau[i]) >= thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            add_pair(i, int(rng.choice(cand)), 0, dis_pairs)
        attempts += 1

    pairs = sim_pairs + dis_pairs

    while len(pairs) < max(1, n_pairs // 2):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        add_pair(i, j, int(np.abs(tau[i] - tau[j]) < thr), pairs)

    if not pairs:
        return _empty_pair_batch(X, T, Y)

    rng.shuffle(pairs)

    idx_a, idx_b, labels = zip(*pairs)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    labels = np.array(labels, dtype=np.int64)

    return X[idx_a], Y[idx_a], T[idx_a], X[idx_b], Y[idx_b], T[idx_b], labels


def make_pairs_feature_knn(X, T, Y, n_pairs, k=20, seed=None):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if n < 2:
        return _empty_pair_batch(X, T, Y)

    n_pairs = int(min(max(1, n_pairs), n))
    idx_a = rng.integers(0, n, size=n_pairs)

    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)

    idx_b = np.zeros(n_pairs, dtype=np.int64)
    labels = np.zeros(n_pairs, dtype=np.int64)

    half = n_pairs // 2

    for p in range(n_pairs):
        i = idx_a[p]
        sorted_idx = np.argsort(d2[i])
        kk = min(k, len(sorted_idx))

        if p < half:
            idx_b[p] = int(rng.choice(sorted_idx[:kk]))
            labels[p] = 1
        else:
            idx_b[p] = int(rng.choice(sorted_idx[-kk:]))
            labels[p] = 0

    return X[idx_a], Y[idx_a], T[idx_a], X[idx_b], Y[idx_b], T[idx_b], labels


class DynamicContrastiveCausalDS(Dataset):
    def __init__(
            self,
            X_all,
            T_all,
            Y_all,
            mu0_hat=None,
            mu1_hat=None,
            bs=256,
            perc=20,
            seed=0,
            pair_mode='dynamic_ite',
            feature_k=20
    ):
        self.X_all = X_all
        self.T_all = T_all.reshape(-1)
        self.Y_all = Y_all.reshape(-1)
        self.bs = int(bs)
        self.perc = float(perc)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.pair_mode = pair_mode
        self.feature_k = feature_k

        self.current_mu0_hat = np.zeros(X_all.shape[0], dtype=np.float32) if mu0_hat is None else mu0_hat
        self.current_mu1_hat = np.zeros(X_all.shape[0], dtype=np.float32) if mu1_hat is None else mu1_hat

        self.update_threshold()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def update_threshold(self):
        self.thr = compute_tau_threshold(
            self.current_mu0_hat,
            self.current_mu1_hat,
            perc=self.perc,
            rng=self.rng
        )

    def update_ite_estimates(self, mu0_hat, mu1_hat):
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

        if self.pair_mode in ['dynamic_ite', 'static_ite']:
            out = make_pairs_from_hat(
                self.X_all,
                self.T_all,
                self.Y_all,
                self.current_mu0_hat,
                self.current_mu1_hat,
                self.thr,
                self.bs,
                seed=seed
            )
        elif self.pair_mode == 'random':
            out = make_pairs_random(
                self.X_all,
                self.T_all,
                self.Y_all,
                self.bs,
                seed=seed
            )
        elif self.pair_mode == 'feature_knn':
            out = make_pairs_feature_knn(
                self.X_all,
                self.T_all,
                self.Y_all,
                self.bs,
                k=self.feature_k,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown pair_mode: {self.pair_mode}")

        x1, y1, t1, x2, y2, t2, lab = out

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
            spectral_norm(nn.Linear(input_dim, 128)),
            nn.GELU(),
            nn.Dropout(dropout),

            spectral_norm(nn.Linear(128, 128)),
            nn.GELU(),
            nn.Dropout(dropout),

            spectral_norm(nn.Linear(128, latent_dim)),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.network(x)


class OutcomeHead(nn.Module):
    def __init__(self, latent_dim, num_treatments=2, use_output_clip=False, clip_val=5.0, dropout=0.1):
        super().__init__()

        self.num_treatments = num_treatments
        self.heads = nn.ModuleList()

        for _ in range(num_treatments):
            layers = [
                spectral_norm(nn.Linear(latent_dim, 64)),
                nn.GELU(),
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(64, 1)),
            ]

            if use_output_clip:
                layers.append(nn.Hardtanh(min_val=-clip_val, max_val=clip_val))

            self.heads.append(nn.Sequential(*layers))

    def forward(self, z):
        return torch.cat([head(z) for head in self.heads], dim=1)


class TreatmentClassifier(nn.Module):
    def __init__(self, latent_dim, num_treatments=2, dropout=0.1):
        super().__init__()

        out_dim = 1 if num_treatments == 2 else num_treatments

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)),
            nn.GELU(),
            nn.Dropout(dropout),

            spectral_norm(nn.Linear(128, 64)),
            nn.GELU(),
            nn.Dropout(dropout),

            spectral_norm(nn.Linear(64, out_dim)),
        )

    def forward(self, z):
        return self.net(z)


# ==============================================================================
# TRAINING HELPERS
# ==============================================================================
def split_train_val_test_jobs(
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        seed: int,
        train_frac: float = 0.56,
        val_frac: float = 0.24,
):
    """
    Jobs split with joint stratification on treatment T and experimental flag E.

    RPol is evaluated only on E=1 units, so random unstratified splits may leave
    too few randomized treated/control units in validation or test. Stratifying
    on the joint cell (T,E) makes the metric substantially more stable.
    """
    X = np.asarray(X)
    T = np.asarray(T).reshape(-1).astype(int)
    E = np.asarray(E).reshape(-1).astype(int)
    n = X.shape[0]

    test_frac = 1.0 - train_frac - val_frac
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1):
        raise ValueError("train_frac, val_frac and test_frac must be in (0,1).")

    strata = T + 2 * E

    # Fallback to random split if a stratum is too small for stratification.
    unique, counts = np.unique(strata, return_counts=True)
    if counts.min() < 3:
        LOGGER.warning(
            "[SPLIT] Joint stratification on (T,E) failed because at least one "
            "cell has <3 samples. Falling back to random split."
        )
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]

    temp_frac = val_frac + test_frac
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_frac,
        random_state=seed,
    )
    train_idx, temp_idx = next(sss1.split(np.zeros(n), strata))

    temp_strata = strata[temp_idx]
    relative_test_frac = test_frac / temp_frac
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_frac,
        random_state=seed + 1000,
    )
    val_rel_idx, test_rel_idx = next(
        sss2.split(np.zeros(len(temp_idx)), temp_strata)
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    return train_idx, val_idx, test_idx


def jobs_split_diagnostics(name: str, T: np.ndarray, E: np.ndarray) -> Dict[str, int]:
    """Return and print split diagnostics for the four (E,T) cells."""
    T = np.asarray(T).reshape(-1).astype(int)
    E = np.asarray(E).reshape(-1).astype(int)
    stats = {
        f'{name}_n': int(len(T)),
        f'{name}_E1': int(np.sum(E == 1)),
        f'{name}_E1_T1': int(np.sum((E == 1) & (T == 1))),
        f'{name}_E1_T0': int(np.sum((E == 1) & (T == 0))),
        f'{name}_E0_T1': int(np.sum((E == 0) & (T == 1))),
        f'{name}_E0_T0': int(np.sum((E == 0) & (T == 0))),
    }
    print(
        f"  [split:{name}] n={stats[f'{name}_n']} | "
        f"E1={stats[f'{name}_E1']} | "
        f"E1_T1={stats[f'{name}_E1_T1']} | "
        f"E1_T0={stats[f'{name}_E1_T0']} | "
        f"E0_T1={stats[f'{name}_E0_T1']} | "
        f"E0_T0={stats[f'{name}_E0_T0']}",
        flush=True,
    )
    return stats


def predict_hat_y(encoder, predictor, X_np, y_mean, y_std, device):
    encoder.eval()
    predictor.eval()

    with torch.no_grad():
        x_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        z = encoder(x_t)
        hat_y_norm = predictor(z).cpu().numpy()

    return hat_y_norm * y_std + y_mean


def train_single_simulation(
        sim_idx: int,
        data_sim: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        device: str,
        hyperparams: Dict[str, Any],
) -> Dict[str, float]:
    torch.manual_seed(sim_idx)
    np.random.seed(sim_idx)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sim_idx)

    X_s, T_s, Y_s, E_s = data_sim

    T_s = np.round(T_s).astype(int).reshape(-1)
    Y_s = Y_s.reshape(-1).astype(np.float32)
    E_s = np.round(E_s).astype(int).reshape(-1)

    num_treatments = infer_num_treatments(T_s)
    if num_treatments != 2:
        raise ValueError(f"Jobs RPol/ATT assumes binary treatment. Found num_treatments={num_treatments}")

    train_idx, val_idx, test_idx = split_train_val_test_jobs(X_s, T_s, E_s, seed=sim_idx)

    full_jobs_att_ref = compute_jobs_att_reference(T_s, Y_s, E_s)

    print(
        f"  [sim={sim_idx:02d}] start | "
        f"n={X_s.shape[0]} | d={X_s.shape[1]} | "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} | "
        f"jobs_ATT_full={full_jobs_att_ref:.6f}",
        flush=True
    )

    X_train = X_s[train_idx].copy()
    X_val = X_s[val_idx].copy()
    X_test = X_s[test_idx].copy()

    T_train = T_s[train_idx]
    T_val = T_s[val_idx]
    T_test = T_s[test_idx]

    E_train = E_s[train_idx]
    E_val = E_s[val_idx]
    E_test = E_s[test_idx]

    Y_train_raw = Y_s[train_idx]
    Y_val_raw = Y_s[val_idx]
    Y_test_raw = Y_s[test_idx]

    split_diag = {}
    split_diag.update(jobs_split_diagnostics("train", T_train, E_train))
    split_diag.update(jobs_split_diagnostics("val", T_val, E_val))
    split_diag.update(jobs_split_diagnostics("test", T_test, E_test))

    cont_indices = get_continuous_indices(X_train)

    if cont_indices:
        x_mean = X_train[:, cont_indices].mean(axis=0, keepdims=True)
        x_std = np.maximum(X_train[:, cont_indices].std(axis=0, keepdims=True), 1e-6)

        X_train[:, cont_indices] = (X_train[:, cont_indices] - x_mean) / x_std
        X_val[:, cont_indices] = (X_val[:, cont_indices] - x_mean) / x_std
        X_test[:, cont_indices] = (X_test[:, cont_indices] - x_mean) / x_std

    y_mean = float(Y_train_raw.mean())
    y_std = float(Y_train_raw.std())
    if y_std < 1e-6:
        y_std = 1.0

    Y_train = ((Y_train_raw - y_mean) / y_std).astype(np.float32)

    BATCH_SIZE = hyperparams.get('batch_size', 128)
    LATENT_DIM = hyperparams.get('latent_dim', 128)
    LR_MAIN = hyperparams.get('lr', 1e-3)
    LR_TREAT_CLF = hyperparams.get('lr_treat_clf', 1e-3)
    EPOCHS = hyperparams.get('epochs', 400)
    PATIENCE = hyperparams.get('patience', 40)

    WARMUP_EPOCHS = hyperparams.get('warmup_epochs', 20)
    ITE_UPDATE_FREQ = hyperparams.get('ite_update_freq', 3)

    ALPHA = hyperparams.get('alpha', 0.5)
    MARGIN = hyperparams.get('margin', 1.0)

    LAMBDA_MI_POS = hyperparams.get('lambda_mi_pos', 0.05)
    MI_START_EPOCH = hyperparams.get('mi_start_epoch', 30)
    MI_RAMP_EPOCHS = hyperparams.get('mi_ramp_epochs', 80)
    POS_MIN_COUNT = hyperparams.get('mi_pos_min_count', 12)
    TREAT_CLF_STEPS = hyperparams.get('treat_clf_steps', 5)

    HUBER_BETA = hyperparams.get('huber_beta', 0.5)
    CLIP_NORM = hyperparams.get('clip_norm', 2.0)

    MAIN_WD = hyperparams.get('main_weight_decay', 1e-3)
    CLF_WD = hyperparams.get('clf_weight_decay', 1e-4)

    USE_OUTPUT_CLIP = hyperparams.get('use_output_clip', False)
    USE_CONTRASTIVE = hyperparams.get('use_contrastive', True)
    USE_LOCAL_MI = hyperparams.get('use_local_mi', True)
    USE_DYNAMIC_UPDATE = hyperparams.get('use_dynamic_update', True)
    PAIR_MODE = hyperparams.get('pair_mode', 'dynamic_ite')
    MI_MODE = hyperparams.get('mi_mode', 'local')  # 'local' or 'global'

    if MI_MODE not in ['local', 'global']:
        raise ValueError(f"Unknown mi_mode={MI_MODE}. Use 'local' or 'global'.")

    max_y_obs_std = float(np.max(np.abs(Y_train)))
    clip_val = max(max_y_obs_std * hyperparams.get('outcome_clip_factor', 4.0), 3.0)

    ds_train = DynamicContrastiveCausalDS(
        X_all=X_train,
        T_all=T_train,
        Y_all=Y_train,
        mu0_hat=None,
        mu1_hat=None,
        bs=BATCH_SIZE,
        perc=hyperparams.get('perc', 22),
        seed=sim_idx,
        pair_mode=PAIR_MODE,
        feature_k=hyperparams.get('feature_k', 20),
    )

    dl_train = DataLoader(ds_train, batch_size=None, shuffle=True)

    encoder = CATEEncoder(
        input_dim=X_s.shape[1],
        latent_dim=LATENT_DIM,
        dropout=hyperparams.get('encoder_dropout', 0.2)
    ).to(device)

    predictor = OutcomeHead(
        latent_dim=LATENT_DIM,
        num_treatments=2,
        use_output_clip=USE_OUTPUT_CLIP,
        clip_val=clip_val,
        dropout=hyperparams.get('head_dropout', 0.05)
    ).to(device)

    treat_clf = TreatmentClassifier(
        latent_dim=LATENT_DIM,
        num_treatments=2,
        dropout=hyperparams.get('clf_dropout', 0.15)
    ).to(device)

    opt_main = optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=LR_MAIN,
        weight_decay=MAIN_WD
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt_main,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    opt_treat_clf = optim.AdamW(
        treat_clf.parameters(),
        lr=LR_TREAT_CLF,
        weight_decay=CLF_WD
    )

    early_stopper = EarlyStoppingMetric(patience=PATIENCE)

    static_pair_initialized = False
    epoch = 0

    # Pre-initialize ITE estimates with a random-weights forward pass so
    # that pair labels are not all "similar" from epoch 0 (which happens
    # when mu0=mu1=0 → tau=0 everywhere → every pair has |Δτ|<thr).
    if PAIR_MODE in ['dynamic_ite', 'static_ite']:
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            hat_y_init = predict_hat_y(encoder, predictor, X_train, 0.0, 1.0, device)
        ds_train.update_ite_estimates(hat_y_init[:, 0].reshape(-1), hat_y_init[:, 1].reshape(-1))
        encoder.train()
        predictor.train()

    for epoch in range(EPOCHS):
        ds_train.set_epoch(epoch)

        # Ramp over 15 epochs after warmup (was 80 — too slow, model was
        # stopping before contrastive ever reached its full weight).
        lambda_ctr = (
            0.0
            if (epoch < WARMUP_EPOCHS or not USE_CONTRASTIVE or PAIR_MODE == 'none')
            else min(ALPHA, ((epoch - WARMUP_EPOCHS) / 15.0) * ALPHA)
        )

        # Safe MI curriculum: on Jobs, the pseudo-ITE signal can be noisy.
        # We therefore delay MI and increase its weight gradually instead of
        # switching it on abruptly.
        lambda_mi = 0.0
        if epoch >= MI_START_EPOCH and USE_LOCAL_MI and PAIR_MODE != 'none':
            mi_progress = min(1.0, (epoch - MI_START_EPOCH) / max(1, MI_RAMP_EPOCHS))
            lambda_mi = LAMBDA_MI_POS * mi_progress

        encoder.train()
        predictor.train()
        treat_clf.train()

        epoch_loss_sup = []
        epoch_loss_ctr = []
        epoch_loss_mi = []
        epoch_pos_rate = []

        for batch in dl_train:
            x1, y1, t1, x2, y2, t2, label = [b.to(device) for b in batch]

            if x1.shape[0] == 0:
                batch_idx = np.random.choice(
                    len(train_idx),
                    size=min(BATCH_SIZE, len(train_idx)),
                    replace=False
                )

                xb = torch.tensor(X_train[batch_idx], dtype=torch.float32, device=device)
                tb = torch.tensor(T_train[batch_idx], dtype=torch.long, device=device)
                yb = torch.tensor(Y_train[batch_idx], dtype=torch.float32, device=device).view(-1, 1)

                opt_main.zero_grad()

                pred_all = predictor(encoder(xb))
                pred_y = pred_all.gather(1, tb.unsqueeze(1))

                loss = F.smooth_l1_loss(pred_y, yb, beta=HUBER_BETA)
                epoch_loss_sup.append(float(loss.detach().cpu()))
                epoch_loss_ctr.append(0.0)
                epoch_loss_mi.append(0.0)
                epoch_pos_rate.append(0.0)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    CLIP_NORM
                )

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
            pos_count = int(pos_mask.sum().item())

            # Decide which latent representations are used by the MI penalty.
            # Local MI: only positive/effect-homogeneous pairs.
            # Global MI: all latent representations in the current pair batch.
            use_mi_this_batch = False
            z_mi_det = None
            t_mi_det = None

            if lambda_mi > 0 and TREAT_CLF_STEPS > 0:
                if MI_MODE == 'local':
                    enough_mi_samples = pos_count >= POS_MIN_COUNT
                    if enough_mi_samples:
                        with torch.no_grad():
                            z_mi_det = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0).detach()
                            t_mi_det = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
                        use_mi_this_batch = True

                elif MI_MODE == 'global':
                    # Each pair batch contains two views of size B. We use both.
                    enough_mi_samples = (2 * x1.shape[0]) >= POS_MIN_COUNT
                    if enough_mi_samples:
                        with torch.no_grad():
                            z_mi_det = torch.cat([z1, z2], dim=0).detach()
                            t_mi_det = torch.cat([t1_idx, t2_idx], dim=0)
                        use_mi_this_batch = True

            if use_mi_this_batch:
                for _ in range(TREAT_CLF_STEPS):
                    opt_treat_clf.zero_grad()

                    clf_loss = treatment_classifier_loss(
                        treat_clf,
                        z_mi_det,
                        t_mi_det,
                        2
                    )

                    clf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(treat_clf.parameters(), CLIP_NORM)
                    opt_treat_clf.step()

            for p in treat_clf.parameters():
                p.requires_grad = False

            opt_main.zero_grad()

            mu1_all = predictor(z1)
            mu2_all = predictor(z2)

            y_pred1 = mu1_all.gather(1, t1_idx.unsqueeze(1))
            y_pred2 = mu2_all.gather(1, t2_idx.unsqueeze(1))

            loss_sup = 0.5 * (
                F.smooth_l1_loss(y_pred1, y1_r, beta=HUBER_BETA)
                + F.smooth_l1_loss(y_pred2, y2_r, beta=HUBER_BETA)
            )

            loss_ctr = (
                contrastive_loss(z1, z2, label, margin=MARGIN)
                if lambda_ctr > 0
                else torch.tensor(0.0, device=device)
            )

            loss_mi = torch.tensor(0.0, device=device)
            if use_mi_this_batch:
                if MI_MODE == 'local':
                    z_mi = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)
                    t_mi = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
                else:  # MI_MODE == 'global'
                    z_mi = torch.cat([z1, z2], dim=0)
                    t_mi = torch.cat([t1_idx, t2_idx], dim=0)

                mi_lb = variational_mi_lower_bound(treat_clf, z_mi, t_mi, 2)

                # Safe MI penalty: penalize only positive estimated treatment information.
                # Negative lower-bound values are treated as estimator noise and ignored.
                loss_mi = torch.clamp(mi_lb, min=0.0, max=1.0)

            loss_main = loss_sup + lambda_ctr * loss_ctr + lambda_mi * loss_mi

            epoch_loss_sup.append(float(loss_sup.detach().cpu()))
            epoch_loss_ctr.append(float(loss_ctr.detach().cpu()))
            epoch_loss_mi.append(float(loss_mi.detach().cpu()))
            epoch_pos_rate.append(float(pos_mask.float().mean().detach().cpu()))

            loss_main.backward()

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()),
                CLIP_NORM
            )

            opt_main.step()

            for p in treat_clf.parameters():
                p.requires_grad = True

        scheduler.step()

        # Validation: on Jobs use RPol as early-stopping criterion.
        hat_y_val = predict_hat_y(
            encoder,
            predictor,
            X_val,
            y_mean,
            y_std,
            device
        )

        val_rpol = RPol(T_val, Y_val_raw, E_val, hat_y_val)
        if not np.isfinite(val_rpol):
            val_rpol = 999.0

        early_stopper(val_rpol, encoder, predictor)

        should_print = (
            epoch == 0
            or (epoch + 1) % PRINT_EVERY == 0
            or epoch == EPOCHS - 1
            or early_stopper.early_stop
        )

        if should_print:
            mean_sup = float(np.mean(epoch_loss_sup)) if epoch_loss_sup else 0.0
            mean_ctr = float(np.mean(epoch_loss_ctr)) if epoch_loss_ctr else 0.0
            mean_mi = float(np.mean(epoch_loss_mi)) if epoch_loss_mi else 0.0
            mean_pos = float(np.mean(epoch_pos_rate)) if epoch_pos_rate else 0.0

            print(
                f"  [sim={sim_idx:02d}] "
                f"ep={epoch + 1:03d}/{EPOCHS} | "
                f"val_RPol={val_rpol:.5f} | "
                f"best={early_stopper.best_metric:.5f} | "
                f"sup={mean_sup:.5f} | "
                f"ctr={mean_ctr:.5f}(lambda={lambda_ctr:.4f}) | "
                f"mi={mean_mi:.5f}(lambda={lambda_mi:.4f}) | "
                f"pos={mean_pos:.2f} | "
                f"pat={early_stopper.counter}/{PATIENCE}",
                flush=True
            )

        if early_stopper.early_stop:
            print(
                f"  [sim={sim_idx:02d}] Early stopping at epoch {epoch + 1}. "
                f"Best val_RPol={early_stopper.best_metric:.5f}",
                flush=True
            )
            break

        should_update = (
            epoch >= WARMUP_EPOCHS
            and ITE_UPDATE_FREQ > 0
            and (epoch % ITE_UPDATE_FREQ == 0)
        )

        if should_update and PAIR_MODE in ['dynamic_ite', 'static_ite']:
            # normalized predictions are enough for pair ranking.
            hat_y_train_norm = predict_hat_y(
                encoder,
                predictor,
                X_train,
                y_mean=0.0,
                y_std=1.0,
                device=device
            )

            new_mu0 = hat_y_train_norm[:, 0].reshape(-1)
            new_mu1 = hat_y_train_norm[:, 1].reshape(-1)

            if PAIR_MODE == 'dynamic_ite' and USE_DYNAMIC_UPDATE:
                ds_train.update_ite_estimates(new_mu0, new_mu1)

            elif PAIR_MODE == 'static_ite' and not static_pair_initialized:
                ds_train.update_ite_estimates(new_mu0, new_mu1)
                static_pair_initialized = True

    # ------------------------------------------------------------------
    # FIX: restore best weights FIRST, then recompute predictions on the
    # correct splits (val with best weights, test with best weights).
    # The old code passed hat_y_val from the last training epoch to RPol
    # and used T_val/Y_val_raw/E_val instead of the test split for
    # test_rpol — both issues are corrected here.
    # ------------------------------------------------------------------
    early_stopper.restore_best_weights(encoder, predictor)

    # Recompute val predictions with best weights (used only for reporting).
    hat_y_val_best = predict_hat_y(encoder, predictor, X_val, y_mean, y_std, device)

    # Test predictions with best weights.
    hat_y_test = predict_hat_y(encoder, predictor, X_test, y_mean, y_std, device)

    # ------------------------------------------------------------------
    # STANDARD PAPER PROTOCOL (Shalit et al. CFRNet/TARNet):
    #
    # RPol      -> test set only (E=1 units in test)
    # ATT_ref   -> FULL simulation dataset (all 2570 units)
    #              Using the test-split ATT ref causes high variance
    #              because only ~40-50 randomized units end up in test.
    #              All published results on Jobs use the full-dataset ref.
    # ATT_hat   -> treated units in the test set only
    # ------------------------------------------------------------------

    # RPol on the held-out test set.
    test_rpol = RPol(T_test, Y_test_raw, E_test, hat_y_test)

    # ATT reference from the FULL randomized component. This is more stable
    # than a test-only ATT reference, while avoiding non-randomized controls.
    test_jobs_att_error = jobs_att_error(T_test, hat_y_test, full_jobs_att_ref)

    # Secondary metrics on the full dataset. Apply the same normalization used
    # during training before predicting on all units.
    X_s_norm = X_s.copy()
    if cont_indices:
        X_s_norm[:, cont_indices] = (X_s_norm[:, cont_indices] - x_mean) / x_std
    hat_y_full = predict_hat_y(encoder, predictor, X_s_norm, y_mean, y_std, device)
    full_rpol = RPol(T_s, Y_s, E_s, hat_y_full)
    full_jobs_att_error = jobs_att_error(T_s, hat_y_full, full_jobs_att_ref)

    pred_policy_rate = jobs_policy_rate(hat_y_test)
    pred_policy_rate_full = jobs_policy_rate(hat_y_full)

    out = {
        'val_rpol': early_stopper.best_metric,
        'test_rpol': test_rpol,
        'full_rpol': full_rpol,
        'test_jobs_att_error': test_jobs_att_error,
        'full_jobs_att_error': full_jobs_att_error,
        'jobs_att_ref_full': full_jobs_att_ref,
        'att_source': 'jobs_randomized_reference_FULL_dataset_E1_only',
        'pred_policy_rate': pred_policy_rate,
        'pred_policy_rate_full': pred_policy_rate_full,
        'epochs': epoch + 1,
    }
    out.update(split_diag)
    return out


# ==============================================================================
# EXPERIMENT RUNNERS
# ==============================================================================
def run_setting(setting_name: str, params: Dict[str, Any], X, T, Y, E, n_sims: int):
    print("\n" + "=" * 80)
    print(f"RUNNING SETTING: {setting_name}")
    print("=" * 80)

    rows = []

    for i in range(n_sims):
        X_s, T_s, Y_s, E_s = slice_sim(X, T, Y, E, i)

        res = train_single_simulation(
            i,
            (X_s, T_s, Y_s, E_s),
            DEVICE,
            params
        )

        row = {
            'setting': setting_name,
            'sim_id': i,
            'test_rpol': res['test_rpol'],
            'full_rpol': res['full_rpol'],
            'test_jobs_att_error': res['test_jobs_att_error'],
            'full_jobs_att_error': res['full_jobs_att_error'],
            'jobs_att_ref_full': res['jobs_att_ref_full'],
            'att_source': res['att_source'],
            'pred_policy_rate': res['pred_policy_rate'],
            'pred_policy_rate_full': res['pred_policy_rate_full'],
            'epochs': res['epochs'],
            'val_rpol_early_stop': res['val_rpol'],
            'train_E1_T1': res.get('train_E1_T1', np.nan),
            'train_E1_T0': res.get('train_E1_T0', np.nan),
            'val_E1_T1': res.get('val_E1_T1', np.nan),
            'val_E1_T0': res.get('val_E1_T0', np.nan),
            'test_E1_T1': res.get('test_E1_T1', np.nan),
            'test_E1_T0': res.get('test_E1_T0', np.nan),
        }

        for k, v in params.items():
            row[f'param_{k}'] = v

        rows.append(row)

        print(
            f"[{setting_name}] [Sim {i + 1}/{n_sims}] "
            f"RPol(test): {res['test_rpol']:.4f} | "
            f"RPol(full): {res['full_rpol']:.4f} | "
            f"ATT_err(test): {res['test_jobs_att_error']:.4f} | "
            f"ATT_err(full): {res['full_jobs_att_error']:.4f} | "
            f"ATT_ref_full: {res['jobs_att_ref_full']:.4f} | "
            f"policy_rate: {res['pred_policy_rate']:.3f} | "
            f"Ep: {res['epochs']}",
            flush=True
        )

    df = pd.DataFrame(rows)

    agg = pd.DataFrame([{
        'setting': setting_name,
        'n_sims': n_sims,
        # PRIMARY: test RPol (standard benchmark metric)
        'mean_rpol_test': df['test_rpol'].mean(),
        'std_rpol_test': df['test_rpol'].std(),
        # SECONDARY: full-dataset RPol (more stable estimate)
        'mean_rpol_full': df['full_rpol'].mean(),
        'std_rpol_full': df['full_rpol'].std(),
        # ATT error uses full-dataset reference (standard protocol)
        'mean_jobs_att_error': df['test_jobs_att_error'].mean(),
        'std_jobs_att_error': df['test_jobs_att_error'].std(),
        'mean_jobs_att_error_full_eval': df['full_jobs_att_error'].mean(),
        'std_jobs_att_error_full_eval': df['full_jobs_att_error'].std(),
        'mean_jobs_att_ref_full': df['jobs_att_ref_full'].mean(),
        'std_jobs_att_ref_full': df['jobs_att_ref_full'].std(),
        'mean_policy_rate': df['pred_policy_rate'].mean(),
        'std_policy_rate': df['pred_policy_rate'].std(),
        'mean_policy_rate_full': df['pred_policy_rate_full'].mean(),
        'std_policy_rate_full': df['pred_policy_rate_full'].std(),
        'mean_test_E1_T1': df['test_E1_T1'].mean(),
        'mean_test_E1_T0': df['test_E1_T0'].mean(),
        'mean_epochs': df['epochs'].mean(),
    }])

    return df, agg


def build_component_ablation_configs(best_params):
    configs = {}

    # Main model: MI is active, but safe/delayed/ramped.
    configs['full_model_safe_mi'] = copy.deepcopy(best_params)

    # Diagnostic: aggressive MI similar to the previous setting.
    # This is useful to show that MI must be stabilized on Jobs.
    p = copy.deepcopy(best_params)
    p['use_local_mi'] = True
    p['lambda_mi_pos'] = 0.1740729025363689
    p['mi_start_epoch'] = 20
    p['mi_ramp_epochs'] = 1
    p['mi_pos_min_count'] = 16
    p['treat_clf_steps'] = 3
    p['lr_treat_clf'] = 7.695468812937624e-05
    configs['full_model_aggressive_mi'] = p

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


def build_pairing_ablation_configs(best_params):
    """
    Pure pairing ablation.

    IMPORTANT: local MI is disabled for all pairing variants so that this block
    isolates the contrastive pairing rule itself. If MI is left active, random
    and feature-kNN variants also receive a treatment-information penalty on
    their own positive pairs, which no longer have the ARTEMIS effect-homogeneous
    interpretation.
    """
    configs = {}

    for mode in ['dynamic_ite', 'static_ite', 'random', 'feature_knn']:
        p = copy.deepcopy(best_params)
        p['pair_mode'] = mode
        p['use_contrastive'] = True
        p['use_local_mi'] = False
        configs[f'pair_{mode}_no_mi'] = p

    p = copy.deepcopy(best_params)
    p['pair_mode'] = 'none'
    p['use_contrastive'] = False
    p['use_local_mi'] = False
    configs['pair_none_supervised'] = p

    return configs


def build_mi_ablation_configs(best_params):
    """
    MI-mode ablation for JOBS.

    These settings isolate how the treatment-information penalty is applied:
      - scheduled local MI: ARTEMIS reference setting;
      - local MI without schedule: MI starts immediately and ramps almost immediately;
      - global MI: treatment-information penalty over the whole pair batch;
      - local MI only: no contrastive loss, but local MI still uses dynamic pair labels;
      - local MI with random pairs: local MI applied to non-effect-informed positive pairs.
    """
    configs = {}

    p = copy.deepcopy(best_params)
    p['use_contrastive'] = True
    p['use_local_mi'] = True
    p['use_dynamic_update'] = True
    p['pair_mode'] = 'dynamic_ite'
    p['mi_mode'] = 'local'
    configs['mi_local_scheduled'] = p

    p = copy.deepcopy(best_params)
    p['use_contrastive'] = True
    p['use_local_mi'] = True
    p['use_dynamic_update'] = True
    p['pair_mode'] = 'dynamic_ite'
    p['mi_mode'] = 'local'
    p['mi_start_epoch'] = 0
    p['mi_ramp_epochs'] = 1
    configs['mi_local_no_schedule'] = p

    p = copy.deepcopy(best_params)
    p['use_contrastive'] = True
    p['use_local_mi'] = True
    p['use_dynamic_update'] = True
    p['pair_mode'] = 'dynamic_ite'
    p['mi_mode'] = 'global'
    configs['mi_global'] = p

    p = copy.deepcopy(best_params)
    p['use_contrastive'] = False
    p['use_local_mi'] = True
    p['use_dynamic_update'] = True
    p['pair_mode'] = 'dynamic_ite'
    p['mi_mode'] = 'local'
    configs['mi_local_only'] = p

    p = copy.deepcopy(best_params)
    p['use_contrastive'] = True
    p['use_local_mi'] = True
    p['use_dynamic_update'] = False
    p['pair_mode'] = 'random'
    p['mi_mode'] = 'local'
    configs['mi_local_random_pairs'] = p

    return configs


def save_experiment_results(experiment_name, df_all, df_agg):
    per_sim_path = os.path.join(OUT_DIR, f"{experiment_name}_per_sim.csv")
    agg_path = os.path.join(OUT_DIR, f"{experiment_name}_aggregate.csv")
    ranking_path = os.path.join(OUT_DIR, f"{experiment_name}_ranking.csv")

    df_all.to_csv(per_sim_path, index=False, sep=';')

    df_agg = (
        df_agg
        .sort_values(by=['mean_rpol_test', 'mean_jobs_att_error'], ascending=[True, True])
        .reset_index(drop=True)
    )

    df_agg.to_csv(agg_path, index=False, sep=';')
    df_agg.to_csv(ranking_path, index=False, sep=';')

    print(f"\nSaved per-sim results to: {per_sim_path}")
    print(f"Saved aggregate results to: {agg_path}")
    print(f"Saved ranking to: {ranking_path}")


def run_experiment_group(experiment_name, configs, X, T, Y, E, n_sims):
    all_rows = []
    all_agg = []

    for setting_name, params in configs.items():
        df, agg = run_setting(setting_name, params, X, T, Y, E, n_sims)
        all_rows.append(df)
        all_agg.append(agg)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_agg = pd.concat(all_agg, ignore_index=True)

    if SAVE_RESULTS:
        save_experiment_results(experiment_name, df_all, df_agg)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {experiment_name}")
    print("=" * 80)
    print(df_agg.sort_values(by=['mean_rpol_test', 'mean_jobs_att_error']).to_string(index=False))

    return df_all, df_agg


# ==============================================================================
# OPTUNA
# ==============================================================================
def suggest_jobs_params(trial):
    """
    Hyperparameter search space for the full model on Jobs.
    """
    params = copy.deepcopy(BEST_PARAMS)

    params.update({
        # Training
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'latent_dim': trial.suggest_categorical('latent_dim', [64, 128, 256]),

        # Regularization
        'main_weight_decay': trial.suggest_float('main_weight_decay', 1e-5, 5e-2, log=True),
        'clf_weight_decay': trial.suggest_float('clf_weight_decay', 1e-6, 1e-3, log=True),
        'encoder_dropout': trial.suggest_float('encoder_dropout', 0.05, 0.35),
        'head_dropout': trial.suggest_float('head_dropout', 0.0, 0.25),
        'clf_dropout': trial.suggest_float('clf_dropout', 0.05, 0.35),

        # Contrastive geometry
        'alpha': trial.suggest_float('alpha', 0.05, 1.0),
        'margin': trial.suggest_float('margin', 0.3, 1.5),
        'perc': trial.suggest_int('perc', 10, 35),

        # Dynamic pairing
        'ite_update_freq': trial.suggest_categorical('ite_update_freq', [1, 2, 3, 5]),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 5, 25),   # was 5-50, now limited

        # Local MI — safe search space for Jobs.
        # Keep MI active, but weak, delayed, and gradually ramped.
        'lambda_mi_pos': trial.suggest_float('lambda_mi_pos', 0.005, 0.06, log=True),
        'mi_start_epoch': trial.suggest_int('mi_start_epoch', 40, 100),
        'mi_ramp_epochs': trial.suggest_categorical('mi_ramp_epochs', [50, 80, 120]),
        'mi_pos_min_count': trial.suggest_categorical('mi_pos_min_count', [24, 32, 48]),
        'lr_treat_clf': trial.suggest_float('lr_treat_clf', 1e-5, 1e-4, log=True),
        'treat_clf_steps': trial.suggest_categorical('treat_clf_steps', [1, 2]),

        # Loss / stability
        'huber_beta': trial.suggest_categorical('huber_beta', [0.25, 0.5, 1.0]),
        'clip_norm': trial.suggest_categorical('clip_norm', [1.0, 2.0, 5.0]),
        'outcome_clip_factor': trial.suggest_categorical('outcome_clip_factor', [3.0, 4.0, 5.0]),
        'use_output_clip': trial.suggest_categorical('use_output_clip', [False, True]),

        # Full model fixed switches
        'use_contrastive': True,
        'use_local_mi': True,
        'use_dynamic_update': True,
        'pair_mode': 'dynamic_ite',
        'feature_k': 20,
        'mi_mode': 'local',

        # Slightly shorter for Optuna speed, but enough for contrastive to activate
        'epochs': 300,
        'patience': 45,
    })

    return params


def optuna_objective(trial, X, T, Y, E, n_sims):
    params = suggest_jobs_params(trial)

    rows = []

    for i in range(n_sims):
        X_s, T_s, Y_s, E_s = slice_sim(X, T, Y, E, i)

        res = train_single_simulation(
            i,
            (X_s, T_s, Y_s, E_s),
            DEVICE,
            params
        )

        rows.append({
            'sim_id': i,
            'val_rpol': res['val_rpol'],
            'test_rpol': res['test_rpol'],
            'full_rpol': res['full_rpol'],
            'test_jobs_att_error': res['test_jobs_att_error'],
            'full_jobs_att_error': res['full_jobs_att_error'],
            'jobs_att_ref_full': res['jobs_att_ref_full'],
            'att_source': res['att_source'],
            'policy_rate': res['pred_policy_rate'],
            'epochs': res['epochs'],
        })

        trial.report(res['val_rpol'], step=i)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    df = pd.DataFrame(rows)

    mean_val_rpol = float(df['val_rpol'].mean())
    mean_test_rpol = float(df['test_rpol'].mean())
    mean_test_jobs_att_error = float(df['test_jobs_att_error'].mean())

    # Methodologically clean objective: validation RPol only.
    objective = mean_val_rpol

    trial.set_user_attr("mean_val_rpol", mean_val_rpol)
    trial.set_user_attr("mean_test_rpol", mean_test_rpol)
    trial.set_user_attr("mean_full_rpol", float(df['full_rpol'].mean()))
    trial.set_user_attr("mean_test_jobs_att_error", mean_test_jobs_att_error)
    trial.set_user_attr("mean_jobs_att_ref_full", float(df['jobs_att_ref_full'].mean()))
    trial.set_user_attr("mean_policy_rate", float(df['policy_rate'].mean()))
    trial.set_user_attr("mean_epochs", float(df['epochs'].mean()))
    trial.set_user_attr("att_source", ",".join(sorted(set(df['att_source'].astype(str)))))

    print(
        f"[OPTUNA Trial {trial.number}] "
        f"objective={objective:.5f} | "
        f"val_RPol={mean_val_rpol:.5f} | "
        f"test_RPol={mean_test_rpol:.5f} | "
        f"Jobs_ATT_err={mean_test_jobs_att_error:.5f}"
    )

    return objective


def run_optuna_search(X, T, Y, E, n_sims):
    if optuna is None:
        raise ImportError("Optuna non è installato. Installa con: pip install optuna")

    print("\n" + "=" * 80)
    print("RUNNING OPTUNA SEARCH ON JOBS")
    print("=" * 80)
    print(f"Trials: {OPTUNA_TRIALS}")
    print(f"Sims per trial: {n_sims}")
    print("Objective: mean validation RPol")

    study = optuna.create_study(
        direction="minimize",
        study_name="jobs_gmi_rpol_optuna",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=1
        )
    )

    study.optimize(
        lambda trial: optuna_objective(trial, X, T, Y, E, n_sims),
        n_trials=OPTUNA_TRIALS
    )

    print("\n" + "=" * 80)
    print("BEST OPTUNA RESULT")
    print("=" * 80)
    print(f"Best value: {study.best_value}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"    '{k}': {repr(v)},")

    best_full_params = copy.deepcopy(BEST_PARAMS)
    best_full_params.update(study.best_params)

    # Force full model switches.
    best_full_params['use_contrastive'] = True
    best_full_params['use_local_mi'] = True
    best_full_params['use_dynamic_update'] = True
    best_full_params['pair_mode'] = 'dynamic_ite'
    best_full_params['feature_k'] = 20
    best_full_params['mi_mode'] = 'local'

    # For final/ablation restore full-budget training.
    best_full_params['epochs'] = 400
    best_full_params['patience'] = 40

    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(best_full_params, f, indent=4)

    pd.DataFrame([best_full_params]).to_csv(BEST_PARAMS_CSV, index=False, sep=';')

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False, sep=';')

    print(f"\nSaved best params JSON to: {BEST_PARAMS_JSON}")
    print(f"Saved best params CSV to:  {BEST_PARAMS_CSV}")
    print(f"Saved all trials to:       {OPTUNA_TRIALS_CSV}")

    print("\nCOPIA QUESTO BLOCCO DENTRO BEST_PARAMS SE NON VUOI USARE AUTO_LOAD_BEST_PARAMS:\n")
    print("BEST_PARAMS = {")
    for k, v in best_full_params.items():
        print(f"    '{k}': {repr(v)},")
    print("}")

    return best_full_params


def load_best_params_if_available(default_params):
    if not AUTO_LOAD_BEST_PARAMS:
        return default_params

    if not os.path.exists(BEST_PARAMS_JSON):
        print(f"[BEST_PARAMS] No saved Optuna params found at: {BEST_PARAMS_JSON}")
        print("[BEST_PARAMS] Using BEST_PARAMS defined in the script.")
        return default_params

    with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    params = copy.deepcopy(default_params)
    params.update(loaded)

    # Safety override: if the loaded JSON has old warmup/patience values that
    # prevent the contrastive from activating, replace them with the fixed ones.
    if params.get('warmup_epochs', 99) > 25:
        print("[BEST_PARAMS] WARNING: loaded warmup_epochs > 25, overriding to 15.")
        params['warmup_epochs'] = 15
    if params.get('patience', 0) < 45:
        print("[BEST_PARAMS] WARNING: loaded patience < 45, overriding to 50.")
        params['patience'] = 50


    # Backward compatibility: old Optuna JSON files do not contain this key.
    if 'mi_ramp_epochs' not in params:
        print("[BEST_PARAMS] WARNING: loaded params missing mi_ramp_epochs, setting to 80.")
        params['mi_ramp_epochs'] = 80

    if 'mi_mode' not in params:
        print("[BEST_PARAMS] WARNING: loaded params missing mi_mode, setting to 'local'.")
        params['mi_mode'] = 'local'

    print(f"[BEST_PARAMS] Loaded Optuna best params from: {BEST_PARAMS_JSON}")
    return params


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    loader = AbstractCausalLoader.get_loader('JOBS')
    X, T, Y, E, JOBS_FILE_ATE = loader.load()

    total_sims = get_num_sims(X, T, Y)

    print(f"Device: {DEVICE}")
    print(f"Total JOBS simulations: {total_sims}")
    print("Primary metrics: RPol ↓, Jobs ATT error ↓.")
    print("Jobs ATT is computed following the TARNet/CFRNet-style protocol:")
    print("ATT_JOBS = E[Y | T=1] - E[Y | T=0, E=1].")
    print(f"EXPERIMENT_MODE = {EXPERIMENT_MODE}")

    if JOBS_FILE_ATE is not None:
        print(f"[JOBS] Diagnostic file scalar 'ate' = {JOBS_FILE_ATE:.8f}")
        print("[JOBS] This scalar is not used as the oracle evaluation target.")

    if EXPERIMENT_MODE == "optuna":
        n_sims = min(OPTUNA_N_SIMS, total_sims)
        run_optuna_search(X, T, Y, E, n_sims)

    elif EXPERIMENT_MODE == "final":
        n_sims = min(N_SIMS, total_sims)
        params = load_best_params_if_available(BEST_PARAMS)

        configs = {
            "full_model": copy.deepcopy(params)
        }

        run_experiment_group(
            "jobs_final_full_model_rpol_jobs_att",
            configs,
            X,
            T,
            Y,
            E,
            n_sims
        )

    elif EXPERIMENT_MODE == "ablation":
        n_sims = min(N_SIMS, total_sims)
        params = load_best_params_if_available(BEST_PARAMS)

        component_configs = build_component_ablation_configs(params)
        pairing_configs = build_pairing_ablation_configs(params)
        mi_configs = build_mi_ablation_configs(params)

        run_experiment_group(
            "jobs_component_ablation_rpol_jobs_att",
            component_configs,
            X,
            T,
            Y,
            E,
            n_sims
        )

        run_experiment_group(
            "jobs_pairing_ablation_rpol_jobs_att",
            pairing_configs,
            X,
            T,
            Y,
            E,
            n_sims
        )

        run_experiment_group(
            "jobs_mi_ablation_rpol_jobs_att",
            mi_configs,
            X,
            T,
            Y,
            E,
            n_sims
        )

    else:
        raise ValueError(
            f"Unknown EXPERIMENT_MODE={EXPERIMENT_MODE}. "
            "Use 'optuna', 'final', or 'ablation'."
        )


if __name__ == "__main__":
    main()