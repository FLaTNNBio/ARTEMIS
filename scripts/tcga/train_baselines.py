import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, TensorDataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("TCGA_FAIR_COMPARISON")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# CONFIG
# =============================================================================
CONFIG: Dict[str, Any] = {
    "tcga_path": "../../datasets/tcga/tcga.npz",
    "out_dir": "./outputs_tcga_fair_comparison",
    "n_runs": 5,
    "epochs": 150,
    "patience": 25,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "hidden_dim": 256,
    "latent_dim": 128,
    "dropout": 0.15,
    "curve_grid_size": 11,
    "curve_batch_size": 1024,
    "dose_loss_weight": 0.05,
    "pair_loss_weight": 0.10,
    "margin": 0.60,
    "warmup_epochs": 10,
    "override_keys": {
        "x": "feature",
        "treatment_type": "t",
        "dose": "d",
        "factual_outcome": "y",
        "mu_all": "eval_y",
        "dose_grid": "eval_d",
    }
}


# =============================================================================
# SEED / HELPERS
# =============================================================================
def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_key(available_keys: List[str], *aliases: str, override: Optional[str] = None) -> Optional[str]:
    if override is not None:
        if override in available_keys:
            return override
        raise KeyError(f"Override key '{override}' not found. Available keys: {available_keys}")
    lowered = {k.lower(): k for k in available_keys}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def get_continuous_indices(X: np.ndarray) -> List[int]:
    cont = []
    for c in range(X.shape[1]):
        unique_vals = np.unique(X[:, c])
        if len(unique_vals) > 2:
            cont.append(c)
    return cont


def standardize_train_val_test(X_train, X_val, X_test):
    cont_idx = get_continuous_indices(X_train)
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    if cont_idx:
        mean = X_train[:, cont_idx].mean(axis=0, keepdims=True)
        std = X_train[:, cont_idx].std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        X_train[:, cont_idx] = (X_train[:, cont_idx] - mean) / std
        X_val[:, cont_idx] = (X_val[:, cont_idx] - mean) / std
        X_test[:, cont_idx] = (X_test[:, cont_idx] - mean) / std

    return X_train, X_val, X_test


def standardize_1d_train(y_train, y_val, y_test):
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    std = max(std, 1e-6)
    return (y_train - mean) / std, (y_val - mean) / std, (y_test - mean) / std, mean, std


def _extract_factual_outcome(y_raw: np.ndarray, a_idx: np.ndarray, num_groups: int) -> np.ndarray:
    y_arr = np.asarray(y_raw, dtype=np.float32)
    N = len(a_idx)
    K = num_groups

    if y_arr.ndim == 1 and y_arr.shape[0] == N:
        return y_arr.astype(np.float32)

    if y_arr.ndim == 1 and y_arr.shape[0] == N * K:
        y_arr = y_arr.reshape(N, K)
        return y_arr[np.arange(N), a_idx].astype(np.float32)

    if y_arr.ndim == 2 and y_arr.shape[0] == N and y_arr.shape[1] == K:
        return y_arr[np.arange(N), a_idx].astype(np.float32)

    raise ValueError(
        f"Cannot interpret outcome array with shape {y_arr.shape}. "
        f"Expected [N], [N,K], or flattened [N*K] with N={N}, K={K}."
    )


class EarlyStoppingMetric:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, value, model):
        if value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class DiscreteTCGABundle:
    X: np.ndarray
    t: np.ndarray
    yf: np.ndarray
    mu_all: np.ndarray
    num_treatments: int


@dataclass
class MixedTCGABundle:
    X: np.ndarray
    a: np.ndarray
    d: np.ndarray
    yf: np.ndarray
    mu_all: Optional[np.ndarray] = None
    dose_grid: Optional[np.ndarray] = None
    num_treatments: Optional[int] = None


# =============================================================================
# LOADING
# =============================================================================
def _normalise_discrete_treatments(t_raw: np.ndarray) -> Tuple[np.ndarray, int]:
    T = np.asarray(t_raw).reshape(-1)
    if np.issubdtype(T.dtype, np.floating):
        T = np.round(T).astype(np.int64)
    unique_t = np.unique(T)
    mapping = {old: new for new, old in enumerate(sorted(unique_t))}
    T_norm = np.array([mapping[t] for t in T], dtype=np.int64)
    return T_norm, len(unique_t)


def load_discrete_bundle(npz_path: str, override_keys: Dict[str, Optional[str]]) -> DiscreteTCGABundle:
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)

    key_x = get_best_key(keys, "x", "covariates", "features", "feature", override=override_keys.get("x"))
    key_t = get_best_key(keys, "t", "treatment", "w", override=override_keys.get("treatment_type"))
    key_yf = get_best_key(keys, "yf", "y", "outcome", override=override_keys.get("factual_outcome"))
    key_mu = get_best_key(keys, "mu_all", "mu", "potential_outcomes", "eval_y", override=override_keys.get("mu_all"))

    if key_x is None or key_t is None or key_yf is None or key_mu is None:
        raise KeyError(f"Could not build discrete bundle. Available keys: {keys}")

    X = np.asarray(data[key_x], dtype=np.float32)
    T, num_treatments = _normalise_discrete_treatments(data[key_t])
    yf = np.asarray(data[key_yf], dtype=np.float32)
    mu_all = np.asarray(data[key_mu], dtype=np.float32)

    if yf.ndim == 2 and yf.shape[1] > 1:
        yf = yf[np.arange(len(yf)), T]
    yf = yf.reshape(-1).astype(np.float32)

    if mu_all.ndim != 2:
        raise ValueError(
            f"Discrete comparison expects mu_all with shape [N, K]. Found shape {mu_all.shape}."
        )

    return DiscreteTCGABundle(X=X, t=T, yf=yf, mu_all=mu_all, num_treatments=num_treatments)


def _normalize_observed_groups(a_raw: np.ndarray) -> Tuple[np.ndarray, int]:
    A = np.asarray(a_raw).reshape(-1)
    if np.issubdtype(A.dtype, np.floating):
        if np.allclose(A, np.round(A)):
            A = np.round(A).astype(np.int64)
        else:
            raise ValueError("Observed treatment groups must be discrete. Got non-integer float values.")

    unique_a = np.unique(A)
    if np.array_equal(unique_a, np.array([0, 1, 2, 3])):
        return A.astype(np.int64), 4

    mapping = {old: new for new, old in enumerate(sorted(unique_a))}
    A_norm = np.array([mapping[a] for a in A], dtype=np.int64)
    return A_norm, len(unique_a)


def _normalize_dose_with_control(d_raw: np.ndarray, a_idx: np.ndarray) -> np.ndarray:
    d_arr = np.asarray(d_raw, dtype=np.float32)

    if d_arr.ndim == 1:
        d = d_arr

    elif d_arr.ndim == 2 and d_arr.shape[0] == len(a_idx):
        kd = d_arr.shape[1]
        unique_a = np.unique(a_idx)

        if kd == 3 and set(unique_a.tolist()).issubset({0, 1, 2, 3}):
            d = np.zeros(len(a_idx), dtype=np.float32)
            active_mask = a_idx > 0
            d[active_mask] = d_arr[np.arange(len(a_idx))[active_mask], a_idx[active_mask] - 1]
        elif np.min(a_idx) >= 0 and np.max(a_idx) < kd:
            d = d_arr[np.arange(len(a_idx)), a_idx]
        else:
            raise ValueError(
                f"Dose matrix shape {d_arr.shape} incompatible with observed groups {unique_a.tolist()}."
            )

    else:
        raise ValueError(
            f"Cannot interpret dose array with shape {d_arr.shape}. Expected [N] or [N, K]."
        )

    d = d.astype(np.float32).reshape(-1)
    d_min = float(np.min(d))
    d_max = float(np.max(d))
    if d_max - d_min < 1e-8:
        return np.zeros_like(d, dtype=np.float32)

    d = (d - d_min) / (d_max - d_min)
    return d.astype(np.float32)


def load_mixed_bundle(npz_path: str, override_keys: Dict[str, Optional[str]]) -> MixedTCGABundle:
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)

    key_x = get_best_key(keys, "x", "covariates", "features", "feature", override=override_keys.get("x"))
    key_a = get_best_key(
        keys,
        "a", "t", "treatment_type", "t_type", "treatment_id", "treatment_class",
        override=override_keys.get("treatment_type"),
    )
    key_d = get_best_key(
        keys,
        "d", "dose", "dosage", "treatment_dose", "continuous_treatment",
        override=override_keys.get("dose"),
    )
    key_yf = get_best_key(keys, "yf", "y", "outcome", override=override_keys.get("factual_outcome"))
    key_mu = get_best_key(keys, "mu_all", "mu", "potential_outcomes", "eval_y", override=override_keys.get("mu_all"))
    key_grid = get_best_key(keys, "dose_grid", "grid", "dosage_grid", "eval_d", override=override_keys.get("dose_grid"))

    missing = []
    if key_x is None:
        missing.append("X")
    if key_a is None:
        missing.append("treatment_type")
    if key_d is None:
        missing.append("dose")
    if key_yf is None:
        missing.append("factual_outcome")
    if missing:
        raise KeyError(
            "Mixed comparison cannot be run from this NPZ because these required keys are missing: "
            f"{missing}. Available keys: {keys}."
        )

    X_raw = np.asarray(data[key_x], dtype=np.float32)
    A_raw = np.asarray(data[key_a]).reshape(-1)
    D_raw = np.asarray(data[key_d], dtype=np.float32)
    Y_raw = np.asarray(data[key_yf], dtype=np.float32)

    N = len(A_raw)

    LOGGER.info(
        f"[MIXED LOAD] X shape={X_raw.shape} | A shape={A_raw.shape} | "
        f"D shape={D_raw.shape} | Y shape={Y_raw.shape}"
    )
    vals, counts = np.unique(A_raw, return_counts=True)
    LOGGER.info(f"[MIXED LOAD] treatment raw values/counts: {list(zip(vals.tolist(), counts.tolist()))}")

    if X_raw.shape[0] != N:
        raise ValueError(f"X first dimension {X_raw.shape[0]} does not match len(A_raw)={N}")

    A, K = _normalize_observed_groups(A_raw)
    D = _normalize_dose_with_control(D_raw, A)
    YF = _extract_factual_outcome(Y_raw, A, K)

    MU = None
    dose_grid = None

    if key_mu is not None:
        MU_raw = np.asarray(data[key_mu], dtype=np.float32)
        LOGGER.info(f"[MIXED LOAD] MU raw shape={MU_raw.shape}")

        try:
            if MU_raw.ndim == 2 and MU_raw.shape[0] == N and MU_raw.shape[1] == K:
                MU = MU_raw
            elif MU_raw.ndim == 3 and MU_raw.shape[0] == N:
                if MU_raw.shape[1] == K:
                    MU = MU_raw
                elif MU_raw.shape[2] == K:
                    MU = np.transpose(MU_raw, (0, 2, 1))
                else:
                    LOGGER.warning(
                        f"MU found but cannot align with K={K}. Shape={MU_raw.shape}. Oracle metrics disabled."
                    )
                    MU = None
            else:
                LOGGER.warning(
                    "MU is neither [N,K] nor patient-aligned 3D. Oracle metrics will be disabled."
                )
                MU = None
        except Exception as exc:
            LOGGER.warning(f"Could not parse MU safely: {exc}. Oracle metrics disabled.")
            MU = None

    if key_grid is not None:
        GRID_raw = np.asarray(data[key_grid], dtype=np.float32)
        LOGGER.info(f"[MIXED LOAD] dose_grid raw shape={GRID_raw.shape}")

        try:
            if GRID_raw.ndim == 1:
                dose_grid = GRID_raw.reshape(-1)
            elif GRID_raw.ndim == 2 and 1 in GRID_raw.shape:
                dose_grid = GRID_raw.reshape(-1)
            else:
                LOGGER.warning(
                    f"dose_grid shape {GRID_raw.shape} does not look like a 1D grid. Ignoring it."
                )
                dose_grid = None
        except Exception as exc:
            LOGGER.warning(f"Could not parse dose_grid safely: {exc}. Ignoring it.")
            dose_grid = None
    elif MU is not None and MU.ndim == 3:
        dose_grid = np.linspace(0.0, 1.0, MU.shape[2], dtype=np.float32)

    return MixedTCGABundle(
        X=X_raw.astype(np.float32),
        a=A.astype(np.int64),
        d=D.astype(np.float32),
        yf=YF.astype(np.float32),
        mu_all=MU,
        dose_grid=dose_grid,
        num_treatments=K,
    )


# =============================================================================
# SPLITTING
# =============================================================================
def split_discrete_bundle(bundle: DiscreteTCGABundle, seed: int):
    X, T, YF, MU = bundle.X, bundle.t, bundle.yf, bundle.mu_all
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    tr_idx, tmp_idx = next(sss1.split(X, T))

    X_tr, T_tr, Y_tr, MU_tr = X[tr_idx], T[tr_idx], YF[tr_idx], MU[tr_idx]
    X_tmp, T_tmp, Y_tmp, MU_tmp = X[tmp_idx], T[tmp_idx], YF[tmp_idx], MU[tmp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    val_rel, te_rel = next(sss2.split(X_tmp, T_tmp))

    X_val, T_val, Y_val, MU_val = X_tmp[val_rel], T_tmp[val_rel], Y_tmp[val_rel], MU_tmp[val_rel]
    X_te, T_te, Y_te, MU_te = X_tmp[te_rel], T_tmp[te_rel], Y_tmp[te_rel], MU_tmp[te_rel]

    MU_val_orig = MU_val.copy()
    MU_te_orig = MU_te.copy()

    X_tr, X_val, X_te = standardize_train_val_test(X_tr, X_val, X_te)
    Y_tr, Y_val, Y_te, y_mean, y_std = standardize_1d_train(Y_tr, Y_val, Y_te)
    MU_tr = (MU_tr - y_mean) / y_std
    MU_val = (MU_val - y_mean) / y_std
    MU_te = (MU_te - y_mean) / y_std

    return {
        "X_tr": X_tr, "T_tr": T_tr, "Y_tr": Y_tr, "MU_tr": MU_tr,
        "X_val": X_val, "T_val": T_val, "Y_val": Y_val, "MU_val": MU_val,
        "X_te": X_te, "T_te": T_te, "Y_te": Y_te, "MU_te": MU_te,
        "MU_val_orig": MU_val_orig, "MU_te_orig": MU_te_orig,
        "y_mean": y_mean, "y_std": y_std,
    }


def split_mixed_bundle(bundle: MixedTCGABundle, seed: int):
    X, A, D, YF = bundle.X, bundle.a, bundle.d, bundle.yf

    def _safe_stratified_split(X_in, labels, test_size, random_state):
        labels = np.asarray(labels)
        _, counts = np.unique(labels, return_counts=True)
        if np.min(counts) >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            idx_a, idx_b = next(sss.split(X_in, labels))
            return idx_a, idx_b, True
        return None, None, False

    dose_bins = np.digitize(D, np.linspace(0.0, 1.0, 6), right=False)
    strat_joint = A.astype(str) + "_" + dose_bins.astype(str)

    tr_idx, tmp_idx, ok_joint = _safe_stratified_split(X, strat_joint, test_size=0.4, random_state=seed)
    strat_mode_1 = "A+dose_bin"

    if not ok_joint:
        LOGGER.warning(
            "[MIXED SPLIT] Joint stratification on A+dose_bin has rare strata. Falling back to stratification on A only."
        )
        tr_idx, tmp_idx, ok_a = _safe_stratified_split(X, A, test_size=0.4, random_state=seed)
        strat_mode_1 = "A_only"
        if not ok_a:
            LOGGER.warning("[MIXED SPLIT] Stratification on A also failed. Falling back to plain random split.")
            rng = np.random.default_rng(seed)
            idx = np.arange(len(X))
            rng.shuffle(idx)
            cut = int(round(len(X) * 0.6))
            tr_idx = idx[:cut]
            tmp_idx = idx[cut:]
            strat_mode_1 = "random"

    X_tr, A_tr, D_tr, Y_tr = X[tr_idx], A[tr_idx], D[tr_idx], YF[tr_idx]
    X_tmp, A_tmp, D_tmp, Y_tmp = X[tmp_idx], A[tmp_idx], D[tmp_idx], YF[tmp_idx]

    dose_bins_tmp = np.digitize(D_tmp, np.linspace(0.0, 1.0, 6), right=False)
    strat_joint_tmp = A_tmp.astype(str) + "_" + dose_bins_tmp.astype(str)

    val_rel, te_rel, ok_joint_2 = _safe_stratified_split(X_tmp, strat_joint_tmp, test_size=0.5, random_state=seed + 1)
    strat_mode_2 = "A+dose_bin"

    if not ok_joint_2:
        LOGGER.warning("[MIXED SPLIT] Joint stratification on tmp split failed. Falling back to A only.")
        val_rel, te_rel, ok_a_2 = _safe_stratified_split(X_tmp, A_tmp, test_size=0.5, random_state=seed + 1)
        strat_mode_2 = "A_only"
        if not ok_a_2:
            LOGGER.warning("[MIXED SPLIT] Stratification on A failed for tmp split. Falling back to plain random split.")
            rng = np.random.default_rng(seed + 1)
            idx = np.arange(len(X_tmp))
            rng.shuffle(idx)
            cut = len(X_tmp) // 2
            val_rel = idx[:cut]
            te_rel = idx[cut:]
            strat_mode_2 = "random"

    LOGGER.info(f"[MIXED SPLIT] first split mode: {strat_mode_1}")
    LOGGER.info(f"[MIXED SPLIT] second split mode: {strat_mode_2}")

    X_val, A_val, D_val, Y_val = X_tmp[val_rel], A_tmp[val_rel], D_tmp[val_rel], Y_tmp[val_rel]
    X_te, A_te, D_te, Y_te = X_tmp[te_rel], A_tmp[te_rel], D_tmp[te_rel], Y_tmp[te_rel]

    Y_tr_orig = Y_tr.copy()
    Y_val_orig = Y_val.copy()
    Y_te_orig = Y_te.copy()

    X_tr, X_val, X_te = standardize_train_val_test(X_tr, X_val, X_te)
    Y_tr, Y_val, Y_te, y_mean, y_std = standardize_1d_train(Y_tr, Y_val, Y_te)

    out = {
        "X_tr": X_tr, "A_tr": A_tr, "D_tr": D_tr, "Y_tr": Y_tr,
        "X_val": X_val, "A_val": A_val, "D_val": D_val, "Y_val": Y_val,
        "X_te": X_te, "A_te": A_te, "D_te": D_te, "Y_te": Y_te,
        "Y_tr_orig": Y_tr_orig,
        "Y_val_orig": Y_val_orig,
        "Y_te_orig": Y_te_orig,
        "y_mean": y_mean, "y_std": y_std,
    }

    if bundle.mu_all is not None:
        MU = bundle.mu_all
        if MU.ndim == 2:
            MU_tr = (MU[tr_idx] - y_mean) / y_std
            MU_val = (MU[tmp_idx][val_rel] - y_mean) / y_std
            MU_te = (MU[tmp_idx][te_rel] - y_mean) / y_std

            out["MU_tr"] = MU_tr
            out["MU_val"] = MU_val
            out["MU_te"] = MU_te
            out["MU_val_orig"] = bundle.mu_all[tmp_idx][val_rel]
            out["MU_te_orig"] = bundle.mu_all[tmp_idx][te_rel]

        elif MU.ndim == 3:
            MU_tr = (MU[tr_idx] - y_mean) / y_std
            MU_val = (MU[tmp_idx][val_rel] - y_mean) / y_std
            MU_te = (MU[tmp_idx][te_rel] - y_mean) / y_std

            out["MU_tr"] = MU_tr
            out["MU_val"] = MU_val
            out["MU_te"] = MU_te
            out["MU_val_orig"] = bundle.mu_all[tmp_idx][val_rel]
            out["MU_te_orig"] = bundle.mu_all[tmp_idx][te_rel]
            out["dose_grid"] = bundle.dose_grid

    return out


# =============================================================================
# MODELS
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class DiscreteOutcomeHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z):
        return self.net(z)


class DiscreteMultiHeadNet(nn.Module):
    def __init__(self, input_dim: int, num_treatments: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.heads = nn.ModuleList([DiscreteOutcomeHead(latent_dim, hidden_dim, dropout) for _ in range(num_treatments)])

    def forward(self, x, t=None):
        z = self.encoder(x)
        y_all = torch.cat([head(z) for head in self.heads], dim=1)
        if t is not None:
            y_f = y_all.gather(1, t.unsqueeze(1))
            return z, y_f, y_all
        return z, y_all


class DoseAwareNet(nn.Module):
    def __init__(self, input_dim: int, num_treatments: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.num_treatments = num_treatments
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.t_embed = nn.Embedding(num_treatments, latent_dim)

        self.dose_net = nn.Sequential(
            nn.Linear(1, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim // 2),
            nn.GELU(),
        )

        fusion_dim = latent_dim + latent_dim + latent_dim // 2
        self.outcome_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dose_reg_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, a, d):
        z = self.encoder(x)
        a_emb = self.t_embed(a)
        d_feat = self.dose_net(d.unsqueeze(1))
        y = self.outcome_net(torch.cat([z, a_emb, d_feat], dim=1))
        d_hat = self.dose_reg_head(z)
        return z, y, d_hat

    @torch.no_grad()
    def predict_curve(self, x: torch.Tensor, dose_grid: torch.Tensor) -> torch.Tensor:
        self.eval()
        B = x.shape[0]
        M = dose_grid.shape[0]
        curves = []

        for a in range(self.num_treatments):
            a_vec = torch.full((B,), a, device=x.device, dtype=torch.long)
            preds_a = []
            for m in range(M):
                if a == 0:
                    d_vec = torch.zeros((B,), device=x.device, dtype=torch.float32)
                else:
                    d_vec = torch.full((B,), float(dose_grid[m]), device=x.device, dtype=torch.float32)
                _, y, _ = self.forward(x, a_vec, d_vec)
                preds_a.append(y.squeeze(1))
            preds_a = torch.stack(preds_a, dim=1)
            curves.append(preds_a)

        return torch.stack(curves, dim=1)


# =============================================================================
# PAIRING
# =============================================================================
def flatten_effect_curves(curves: np.ndarray, reference_treatment: int = 0) -> np.ndarray:
    ref = curves[:, [reference_treatment], :]
    pieces = []
    K = curves.shape[1]
    for a in range(K):
        if a == reference_treatment:
            continue
        pieces.append(curves[:, [a], :] - ref)
    eff = np.concatenate(pieces, axis=1)
    return eff.reshape(curves.shape[0], -1).astype(np.float32)


def compute_effect_threshold(effect_vectors: np.ndarray, perc: float = 30.0, sample: int = 50000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    N = effect_vectors.shape[0]
    if N < 2:
        return 0.1
    m = min(sample, N)
    idx1 = rng.integers(0, N, size=m)
    idx2 = rng.integers(0, N, size=m)
    dists = np.linalg.norm(effect_vectors[idx1] - effect_vectors[idx2], axis=1)
    thr = float(np.percentile(dists, perc))
    g = float(np.std(effect_vectors))
    g = max(g, 1e-6)
    return float(np.clip(thr, max(1e-4, 0.05 * g), max(2.0 * g, 1e-4)))


class EffectPairDataset(Dataset):
    def __init__(self, X, A, D, Y, effect_vectors, batch_pairs=128, perc=30.0, seed=0):
        self.X = X
        self.A = A
        self.D = D
        self.Y = Y
        self.effect_vectors = effect_vectors.astype(np.float32)
        self.batch_pairs = int(batch_pairs)
        self.perc = float(perc)
        self.seed = int(seed)
        self.epoch = 0
        self.update_threshold()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def update_effect_vectors(self, effect_vectors: np.ndarray):
        self.effect_vectors = effect_vectors.astype(np.float32)
        self.update_threshold()

    def update_threshold(self):
        self.thr = compute_effect_threshold(self.effect_vectors, perc=self.perc, seed=self.seed + self.epoch)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_pairs))

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + 10007 * self.epoch + 7919 * int(idx))
        N = len(self.X)
        used = set()
        pairs = []
        target = self.batch_pairs
        half = target // 2

        def add(i, j):
            if i == j:
                return False
            key = (min(i, j), max(i, j))
            if key in used:
                return False
            used.add(key)
            d = np.linalg.norm(self.effect_vectors[i] - self.effect_vectors[j])
            lab = 1 if d < self.thr else 0
            pairs.append((i, j, lab))
            return True

        tries = 0
        while sum(p[2] for p in pairs) < half and tries < 20 * target:
            i = int(rng.integers(0, N))
            dists = np.linalg.norm(self.effect_vectors - self.effect_vectors[i], axis=1)
            cand = np.where((dists < self.thr) & (np.arange(N) != i))[0]
            if len(cand) > 0:
                j = int(rng.choice(cand))
                add(i, j)
            tries += 1

        tries = 0
        while len(pairs) < target and tries < 20 * target:
            i = int(rng.integers(0, N))
            dists = np.linalg.norm(self.effect_vectors - self.effect_vectors[i], axis=1)
            cand = np.where((dists >= self.thr) & (np.arange(N) != i))[0]
            if len(cand) > 0:
                j = int(rng.choice(cand))
                add(i, j)
            tries += 1

        if len(pairs) == 0:
            i = np.array([], dtype=np.int64)
            empty_x = np.zeros((0, self.X.shape[1]), dtype=np.float32)
            return tuple(torch.tensor(v) for v in [
                empty_x, i, i.astype(np.float32), np.zeros((0,), dtype=np.float32),
                empty_x, i, i.astype(np.float32), np.zeros((0,), dtype=np.float32), i
            ])

        ia = np.array([p[0] for p in pairs], dtype=np.int64)
        ib = np.array([p[1] for p in pairs], dtype=np.int64)
        lab = np.array([p[2] for p in pairs], dtype=np.int64)

        return tuple(torch.tensor(v) for v in [
            self.X[ia], self.A[ia], self.D[ia], self.Y[ia],
            self.X[ib], self.A[ib], self.D[ib], self.Y[ib],
            lab,
        ])


def contrastive_loss_pairs(z1, z2, label, margin=1.0):
    label = label.float()
    dist_sq = torch.sum((z1 - z2) ** 2, dim=1)
    loss_sim = label * dist_sq
    loss_dis = (1.0 - label) * torch.pow(torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0), 2)
    return torch.mean(loss_sim + loss_dis) / 2.0


# =============================================================================
# METRICS
# =============================================================================
def compute_discrete_metrics(y_pred_all: torch.Tensor, mu_all: torch.Tensor) -> Dict[str, float]:
    mise = F.mse_loss(y_pred_all, mu_all).item()
    mate_errors = []
    K = y_pred_all.shape[1]
    for i in range(K):
        for j in range(i + 1, K):
            pred_eff = y_pred_all[:, i] - y_pred_all[:, j]
            true_eff = mu_all[:, i] - mu_all[:, j]
            mate_errors.append(torch.abs(pred_eff - true_eff).mean().item())
    mate = float(np.mean(mate_errors)) if mate_errors else 0.0
    return {"mise": mise, "sqrt_mise": float(np.sqrt(mise)), "mate": mate}


def compute_curve_metrics(curves_pred: torch.Tensor, curves_true: torch.Tensor) -> Dict[str, float]:
    mise = F.mse_loss(curves_pred, curves_true).item()
    effect_pred = curves_pred[:, 1:, :] - curves_pred[:, [0], :]
    effect_true = curves_true[:, 1:, :] - curves_true[:, [0], :]
    mate_curve = torch.mean(torch.abs(effect_pred - effect_true)).item()
    return {"curve_mise": mise, "curve_sqrt_mise": float(np.sqrt(mise)), "curve_mate": float(mate_curve)}


def compute_group_metrics(y_pred_all: torch.Tensor, mu_all: torch.Tensor) -> Dict[str, float]:
    mise = F.mse_loss(y_pred_all, mu_all).item()
    mate_errors = []
    K = y_pred_all.shape[1]
    for i in range(K):
        for j in range(i + 1, K):
            pred_eff = y_pred_all[:, i] - y_pred_all[:, j]
            true_eff = mu_all[:, i] - mu_all[:, j]
            mate_errors.append(torch.abs(pred_eff - true_eff).mean().item())
    mate = float(np.mean(mate_errors)) if mate_errors else 0.0
    return {"mise": mise, "sqrt_mise": float(np.sqrt(mise)), "mate": mate}


@torch.no_grad()
def predict_group_outcomes(model: DoseAwareNet, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    B = x.shape[0]
    preds = []
    for a in range(model.num_treatments):
        a_vec = torch.full((B,), a, device=x.device, dtype=torch.long)
        if a == 0:
            d_vec = torch.zeros((B,), device=x.device, dtype=torch.float32)
        else:
            d_vec = torch.ones((B,), device=x.device, dtype=torch.float32)
        _, y, _ = model(x, a_vec, d_vec)
        preds.append(y.squeeze(1))
    return torch.stack(preds, dim=1)


# =============================================================================
# TRAINING: DISCRETE
# =============================================================================
def make_discrete_loader(X, T, Y, MU, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(T, dtype=torch.long),
        torch.tensor(Y, dtype=torch.float32).unsqueeze(1),
        torch.tensor(MU, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_discrete_model(split: Dict[str, Any], num_treatments: int, cfg: Dict[str, Any], seed: int, model_name: str):
    set_all_seeds(seed)
    model = DiscreteMultiHeadNet(
        input_dim=split["X_tr"].shape[1],
        num_treatments=num_treatments,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    early = EarlyStoppingMetric(patience=cfg["patience"])

    tr_loader = make_discrete_loader(split["X_tr"], split["T_tr"], split["Y_tr"], split["MU_tr"], cfg["batch_size"], True)
    val_loader = make_discrete_loader(split["X_val"], split["T_val"], split["Y_val"], split["MU_val"], cfg["batch_size"], False)
    te_loader = make_discrete_loader(split["X_te"], split["T_te"], split["Y_te"], split["MU_te"], cfg["batch_size"], False)

    for epoch in range(cfg["epochs"]):
        model.train()
        for xb, tb, yb, _ in tr_loader:
            xb = xb.to(DEVICE)
            tb = tb.to(DEVICE)
            yb = yb.to(DEVICE)
            _, y_f, _ = model(xb, tb)
            loss = F.mse_loss(y_f, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _, _, _ in val_loader:
                xb = xb.to(DEVICE)
                _, y_all = model(xb)
                preds.append(y_all)
        preds = torch.cat(preds, dim=0)
        preds_denorm = preds * split["y_std"] + split["y_mean"]
        mu_val_orig = torch.tensor(split["MU_val_orig"], dtype=torch.float32, device=preds.device)
        val_metrics = compute_discrete_metrics(preds_denorm, mu_val_orig)
        early(val_metrics["sqrt_mise"], model)
        if early.early_stop:
            break

    early.restore(model)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _, _, _ in te_loader:
            xb = xb.to(DEVICE)
            _, y_all = model(xb)
            preds.append(y_all)
    preds = torch.cat(preds, dim=0)
    preds_denorm = preds * split["y_std"] + split["y_mean"]
    mu_te_orig = torch.tensor(split["MU_te_orig"], dtype=torch.float32, device=preds.device)
    test_metrics = compute_discrete_metrics(preds_denorm, mu_te_orig)
    return {"model": model_name, "epochs": epoch + 1, **test_metrics, "best_val_sqrt_mise": early.best_value}


# =============================================================================
# TRAINING: MIXED
# =============================================================================
def make_mixed_loader(X, A, D, Y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(A, dtype=torch.long),
        torch.tensor(D, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32).unsqueeze(1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_mixed_curve_model(model: DoseAwareNet, split: Dict[str, Any], part: str, batch_size: int):
    X = split[f"X_{part}"]
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    dose_grid = split.get("dose_grid")
    if dose_grid is None:
        raise ValueError("Oracle curve evaluation requested but dose_grid is missing.")
    dose_grid_t = torch.tensor(dose_grid, dtype=torch.float32, device=DEVICE)

    curves = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(DEVICE)
            curves_b = model.predict_curve(xb, dose_grid_t)
            curves.append(curves_b)

    curves = torch.cat(curves, dim=0)
    curves_denorm = curves * split["y_std"] + split["y_mean"]
    true_curves = torch.tensor(split[f"MU_{part}_orig"], dtype=torch.float32, device=DEVICE)
    return compute_curve_metrics(curves_denorm, true_curves)


def train_mixed_model(split: Dict[str, Any], num_treatments: int, cfg: Dict[str, Any], seed: int, model_name: str, use_contrastive: bool):
    set_all_seeds(seed)
    model = DoseAwareNet(
        input_dim=split["X_tr"].shape[1],
        num_treatments=num_treatments,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    early = EarlyStoppingMetric(patience=cfg["patience"])

    tr_loader = make_mixed_loader(split["X_tr"], split["A_tr"], split["D_tr"], split["Y_tr"], cfg["batch_size"], True)
    val_loader = make_mixed_loader(split["X_val"], split["A_val"], split["D_val"], split["Y_val"], cfg["batch_size"], False)

    pair_loader = None
    pair_ds = None
    if use_contrastive:
        if "MU_tr" in split and split["MU_tr"].ndim == 3:
            effect_vectors = flatten_effect_curves(split["MU_tr"])
        else:
            effect_vectors = np.zeros((len(split["X_tr"]), (num_treatments - 1) * cfg["curve_grid_size"]), dtype=np.float32)
        pair_ds = EffectPairDataset(split["X_tr"], split["A_tr"], split["D_tr"], split["Y_tr"], effect_vectors, batch_pairs=cfg["batch_size"], seed=seed)
        pair_loader = DataLoader(pair_ds, batch_size=None, shuffle=True)

    for epoch in range(cfg["epochs"]):
        model.train()

        if use_contrastive and epoch >= cfg["warmup_epochs"] and pair_ds is not None and epoch % 5 == 0:
            dose_grid = np.linspace(0.0, 1.0, cfg["curve_grid_size"], dtype=np.float32)
            loader_tmp = DataLoader(torch.tensor(split["X_tr"], dtype=torch.float32), batch_size=cfg["curve_batch_size"], shuffle=False)
            curves = []
            with torch.no_grad():
                for xb in loader_tmp:
                    xb = xb.to(DEVICE)
                    curves_b = model.predict_curve(xb, torch.tensor(dose_grid, dtype=torch.float32, device=DEVICE))
                    curves.append(curves_b.cpu().numpy())
            curves = np.concatenate(curves, axis=0)
            pair_ds.update_effect_vectors(flatten_effect_curves(curves))
            pair_ds.set_epoch(epoch)

        for xb, ab, db, yb in tr_loader:
            xb = xb.to(DEVICE)
            ab = ab.to(DEVICE)
            db = db.to(DEVICE)
            yb = yb.to(DEVICE)

            _, y_hat, d_hat = model(xb, ab, db)
            loss_y = F.mse_loss(y_hat, yb)

            active_mask = (ab > 0)
            if torch.any(active_mask):
                loss_d = F.mse_loss(d_hat[active_mask], db[active_mask].unsqueeze(1))
            else:
                loss_d = torch.tensor(0.0, device=DEVICE)

            loss = loss_y + cfg["dose_loss_weight"] * loss_d
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if use_contrastive and pair_loader is not None and epoch >= cfg["warmup_epochs"]:
            model.train()
            for x1, a1, d1, y1, x2, a2, d2, y2, lab in pair_loader:
                if x1.shape[0] == 0:
                    continue

                x1 = x1.to(DEVICE)
                a1 = a1.to(DEVICE)
                d1 = d1.to(DEVICE)
                y1 = y1.to(DEVICE).unsqueeze(1)
                x2 = x2.to(DEVICE)
                a2 = a2.to(DEVICE)
                d2 = d2.to(DEVICE)
                y2 = y2.to(DEVICE).unsqueeze(1)
                lab = lab.to(DEVICE).float()

                z1, yhat1, _ = model(x1, a1, d1)
                z2, yhat2, _ = model(x2, a2, d2)
                loss_y = 0.5 * (F.mse_loss(yhat1, y1) + F.mse_loss(yhat2, y2))
                loss_ctr = contrastive_loss_pairs(z1, z2, lab, margin=cfg["margin"])
                loss = loss_y + cfg["pair_loss_weight"] * loss_ctr

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, ab, db, yb in val_loader:
                xb = xb.to(DEVICE)
                ab = ab.to(DEVICE)
                db = db.to(DEVICE)
                yb = yb.to(DEVICE)
                _, y_hat, _ = model(xb, ab, db)
                val_losses.append(F.mse_loss(y_hat, yb).item())

        val_loss = float(np.mean(val_losses))
        early(val_loss, model)
        if early.early_stop:
            break

    early.restore(model)

    result = {"model": model_name, "epochs": epoch + 1, "best_val_loss": early.best_value}

    te_loader = make_mixed_loader(split["X_te"], split["A_te"], split["D_te"], split["Y_te"], cfg["batch_size"], False)
    model.eval()
    test_losses_norm = []
    preds_norm = []

    with torch.no_grad():
        for xb, ab, db, yb in te_loader:
            xb = xb.to(DEVICE)
            ab = ab.to(DEVICE)
            db = db.to(DEVICE)
            yb = yb.to(DEVICE)
            _, y_hat, _ = model(xb, ab, db)
            test_losses_norm.append(F.mse_loss(y_hat, yb).item())
            preds_norm.append(y_hat)

    result["test_factual_rmse_norm"] = float(np.sqrt(np.mean(test_losses_norm)))

    preds_norm = torch.cat(preds_norm, dim=0).squeeze(1)
    preds_orig = preds_norm * split["y_std"] + split["y_mean"]
    y_te_orig = torch.tensor(split["Y_te_orig"], dtype=torch.float32, device=preds_orig.device)
    test_mise = F.mse_loss(preds_orig, y_te_orig).item()
    result["test_mise"] = test_mise
    result["test_rmse"] = float(np.sqrt(test_mise))

    if "MU_te_orig" in split and split["MU_te_orig"] is not None:
        loader_x = DataLoader(torch.tensor(split["X_te"], dtype=torch.float32), batch_size=cfg["curve_batch_size"], shuffle=False)
        preds_all = []
        with torch.no_grad():
            for xb in loader_x:
                xb = xb.to(DEVICE)
                y_all = predict_group_outcomes(model, xb)
                preds_all.append(y_all)

        preds_all = torch.cat(preds_all, dim=0)
        preds_all_denorm = preds_all * split["y_std"] + split["y_mean"]
        mu_te_orig = torch.tensor(split["MU_te_orig"], dtype=torch.float32, device=DEVICE)

        if mu_te_orig.ndim == 2:
            result.update(compute_group_metrics(preds_all_denorm, mu_te_orig))
        elif mu_te_orig.ndim == 3 and "dose_grid" in split and split["dose_grid"] is not None:
            try:
                result.update(evaluate_mixed_curve_model(model, split, "te", cfg["curve_batch_size"]))
            except Exception as exc:
                LOGGER.warning(f"Curve evaluation skipped for {model_name}: {exc}")
                result["curve_sqrt_mise"] = np.nan
                result["curve_mate"] = np.nan
        else:
            result["mise"] = np.nan
            result["sqrt_mise"] = np.nan
            result["mate"] = np.nan
    else:
        result["mise"] = np.nan
        result["sqrt_mise"] = np.nan
        result["mate"] = np.nan

    return result


# =============================================================================
# RUNNERS
# =============================================================================
def run_discrete_comparison(bundle: DiscreteTCGABundle, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run in range(cfg["n_runs"]):
        seed = 100 + run
        split = split_discrete_bundle(bundle, seed)
        res = train_discrete_model(split, bundle.num_treatments, cfg, seed, model_name="discrete_multihead_baseline")
        res["run"] = run
        rows.append(res)
        LOGGER.info(f"[DISCRETE] run={run} sqrt_mise={res['sqrt_mise']:.4f} mate={res['mate']:.4f}")
    return pd.DataFrame(rows)


def run_mixed_comparison(bundle: MixedTCGABundle, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run in range(cfg["n_runs"]):
        seed = 100 + run
        split = split_mixed_bundle(bundle, seed)

        res1 = train_mixed_model(split, bundle.num_treatments, cfg, seed, model_name="dose_aware_baseline", use_contrastive=False)
        res1["run"] = run
        rows.append(res1)
        LOGGER.info(
            f"[MIXED][baseline] run={run} factual_rmse={res1.get('test_rmse', np.nan):.4f} "
            f"sqrt_mise={res1.get('sqrt_mise', np.nan):.4f} mate={res1.get('mate', np.nan):.4f}"
        )

        res2 = train_mixed_model(split, bundle.num_treatments, cfg, seed, model_name="dose_aware_hermes", use_contrastive=True)
        res2["run"] = run
        rows.append(res2)
        LOGGER.info(
            f"[MIXED][hermes] run={run} factual_rmse={res2.get('test_rmse', np.nan):.4f} "
            f"sqrt_mise={res2.get('sqrt_mise', np.nan):.4f} mate={res2.get('mate', np.nan):.4f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg = copy.deepcopy(CONFIG)
    os.makedirs(cfg["out_dir"], exist_ok=True)

    discrete_bundle = load_discrete_bundle(cfg["tcga_path"], cfg["override_keys"])
    df_discrete = run_discrete_comparison(discrete_bundle, cfg)
    df_discrete.to_csv(os.path.join(cfg["out_dir"], "tcga_discrete_comparison.csv"), index=False)

    print("\n" + "=" * 120)
    print("DISCRETE TCGA COMPARISON (your current setting)")
    print("=" * 120)
    print(
        df_discrete.groupby("model")[["mise", "sqrt_mise", "mate", "epochs"]]
        .agg(["mean", "std"])
        .to_string(float_format="%.4f")
    )

    try:
        mixed_bundle = load_mixed_bundle(cfg["tcga_path"], cfg["override_keys"])
        LOGGER.info(
            f"[MIXED READY] num_groups={mixed_bundle.num_treatments} | "
            f"X={mixed_bundle.X.shape} | A={mixed_bundle.a.shape} | D={mixed_bundle.d.shape}"
        )

        df_mixed = run_mixed_comparison(mixed_bundle, cfg)
        df_mixed.to_csv(os.path.join(cfg["out_dir"], "tcga_mixed_comparison.csv"), index=False)

        print("\n" + "=" * 120)
        print("MIXED TCGA COMPARISON (observed groups + continuous dose)")
        print("=" * 120)
        cols = [c for c in [
            "test_mise",
            "test_rmse",
            "test_factual_rmse_norm",
            "mise",
            "sqrt_mise",
            "mate",
            "curve_sqrt_mise",
            "curve_mate",
            "epochs",
        ] if c in df_mixed.columns]
        print(df_mixed.groupby("model")[cols].agg(["mean", "std"]).to_string(float_format="%.4f"))

    except Exception as exc:
        LOGGER.warning("Mixed comparison not executed: %s", exc)
        print("\n" + "=" * 120)
        print("MIXED TCGA COMPARISON NOT RUN")
        print("=" * 120)
        print(str(exc))


if __name__ == "__main__":
    main()
