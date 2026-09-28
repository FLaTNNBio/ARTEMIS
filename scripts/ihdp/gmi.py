import os
import os.path
import logging
import zipfile
import requests
import numpy as np
import copy
import hashlib
import json
import pandas as pd
import time
import subprocess
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artemis.evaluation.artifacts import save_run_artifacts
from artemis.evaluation.mi_diagnostics import summarize_mi_selection
from artemis.evaluation.pair_diagnostics import compute_pair_audit
from artemis.evaluation.probes import evaluate_probe_suite
from artemis.evaluation.schemas import make_run_id, normalize_per_run_row
from artemis.models.cate import CATEEncoder, OutcomeHead, TreatmentClassifier

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("IHDP_ARTEMIS_MI_ABLATION")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# USER CONFIG
# ==============================================================================
# 10  -> smoke test
# 50  -> indicative results
# 1000 -> full IHDP benchmark
N_SIMS = 1000

SAVE_RESULTS = True
OUT_DIR = "ablation_outputs_ihdp_artemis_mi"
os.makedirs(OUT_DIR, exist_ok=True)

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
    'pair_mode': 'dynamic_ite',   # dynamic_ite | static_ite | random | feature_knn | none
    'feature_k': 20,
    # Legacy spelling retained here so the frozen default remains recognizable.
    # It normalizes to CurrentLocalMI; experiment configs use explicit names.
    'mi_mode': 'local_pos',
    'mi_use_schedule': True,
}

# ==============================================================================
# DATA LOADER
# ==============================================================================
def download_url(url, save_path, chunk_size=128):
    LOGGER.info(f"Downloading {url} into {save_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
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
        raise ValueError(f"dataset not supported: {dataset_name}")

    @abstractmethod
    def load(self):
        raise NotImplementedError


class IHDPLoader(AbstractCausalLoader):
    def __init__(self):
        super().__init__()

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

        os.makedirs(path, exist_ok=True)

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
            self.X_te, self.T_te, self.YF_te, self.YCF_te, self.mu_0_te, self.mu_1_te,
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
            self.best_pehe = float(val_pehe)
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


def treatment_log_prob_mean(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
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


def treatment_classifier_loss(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    logits = classifier(z)
    if num_treatments == 2:
        return F.binary_cross_entropy_with_logits(logits, t_idx.view(-1, 1).float())
    return F.cross_entropy(logits, t_idx.view(-1).long())


def variational_mi_lower_bound(classifier: nn.Module, z: torch.Tensor, t_idx: torch.Tensor, num_treatments: int) -> torch.Tensor:
    h_t = empirical_entropy_from_labels(t_idx.detach(), num_treatments)
    mean_log_q = treatment_log_prob_mean(classifier, z, t_idx, num_treatments)
    return h_t + mean_log_q


def balanced_selected_support_ce(logits, treatment):
    """Diagnostic CE under Q(T=0)=Q(T=1)=1/2, retaining every occurrence.

    This is a balanced selected-support objective, not natural-prior
    population mutual information. Duplicate occurrences keep their frequency
    within each arm. Each arm contributes total weight 1/2.
    """
    labels = treatment.reshape(-1)
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Balanced selected-support CE requires binary 0/1 labels')
    if not ((labels == 0).any() and (labels == 1).any()):
        raise ValueError('Balanced selected-support CE requires both treatment arms')
    losses = F.binary_cross_entropy_with_logits(
        logits.reshape(-1), labels.float(), reduction='none'
    )
    return 0.5 * losses[labels == 0].mean() + 0.5 * losses[labels == 1].mean()


def _diagnostic_classifier_scores(classifier, z_detached, t_idx, num_treatments,
                                  balanced_weighted=False):
    """Evaluate the actual classifier without dropout draws or buffer updates."""
    was_training = classifier.training
    classifier.eval()
    try:
        with torch.no_grad():
            logits = classifier(z_detached)
            if num_treatments == 2:
                ce = F.binary_cross_entropy_with_logits(
                    logits, t_idx.view(-1, 1).float()
                )
                if balanced_weighted:
                    ce = balanced_selected_support_ce(logits, t_idx)
                predicted = (logits.view(-1) >= 0).long()
            else:
                ce = F.cross_entropy(logits, t_idx.view(-1).long())
                predicted = logits.argmax(dim=1)
            accuracy = (predicted == t_idx.view(-1).long()).float().mean()
            return float(ce.item()), float(accuracy.item())
    finally:
        classifier.train(was_training)


def _diagnostic_encoder_gradient_norm(loss, encoder_parameters):
    """Read a component gradient without touching parameter.grad or stepping."""
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss, encoder_parameters, retain_graph=True, allow_unused=True
    )
    squared_norm = sum(
        float(gradient.detach().double().square().sum().item())
        for gradient in gradients if gradient is not None
    )
    return float(np.sqrt(squared_norm))


def _diagnostic_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-12 else float('nan')


def contrastive_loss(z1, z2, label, margin=1.0):
    dist_sq = torch.sum(torch.pow(z1 - z2, 2), dim=1)
    loss_sim = label * dist_sq
    loss_dissim = (1 - label) * torch.pow(torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0), 2)
    return torch.mean(loss_sim + loss_dissim) / 2.0


def get_continuous_indices(X):
    is_cont = []
    for c in range(X.shape[1]):
        unique_vals = np.unique(X[:, c])
        if len(unique_vals) > 2:
            is_cont.append(c)
    return is_cont


def stable_config_hash(hyperparams: Dict[str, Any]) -> str:
    """Hash only training-relevant configuration, excluding audit switches."""
    training_config = {
        key: value
        for key, value in hyperparams.items()
        if not key.startswith("instrumentation_")
        and not key.startswith("representation_probe_")
    }
    payload = json.dumps(
        training_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_index_hash(train_idx: np.ndarray, val_idx: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, values in (("train", train_idx), ("val", val_idx)):
        arr = np.asarray(values, dtype=np.int64)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def torch_module_state_hash(*modules: nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(f"{module_index}:{name}".encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def numpy_generator_state_hash(generator: np.random.Generator) -> str:
    payload = json.dumps(
        generator.bit_generator.state, sort_keys=True, separators=(",", ":"), default=int
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def torch_rng_state_hash() -> str:
    digest = hashlib.sha256(torch.random.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def build_fixed_audit_pair_panel(n_units: int, panel_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a fixed unordered pair panel without touching any training RNG."""
    if n_units < 2 or panel_size <= 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty

    panel_rng = np.random.default_rng(seed)
    idx_a_all, idx_b_all = np.triu_indices(n_units, k=1)
    if panel_size < idx_a_all.size:
        selected = panel_rng.choice(idx_a_all.size, size=panel_size, replace=False)
        return idx_a_all[selected], idx_b_all[selected]
    return idx_a_all, idx_b_all


def audit_pair_panel_hash(audit_pair_panel: Tuple[np.ndarray, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for values in audit_pair_panel:
        arr = np.asarray(values, dtype=np.int64)
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x).reshape(-1)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y).reshape(-1)).rank(method="average").to_numpy()
    return _safe_pearson(x_rank, y_rank)


def compute_refresh_diagnostics(
    dataset,
    audit_pair_panel: Tuple[np.ndarray, np.ndarray],
    true_tau: np.ndarray,
    y_std: float,
    epoch: int,
    refresh_id: int,
    previous_positive_mask: Optional[np.ndarray],
    val_metric: Optional[float] = None,
    active_support_indices: Optional[np.ndarray] = None,
    local_mi_unique_indices: Optional[np.ndarray] = None,
):
    """Compute audit-only metrics on one fixed pair panel."""
    pseudo_tau_norm = (
        np.asarray(dataset.current_mu1_hat).reshape(-1)
        - np.asarray(dataset.current_mu0_hat).reshape(-1)
    )
    pseudo_tau = pseudo_tau_norm * float(y_std)
    true_tau = np.asarray(true_tau).reshape(-1)
    errors = pseudo_tau - true_tau
    idx_a, idx_b = audit_pair_panel

    previous_support_mask = None
    if previous_positive_mask is not None:
        previous_support_mask = np.zeros(true_tau.size, dtype=bool)
        previous_a = audit_pair_panel[0][previous_positive_mask]
        previous_b = audit_pair_panel[1][previous_positive_mask]
        previous_support_mask[np.unique(np.concatenate([previous_a, previous_b]))] = True

    pair_row, pair_arrays = compute_pair_audit(
        pseudo_tau=pseudo_tau,
        treatment=dataset.T_all,
        panel=audit_pair_panel,
        predicted_threshold=float(dataset.thr) * float(y_std),
        percentile=float(dataset.perc),
        true_tau=true_tau,
        previous_positive_mask=previous_positive_mask,
        previous_support_mask=previous_support_mask,
        active_support_indices=active_support_indices,
        local_mi_unique_indices=local_mi_unique_indices,
    )
    positive_mask = pair_arrays["predicted_positive_mask"]

    diagnostics = {
        "epoch": int(epoch),
        "refresh_id": int(refresh_id),
        "pseudo_ite_mean": float(np.mean(pseudo_tau)),
        "pseudo_ite_std": float(np.std(pseudo_tau)),
        "pseudo_ite_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "pseudo_ite_mae": float(np.mean(np.abs(errors))),
        "pseudo_ite_pearson": _safe_pearson(pseudo_tau, true_tau),
        "pseudo_ite_spearman": _safe_spearman(pseudo_tau, true_tau),
        "pseudo_ite_rmse_vs_true": float(np.sqrt(np.mean(errors ** 2))),
        "pseudo_ite_mae_vs_true": float(np.mean(np.abs(errors))),
        "pseudo_ite_pearson_vs_true": _safe_pearson(pseudo_tau, true_tau),
        "pseudo_ite_spearman_vs_true": _safe_spearman(pseudo_tau, true_tau),
        "true_positive_pair_effect_gap_mean": pair_row.get("true_effect_gap_mean"),
        "true_positive_pair_effect_gap_median": pair_row.get("true_effect_gap_median"),
        "true_positive_pair_effect_gap_q90": pair_row.get("true_effect_gap_q90"),
        "positive_edge_jaccard_previous_refresh": pair_row.get("positive_edge_jaccard"),
        "audit_pair_panel_size": int(idx_a.size),
        "pair_threshold_normalized": float(dataset.thr),
        "pair_threshold_original_outcome_units": float(dataset.thr) * float(y_std),
        "validation_metric_at_refresh": float(val_metric) if val_metric is not None else float("nan"),
        "pseudo_ite_and_true_gap_scale": "original_outcome_units",
        **pair_row,
    }
    return diagnostics, positive_mask.copy()


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
    return float(np.clip(thr, thr_min, thr_max))


def _empty_pair_batch(X, T, Y):
    empty_shape = (0,) + X.shape[1:]
    return (
        np.zeros(empty_shape, dtype=X.dtype), np.zeros((0,) + Y.shape[1:], dtype=Y.dtype), np.zeros((0,) + T.shape[1:], dtype=T.dtype),
        np.zeros(empty_shape, dtype=X.dtype), np.zeros((0,) + Y.shape[1:], dtype=Y.dtype), np.zeros((0,) + T.shape[1:], dtype=T.dtype),
        np.array([], dtype=np.int64),
    )

# ==============================================================================
# PAIRING GENERATORS
# ==============================================================================
def make_pairs_random(X, T, Y, n_pairs, seed=None, audit_callback=None):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 2:
        return _empty_pair_batch(X, T, Y)

    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N - 1, size=n_pairs)
    idx_b = np.where(idx_b >= idx_a, idx_b + 1, idx_b)

    labels = rng.integers(0, 2, size=n_pairs).astype(np.int64)
    if audit_callback is not None:
        audit_callback(idx_a, idx_b, labels)
    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels,
    )


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None, audit_callback=None):
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

    if audit_callback is not None:
        audit_callback(idx_a, idx_b, labels)
    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels,
    )


def make_pairs_feature_knn(X, T, Y, n_pairs, k=20, seed=None, audit_callback=None):
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
            topk = sorted_idx[:min(k, len(sorted_idx))]
            j = int(rng.choice(topk))
            labels[p] = 1
        else:
            far = sorted_idx[max(1, len(sorted_idx) - min(k, len(sorted_idx))):]
            j = int(rng.choice(far))
            labels[p] = 0
        idx_b[p] = j

    if audit_callback is not None:
        audit_callback(idx_a, idx_b, labels)
    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels,
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
        feature_k=20,
        instrumentation_enabled=False,
        include_pair_indices=False,
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
        self.instrumentation_enabled = bool(instrumentation_enabled)
        self.include_pair_indices = bool(include_pair_indices)
        self._active_positive_endpoints = set()
        self._last_idx_a = np.array([], dtype=np.int64)
        self._last_idx_b = np.array([], dtype=np.int64)

        if mu0_hat is None or mu1_hat is None:
            self.current_mu0_hat = np.zeros(X_all.shape[0], dtype=np.float32)
            self.current_mu1_hat = np.zeros(X_all.shape[0], dtype=np.float32)
        else:
            self.current_mu0_hat = mu0_hat
            self.current_mu1_hat = mu1_hat

        self.update_threshold()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._active_positive_endpoints = set()

    def _record_sampled_pairs(self, idx_a, idx_b, labels):
        self._last_idx_a = np.asarray(idx_a, dtype=np.int64).copy()
        self._last_idx_b = np.asarray(idx_b, dtype=np.int64).copy()
        if not self.instrumentation_enabled:
            return
        positive_mask = np.asarray(labels).reshape(-1) == 1
        if positive_mask.any():
            self._active_positive_endpoints.update(np.asarray(idx_a)[positive_mask].tolist())
            self._active_positive_endpoints.update(np.asarray(idx_b)[positive_mask].tolist())

    def sampled_active_positive_support_fraction(self) -> float:
        if self.X_all.shape[0] == 0:
            return float("nan")
        return float(len(self._active_positive_endpoints) / self.X_all.shape[0])

    def sampled_active_positive_support_indices(self) -> np.ndarray:
        return np.asarray(sorted(self._active_positive_endpoints), dtype=np.int64)

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
        self._last_idx_a = np.array([], dtype=np.int64)
        self._last_idx_b = np.array([], dtype=np.int64)

        if self.pair_mode == 'none':
            empty_batch = (
                torch.zeros((0, self.X_all.shape[1]), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0, self.X_all.shape[1]), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )
            if self.include_pair_indices:
                empty_indices = torch.zeros((0,), dtype=torch.long)
                return empty_batch + (empty_indices, empty_indices.clone())
            return empty_batch

        capture_indices = self.instrumentation_enabled or self.include_pair_indices

        if self.pair_mode in ['dynamic_ite', 'static_ite']:
            x1, y1, t1, x2, y2, t2, lab = make_pairs_from_hat(
                self.X_all, self.T_all, self.Y_all,
                self.current_mu0_hat, self.current_mu1_hat,
                self.thr, self.bs, seed=seed,
                audit_callback=self._record_sampled_pairs if capture_indices else None,
            )
        elif self.pair_mode == 'random':
            x1, y1, t1, x2, y2, t2, lab = make_pairs_random(
                self.X_all, self.T_all, self.Y_all,
                self.bs, seed=seed,
                audit_callback=self._record_sampled_pairs if capture_indices else None,
            )
        elif self.pair_mode == 'feature_knn':
            x1, y1, t1, x2, y2, t2, lab = make_pairs_feature_knn(
                self.X_all, self.T_all, self.Y_all,
                self.bs, k=self.feature_k, seed=seed,
                audit_callback=self._record_sampled_pairs if capture_indices else None,
            )
        else:
            raise ValueError(f"Unknown pair_mode: {self.pair_mode}")

        pair_batch = (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.float32),
            torch.tensor(t1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y2, dtype=torch.float32),
            torch.tensor(t2, dtype=torch.float32),
            torch.tensor(lab, dtype=torch.long),
        )
        if self.include_pair_indices:
            return pair_batch + (
                torch.tensor(self._last_idx_a, dtype=torch.long),
                torch.tensor(self._last_idx_b, dtype=torch.long),
            )
        return pair_batch

# ==============================================================================
# MODEL
# ==============================================================================

def _crossfit_seed(base_seed: int, fold: int, epoch: int, purpose: str) -> int:
    payload = f"{int(base_seed)}|{int(fold)}|{int(epoch)}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _torch_fork_devices(device: str):
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return []
    return [resolved.index if resolved.index is not None else torch.cuda.current_device()]


def build_persistent_crossfit_teachers(
    x_train_raw: np.ndarray,
    treatment_train: np.ndarray,
    outcome_train_raw: np.ndarray,
    n_splits: int,
    base_seed: int,
    device: str,
    latent_dim: int,
    encoder_dropout: float,
    head_dropout: float,
    num_treatments: int,
    use_output_clip: bool,
    outcome_clip_factor: float,
    learning_rate: float,
    weight_decay: float,
    training_horizon: int,
):
    """Create independent OOF teachers without changing the main-model RNG.

    Teachers have the ARTEMIS encoder/outcome-head architecture but use only
    factual supervised loss.  They are persistent and trained in lock-step;
    no teacher is initialized from, or shares parameters with, the main model.
    """
    from sklearn.model_selection import StratifiedKFold

    x_train_raw = np.asarray(x_train_raw)
    treatment_train = np.asarray(treatment_train).reshape(-1).astype(np.int64)
    outcome_train_raw = np.asarray(outcome_train_raw, dtype=np.float64).reshape(-1)
    if int(n_splits) != 5:
        raise ValueError("CrossFittedTeacher reviewer diagnostic requires exactly K=5")
    class_counts = np.bincount(treatment_train, minlength=num_treatments)
    if np.any(class_counts < int(n_splits)):
        raise ValueError(
            f"Every treatment arm needs at least K={n_splits} rows; found {class_counts.tolist()}"
        )
    split_seed = _crossfit_seed(base_seed, -1, -1, "stratified_fold_partition")
    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(split_seed)
    )
    coverage_count = np.zeros(x_train_raw.shape[0], dtype=np.int64)
    teachers = []
    fork_devices = _torch_fork_devices(device)
    for fold, (fit_indices, heldout_indices) in enumerate(
        splitter.split(np.zeros(x_train_raw.shape[0]), treatment_train)
    ):
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        heldout_indices = np.asarray(heldout_indices, dtype=np.int64)
        if np.intersect1d(fit_indices, heldout_indices).size:
            raise RuntimeError(f"Cross-fit leakage in fold {fold}: fit/held-out overlap")
        coverage_count[heldout_indices] += 1

        # Even feature-type detection is fold-local: held-out covariates do not
        # contribute to teacher preprocessing decisions or statistics.
        fold_continuous_indices = get_continuous_indices(x_train_raw[fit_indices])
        x_processed = x_train_raw.copy()
        if fold_continuous_indices:
            fit_continuous = x_processed[fit_indices][:, fold_continuous_indices]
            x_mean = np.mean(fit_continuous, axis=0, keepdims=True)
            x_std = np.maximum(np.std(fit_continuous, axis=0, keepdims=True), 1e-6)
            x_processed[:, fold_continuous_indices] = (
                x_processed[:, fold_continuous_indices] - x_mean
            ) / x_std
        y_mean = float(np.mean(outcome_train_raw[fit_indices]))
        y_std = float(np.std(outcome_train_raw[fit_indices]))
        if y_std < 1e-6:
            y_std = 1.0
        y_normalized = (outcome_train_raw - y_mean) / y_std
        maximum_normalized_outcome = float(np.max(np.abs(y_normalized[fit_indices])))
        clip_value = max(maximum_normalized_outcome * float(outcome_clip_factor), 3.0)

        initialization_seed = _crossfit_seed(base_seed, fold, -1, "teacher_initialization")
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.manual_seed(initialization_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(initialization_seed)
            teacher_encoder = CATEEncoder(
                x_train_raw.shape[1], latent_dim, dropout=encoder_dropout
            ).to(device)
            teacher_predictor = OutcomeHead(
                latent_dim,
                num_treatments=num_treatments,
                use_output_clip=use_output_clip,
                clip_val=clip_value,
                dropout=head_dropout,
            ).to(device)
        teacher_optimizer = optim.AdamW(
            list(teacher_encoder.parameters()) + list(teacher_predictor.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            teacher_optimizer, T_max=int(training_horizon), eta_min=1e-6
        )
        teachers.append({
            "fold": int(fold),
            "fit_indices": fit_indices,
            "heldout_indices": heldout_indices,
            "x_processed": x_processed,
            "continuous_indices": tuple(fold_continuous_indices),
            "y_normalized": y_normalized,
            "y_mean": y_mean,
            "y_std": y_std,
            "encoder": teacher_encoder,
            "predictor": teacher_predictor,
            "optimizer": teacher_optimizer,
            "scheduler": teacher_scheduler,
            "optimization_steps": 0,
        })
    if not np.all(coverage_count == 1):
        raise RuntimeError(
            "Invalid cross-fit partition: every training unit must be held out exactly once"
        )
    return teachers, {
        "crossfit_folds": int(n_splits),
        "crossfit_fold_seed": int(split_seed),
        "crossfit_oof_coverage_fraction": float(np.mean(coverage_count > 0)),
        "crossfit_oof_exactly_once_fraction": float(np.mean(coverage_count == 1)),
        "crossfit_oof_exactly_once": bool(np.all(coverage_count == 1)),
        "crossfit_no_fit_holdout_overlap": True,
        "crossfit_teacher_data_scope": "artemis_training_partition_only",
    }


def train_crossfit_teachers_one_epoch(
    teachers,
    treatment_train: np.ndarray,
    epoch: int,
    base_seed: int,
    batch_size: int,
    device: str,
    huber_beta: float,
    clip_norm: float,
) -> None:
    """Advance each persistent teacher once with an isolated deterministic RNG."""
    treatment_train = np.asarray(treatment_train).reshape(-1).astype(np.int64)
    fork_devices = _torch_fork_devices(device)
    for teacher in teachers:
        fold = int(teacher["fold"])
        epoch_seed = _crossfit_seed(base_seed, fold, epoch, "teacher_epoch")
        order = np.random.default_rng(epoch_seed).permutation(teacher["fit_indices"])
        encoder, predictor = teacher["encoder"], teacher["predictor"]
        optimizer = teacher["optimizer"]
        encoder.train()
        predictor.train()
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.manual_seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(epoch_seed)
            for start in range(0, order.size, int(batch_size)):
                indices = order[start:start + int(batch_size)]
                x_batch = torch.as_tensor(
                    teacher["x_processed"][indices], dtype=torch.float32, device=device
                )
                t_batch = torch.as_tensor(
                    treatment_train[indices], dtype=torch.long, device=device
                )
                y_batch = torch.as_tensor(
                    teacher["y_normalized"][indices], dtype=torch.float32, device=device
                ).view(-1, 1)
                optimizer.zero_grad()
                prediction = predictor(encoder(x_batch)).gather(1, t_batch.unsqueeze(1))
                loss = F.smooth_l1_loss(prediction, y_batch, beta=float(huber_beta))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), float(clip_norm)
                )
                optimizer.step()
                teacher["optimization_steps"] += 1
        teacher["scheduler"].step()


def predict_crossfit_potential_outcomes(
    teachers,
    n_training_units: int,
    main_y_mean: float,
    main_y_std: float,
    device: str,
):
    """Join exactly one held-out prediction per row in the main outcome scale."""
    mu0 = np.full(int(n_training_units), np.nan, dtype=np.float64)
    mu1 = np.full(int(n_training_units), np.nan, dtype=np.float64)
    coverage_count = np.zeros(int(n_training_units), dtype=np.int64)
    for teacher in teachers:
        heldout = teacher["heldout_indices"]
        teacher["encoder"].eval()
        teacher["predictor"].eval()
        with torch.no_grad():
            x_heldout = torch.as_tensor(
                teacher["x_processed"][heldout], dtype=torch.float32, device=device
            )
            prediction_normalized = teacher["predictor"](
                teacher["encoder"](x_heldout)
            ).cpu().numpy()
        prediction_raw = prediction_normalized * teacher["y_std"] + teacher["y_mean"]
        prediction_main_scale = (prediction_raw - float(main_y_mean)) / float(main_y_std)
        mu0[heldout] = prediction_main_scale[:, 0]
        mu1[heldout] = prediction_main_scale[:, 1]
        coverage_count[heldout] += 1
    complete = bool(
        np.all(coverage_count == 1) and np.all(np.isfinite(mu0)) and np.all(np.isfinite(mu1))
    )
    if not complete:
        raise RuntimeError("CrossFittedTeacher failed complete exactly-once OOF prediction")
    audit = {
        "crossfit_oof_coverage_fraction": float(np.mean(coverage_count > 0)),
        "crossfit_oof_exactly_once_fraction": float(np.mean(coverage_count == 1)),
        "crossfit_oof_exactly_once": True,
        "crossfit_no_fit_holdout_overlap": True,
        "crossfit_teacher_optimization_steps_total": int(sum(
            teacher["optimization_steps"] for teacher in teachers
        )),
    }
    return mu0, mu1, audit


# ==============================================================================
# TRAIN
# ==============================================================================
MI_NO = "NoMI"
MI_CURRENT_LOCAL = "CurrentLocalMI"
MI_BALANCED_LOCAL = "BalancedLocalMI"
MI_WEIGHTED_BALANCED_LOCAL = "WeightedBalancedLocalMI"
MI_CURRENT_GLOBAL = "CurrentGlobalMI"
MI_TRUE_GLOBAL = "TrueGlobalMI"
MI_DEDUPLICATED_SUPPORT = "DeduplicatedSupportMI"

CHECKPOINT_SUBMISSION = "submission_checkpoint_protocol"
CHECKPOINT_REVISED_FULL_OBJECTIVE = "revised_full_objective_checkpoint_protocol"
# Backward-compatible Python symbol retained for the earlier audit runner.  It
# now resolves to the final reviewer-facing protocol name; the old serialized
# string is accepted below as an input alias only.
CHECKPOINT_REVISED_POST_MI = CHECKPOINT_REVISED_FULL_OBJECTIVE

MI_MODE_ALIASES = {
    "none": MI_NO,
    "nomi": MI_NO,
    "local": MI_CURRENT_LOCAL,
    "local_pos": MI_CURRENT_LOCAL,
    "currentlocalmi": MI_CURRENT_LOCAL,
    "balancedlocalmi": MI_BALANCED_LOCAL,
    "weightedbalancedlocalmi": MI_WEIGHTED_BALANCED_LOCAL,
    "global": MI_CURRENT_GLOBAL,
    "global_batch": MI_CURRENT_GLOBAL,
    "currentglobalmi": MI_CURRENT_GLOBAL,
    "trueglobalmi": MI_TRUE_GLOBAL,
    "deduplicatedsupportmi": MI_DEDUPLICATED_SUPPORT,
}


def normalize_mi_mode(mi_mode: str) -> str:
    key = str(mi_mode).strip().lower().replace("-", "").replace("_", "")
    normalized_aliases = {
        alias.replace("_", ""): canonical
        for alias, canonical in MI_MODE_ALIASES.items()
    }
    if key not in normalized_aliases:
        supported = ", ".join(
            [MI_NO, MI_CURRENT_LOCAL, MI_CURRENT_GLOBAL, MI_TRUE_GLOBAL, MI_DEDUPLICATED_SUPPORT, MI_BALANCED_LOCAL, MI_WEIGHTED_BALANCED_LOCAL]
        )
        raise ValueError(f"Unknown mi_mode: {mi_mode}. Supported modes: {supported}")
    return normalized_aliases[key]


def normalize_checkpoint_protocol(checkpoint_protocol: str) -> str:
    key = str(checkpoint_protocol).strip().lower().replace("-", "_")
    aliases = {
        CHECKPOINT_SUBMISSION: CHECKPOINT_SUBMISSION,
        "submission": CHECKPOINT_SUBMISSION,
        "current": CHECKPOINT_SUBMISSION,
        CHECKPOINT_REVISED_FULL_OBJECTIVE: CHECKPOINT_REVISED_FULL_OBJECTIVE,
        "revised_post_mi_checkpoint_protocol": CHECKPOINT_REVISED_FULL_OBJECTIVE,
        "revised": CHECKPOINT_REVISED_FULL_OBJECTIVE,
        "post_mi": CHECKPOINT_REVISED_FULL_OBJECTIVE,
        "full_objective": CHECKPOINT_REVISED_FULL_OBJECTIVE,
    }
    if key not in aliases:
        supported = ", ".join((CHECKPOINT_SUBMISSION, CHECKPOINT_REVISED_FULL_OBJECTIVE))
        raise ValueError(
            f"Unknown checkpoint protocol: {checkpoint_protocol}. "
            f"Supported protocols: {supported}"
        )
    return aliases[key]


def resolve_checkpoint_start_epoch(
    checkpoint_protocol: str,
    mi_start_epoch: int,
    warmup_epochs: int,
    use_contrastive: bool,
    explicit_start_epoch=None,
) -> int:
    """Return the first validation checkpoint eligible for model selection.

    The frozen submission protocol remains eligible from epoch zero.  The
    revised protocol uses one common post-objective-activation epoch for every
    method; callers running an ablation should pass the same explicit value to
    all configurations.
    """
    protocol = normalize_checkpoint_protocol(checkpoint_protocol)
    if protocol == CHECKPOINT_SUBMISSION:
        return 0
    if explicit_start_epoch is not None:
        start_epoch = int(explicit_start_epoch)
    else:
        contrastive_start = int(warmup_epochs) + 1 if use_contrastive else 0
        start_epoch = max(int(mi_start_epoch), contrastive_start)
    if start_epoch < 0:
        raise ValueError("checkpoint_start_epoch must be non-negative")
    return start_epoch


def checkpoint_is_eligible(epoch: int, checkpoint_start_epoch: int) -> bool:
    return int(epoch) >= int(checkpoint_start_epoch)


def cross_fitted_ridge_potential_outcomes(
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_splits: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagnostic out-of-fold T-learner; never used by standard ARTEMIS."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import StratifiedKFold

    x = np.asarray(x)
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    outcome = np.asarray(outcome).reshape(-1)
    if np.unique(treatment).size != 2:
        raise ValueError("cross_fitted_ridge currently requires binary treatment")
    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    mu0 = np.zeros(outcome.size, dtype=np.float64)
    mu1 = np.zeros(outcome.size, dtype=np.float64)
    for fit_indices, held_out_indices in splitter.split(x, treatment):
        for treatment_value, destination in ((0, mu0), (1, mu1)):
            arm_indices = fit_indices[treatment[fit_indices] == treatment_value]
            if arm_indices.size < 2:
                destination[held_out_indices] = float(np.mean(outcome[fit_indices]))
                continue
            model = Ridge(alpha=1.0)
            model.fit(x[arm_indices], outcome[arm_indices])
            destination[held_out_indices] = model.predict(x[held_out_indices])
    return mu0, mu1


def _as_numpy_indices(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.int64)
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _stable_unique(values: np.ndarray) -> np.ndarray:
    seen = set()
    unique_values = []
    for value in _as_numpy_indices(values).tolist():
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return np.asarray(unique_values, dtype=np.int64)


def select_mi_training_indices(
    mi_mode: str,
    idx_a,
    idx_b,
    positive_mask,
    independent_unit_indices=None,
) -> np.ndarray:
    """Pure audit helper describing the training-unit IDs used by each MI mode."""
    mode = normalize_mi_mode(mi_mode)
    idx_a_np = _as_numpy_indices(idx_a)
    idx_b_np = _as_numpy_indices(idx_b)
    positive_np = np.asarray(positive_mask, dtype=bool).reshape(-1)

    if mode == MI_NO:
        return np.array([], dtype=np.int64)
    if mode == MI_TRUE_GLOBAL:
        if independent_unit_indices is None:
            raise ValueError("TrueGlobalMI requires independent_unit_indices")
        return _as_numpy_indices(independent_unit_indices).copy()
    if idx_a_np.size != idx_b_np.size or idx_a_np.size != positive_np.size:
        raise ValueError("idx_a, idx_b, and positive_mask must have equal length")
    if mode == MI_CURRENT_GLOBAL:
        return np.concatenate([idx_a_np, idx_b_np])

    positive_endpoints = np.concatenate([idx_a_np[positive_np], idx_b_np[positive_np]])
    if mode == MI_CURRENT_LOCAL:
        return positive_endpoints
    if mode == MI_DEDUPLICATED_SUPPORT:
        return _stable_unique(positive_endpoints)
    raise AssertionError(f"Unhandled normalized MI mode: {mode}")


def sample_independent_unit_indices(
    n_units: int,
    batch_size: int,
    base_seed: int,
    epoch: int,
    batch_id: int,
) -> np.ndarray:
    """Return one slice of an independent per-epoch unit permutation."""
    if n_units <= 0:
        return np.array([], dtype=np.int64)
    draw_seed = (
        int(base_seed)
        + 1_000_003 * int(epoch)
        + 2_000_000_011
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(draw_seed)
    permutation = rng.permutation(n_units).astype(np.int64)
    start = int(batch_id) * int(batch_size)
    stop = min(start + int(batch_size), int(n_units))
    return permutation[start:stop]


def _first_occurrence_positions(unit_indices) -> np.ndarray:
    unit_indices_np = _as_numpy_indices(unit_indices)
    seen = set()
    positions = []
    for position, unit_index in enumerate(unit_indices_np.tolist()):
        if unit_index not in seen:
            seen.add(unit_index)
            positions.append(position)
    return np.asarray(positions, dtype=np.int64)


def balanced_endpoint_positions(treatment, replication_id, epoch, batch_id):
    """Diagnostic selection of endpoint occurrences, never of outside units.

    Downsample each class to the minority count without replacement. Repeated
    patients remain distinct endpoint occurrences. Uses a private RNG.
    """
    labels = treatment.detach().cpu().numpy().reshape(-1)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("BalancedLocalMI is binary-treatment diagnostic only")
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    count = min(map(len, groups))
    rng = np.random.default_rng(np.random.SeedSequence(
        [910003, int(replication_id), int(epoch), int(batch_id)]
    ))
    return np.concatenate([rng.choice(group, count, replace=False) for group in groups])


def _diagnostic_binary_classifier_extra(classifier, z, treatment):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    was_training = classifier.training
    classifier.eval()
    try:
        with torch.no_grad():
            probability = torch.sigmoid(classifier(z)).view(-1).cpu().numpy()
        labels = treatment.view(-1).cpu().numpy()
        predicted = probability >= 0.5
        both = np.unique(labels).size == 2
        return {
            'clf_auc': float(roc_auc_score(labels, probability)) if both else float('nan'),
            'clf_balanced_accuracy': float(balanced_accuracy_score(labels, predicted)) if both else float('nan'),
            'clf_predicted_positive_rate': float(predicted.mean()),
            'clf_single_class_prediction': bool(np.unique(predicted).size == 1),
        }
    finally:
        classifier.train(was_training)


def select_mi_subset(
    mi_mode: str,
    z1,
    z2,
    t1_idx,
    t2_idx,
    pos_mask,
    idx_a=None,
    idx_b=None,
    z_independent=None,
    t_independent=None,
    independent_unit_indices=None,
    balanced_positions=None,
):
    mode = normalize_mi_mode(mi_mode)
    if mode == MI_WEIGHTED_BALANCED_LOCAL:
        z, t, indices = select_mi_subset(
            MI_CURRENT_LOCAL, z1, z2, t1_idx, t2_idx, pos_mask,
            idx_a=idx_a, idx_b=idx_b,
        )
        if t is None or not ((t == 0).any() and (t == 1).any()):
            return None, None, np.array([], dtype=np.int64)
        return z, t, indices
    if mode == MI_BALANCED_LOCAL:
        if balanced_positions is None:
            raise ValueError("BalancedLocalMI requires shared endpoint positions for both updates")
        if idx_a is None or idx_b is None:
            raise ValueError("BalancedLocalMI requires original endpoint indices")
        z, t, indices = select_mi_subset(
            MI_CURRENT_LOCAL, z1, z2, t1_idx, t2_idx, pos_mask,
            idx_a=idx_a, idx_b=idx_b,
        )
        if z is None:
            return None, None, indices
        keep = torch.as_tensor(balanced_positions, dtype=torch.long, device=z.device)
        return z.index_select(0, keep), t.index_select(0, keep), indices[balanced_positions]
    if mode == MI_NO:
        return None, None, np.array([], dtype=np.int64)
    if mode == MI_TRUE_GLOBAL:
        if z_independent is None or t_independent is None or independent_unit_indices is None:
            raise ValueError("TrueGlobalMI requires an independent unit minibatch")
        return z_independent, t_independent, _as_numpy_indices(independent_unit_indices)
    if mode == MI_CURRENT_LOCAL:
        if pos_mask.sum().item() == 0:
            return None, None, np.array([], dtype=np.int64)
        z_sel = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)
        t_sel = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
        unit_indices = (
            select_mi_training_indices(mode, idx_a, idx_b, _as_numpy_indices(pos_mask))
            if idx_a is not None and idx_b is not None
            else np.array([], dtype=np.int64)
        )
        return z_sel, t_sel, unit_indices
    if mode == MI_CURRENT_GLOBAL:
        z_sel = torch.cat([z1, z2], dim=0)
        t_sel = torch.cat([t1_idx, t2_idx], dim=0)
        unit_indices = (
            select_mi_training_indices(mode, idx_a, idx_b, _as_numpy_indices(pos_mask))
            if idx_a is not None and idx_b is not None
            else np.array([], dtype=np.int64)
        )
        return z_sel, t_sel, unit_indices
    if mode == MI_DEDUPLICATED_SUPPORT:
        if pos_mask.sum().item() == 0:
            return None, None, np.array([], dtype=np.int64)
        if idx_a is None or idx_b is None:
            raise ValueError("DeduplicatedSupportMI requires pair endpoint indices")
        z_positive = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)
        t_positive = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
        positive_unit_indices = torch.cat([idx_a[pos_mask], idx_b[pos_mask]], dim=0)
        keep_positions_np = _first_occurrence_positions(positive_unit_indices)
        keep_positions = torch.as_tensor(keep_positions_np, dtype=torch.long, device=z_positive.device)
        return (
            z_positive.index_select(0, keep_positions),
            t_positive.index_select(0, keep_positions),
            _as_numpy_indices(positive_unit_indices)[keep_positions_np],
        )
    raise AssertionError(f"Unhandled normalized MI mode: {mode}")


def evaluate_frozen_representation_probes(
    encoder: nn.Module,
    x_probe_train: np.ndarray,
    treatment_probe_train: np.ndarray,
    tau_probe_train: np.ndarray,
    x_probe_eval: np.ndarray,
    treatment_probe_eval: np.ndarray,
    tau_probe_eval: np.ndarray,
    device: str,
    probe_seed: int,
) -> Dict[str, Any]:
    """Fit fixed linear probes on frozen Z and evaluate on the held-out split.

    Probe fitting happens only after the selected model checkpoint has been
    restored, so it cannot alter optimization or checkpoint selection.  Both
    probes fit on the IHDP training partition and evaluate on the untouched
    IHDP test sample, which is identical across paired methods.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    encoder.eval()
    with torch.no_grad():
        z_probe_train = encoder(
            torch.as_tensor(x_probe_train, dtype=torch.float32, device=device)
        ).cpu().numpy()
        z_probe_eval = encoder(
            torch.as_tensor(x_probe_eval, dtype=torch.float32, device=device)
        ).cpu().numpy()

    treatment_probe_train = np.asarray(treatment_probe_train).reshape(-1).astype(int)
    treatment_probe_eval = np.asarray(treatment_probe_eval).reshape(-1).astype(int)
    tau_probe_train = np.asarray(tau_probe_train, dtype=np.float64).reshape(-1)
    tau_probe_eval = np.asarray(tau_probe_eval, dtype=np.float64).reshape(-1)

    treatment_auc = float("nan")
    if (
        np.unique(treatment_probe_train).size >= 2
        and np.unique(treatment_probe_eval).size >= 2
    ):
        treatment_probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2_000,
                solver="lbfgs",
                random_state=int(probe_seed),
            ),
        )
        treatment_probe.fit(z_probe_train, treatment_probe_train)
        treatment_probability = treatment_probe.predict_proba(z_probe_eval)[:, 1]
        treatment_auc = float(
            roc_auc_score(treatment_probe_eval, treatment_probability)
        )

    effect_r2 = float("nan")
    if tau_probe_train.size >= 2 and tau_probe_eval.size >= 2:
        effect_probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        effect_probe.fit(z_probe_train, tau_probe_train)
        effect_r2 = float(r2_score(tau_probe_eval, effect_probe.predict(z_probe_eval)))

    return {
        "treatment_probe_auc": treatment_auc,
        "effect_probe_r2": effect_r2,
        "representation_probe_protocol": (
            "linear_probes_fit_on_train_partition_evaluated_on_ihdp_test"
        ),
        "treatment_probe_model": (
            "StandardScaler+LogisticRegression(C=1.0,solver=lbfgs,max_iter=2000)"
        ),
        "effect_probe_model": "StandardScaler+Ridge(alpha=1.0)",
        "representation_probe_seed": int(probe_seed),
        "representation_probe_train_n": int(z_probe_train.shape[0]),
        "representation_probe_eval_n": int(z_probe_eval.shape[0]),
    }


def train_single_simulation(sim_idx: int, data_train: Tuple, data_test: Tuple, device: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    run_started_at = time.perf_counter()
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
    USE_DYNAMIC_UPDATE = hyperparams.get('use_dynamic_update', True)
    PAIR_MODE = hyperparams.get('pair_mode', 'dynamic_ite')
    FEATURE_K = hyperparams.get('feature_k', 20)
    PAIR_TEACHER_MODE = str(hyperparams.get('pair_teacher_mode', 'self_updating'))
    if PAIR_TEACHER_MODE not in {
        'self_updating', 'standard_artemis', 'cross_fitted_ridge',
        'cross_fitted_teacher', 'oracle', 'oracle_pairing',
        'frozen_after_warmup'
    }:
        raise ValueError(f"Unknown pair_teacher_mode: {PAIR_TEACHER_MODE}")
    MI_MODE = normalize_mi_mode(hyperparams.get('mi_mode', MI_CURRENT_LOCAL))
    MI_USE_SCHEDULE = hyperparams.get('mi_use_schedule', True)
    # Opt-in so importing/running the frozen submission configuration keeps its
    # historical behavior.  Canonical reviewer runners explicitly enable it.
    INSTRUMENTATION_ENABLED = hyperparams.get('instrumentation_enabled', False)
    MI_GRADIENT_DIAGNOSTICS = bool(
        hyperparams.get('instrumentation_mi_gradient_diagnostics', False)
    )
    MI_GRADIENT_MAX_BATCHES = int(
        hyperparams.get('instrumentation_mi_gradient_max_batches', 30)
    )
    if MI_GRADIENT_MAX_BATCHES < 1:
        raise ValueError('instrumentation_mi_gradient_max_batches must be positive')
    AUDIT_PAIR_PANEL_SIZE = int(hyperparams.get('instrumentation_pair_panel_size', 20_000))
    CHECKPOINT_PROTOCOL = normalize_checkpoint_protocol(
        hyperparams.get('checkpoint_protocol', CHECKPOINT_SUBMISSION)
    )
    CHECKPOINT_START_EPOCH = resolve_checkpoint_start_epoch(
        checkpoint_protocol=CHECKPOINT_PROTOCOL,
        mi_start_epoch=MI_START_EPOCH,
        warmup_epochs=WARMUP_EPOCHS,
        use_contrastive=bool(USE_CONTRASTIVE and PAIR_MODE != 'none'),
        explicit_start_epoch=hyperparams.get('checkpoint_start_epoch'),
    )
    REVISED_REFERENCE_START_EPOCH = resolve_checkpoint_start_epoch(
        checkpoint_protocol=CHECKPOINT_REVISED_FULL_OBJECTIVE,
        mi_start_epoch=MI_START_EPOCH,
        warmup_epochs=WARMUP_EPOCHS,
        use_contrastive=bool(USE_CONTRASTIVE and PAIR_MODE != 'none'),
        explicit_start_epoch=hyperparams.get('checkpoint_start_epoch'),
    )
    REPRESENTATION_PROBE_ENABLED = bool(
        hyperparams.get('representation_probe_enabled', False)
    )
    REPRESENTATION_PROBE_SUITE_ENABLED = bool(
        hyperparams.get('representation_probe_suite_enabled', False)
    )
    SAVE_ARTIFACTS_ENABLED = bool(
        hyperparams.get('instrumentation_save_artifacts', False)
    )
    REPRESENTATION_PROBE_SEED = int(
        hyperparams.get('representation_probe_seed', 700_001 + sim_idx)
    )
    if CHECKPOINT_START_EPOCH >= EPOCHS:
        raise ValueError(
            "checkpoint_start_epoch must be smaller than the training horizon: "
            f"start={CHECKPOINT_START_EPOCH}, epochs={EPOCHS}"
        )

    X_tr_s, T_tr_s, Y_tr_s, mu0_tr_s, mu1_tr_s = data_train
    X_te_s, T_te_s, Y_te_s, mu0_te_s, mu1_te_s = data_test

    num_treatments = infer_num_treatments(T_tr_s)

    n_total = X_tr_s.shape[0]
    validation_fraction = float(hyperparams.get('validation_fraction', 0.2))
    validation_criterion = hyperparams.get('validation_criterion', 'oracle_pehe')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be strictly between zero and one')
    if validation_criterion not in ('oracle_pehe', 'factual_huber'):
        raise ValueError('Unknown validation_criterion')
    n_val = int(validation_fraction * n_total)
    n_train = n_total - n_val

    rng = np.random.default_rng(sim_idx)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    config_hash = stable_config_hash(hyperparams)
    current_split_hash = split_index_hash(train_idx, val_idx)

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

    cross_fitted_mu0 = None
    cross_fitted_mu1 = None
    if PAIR_TEACHER_MODE == 'cross_fitted_ridge':
        cross_fitted_mu0, cross_fitted_mu1 = cross_fitted_ridge_potential_outcomes(
            x=X_train_processed,
            treatment=T_tr_s[train_idx],
            outcome=Y_tr_norm_all[train_idx],
            n_splits=int(hyperparams.get('cross_fit_folds', 5)),
            seed=80_000_003 + sim_idx,
        )

    max_y_obs_std = float(np.max(np.abs(Y_tr_norm_all)))
    clip_factor = hyperparams.get('outcome_clip_factor', 3.0)
    calculated_clip_val = max(max_y_obs_std * clip_factor, 3.0)

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
        feature_k=FEATURE_K,
        instrumentation_enabled=INSTRUMENTATION_ENABLED,
        include_pair_indices=True,
    )
    dl_train = DataLoader(ds_train, batch_size=None, shuffle=True)

    audit_pair_panel_seed = 410_000_003 + sim_idx
    audit_pair_panel = build_fixed_audit_pair_panel(
        n_units=n_train,
        panel_size=AUDIT_PAIR_PANEL_SIZE,
        seed=audit_pair_panel_seed,
    )
    current_audit_pair_panel_hash = audit_pair_panel_hash(audit_pair_panel)
    true_tau_train = mu1_tr_s[train_idx] - mu0_tr_s[train_idx]

    X_val_t = torch.tensor(X_val_processed, dtype=torch.float32).to(device)
    gt_val_ite = mu1_tr_s[val_idx] - mu0_tr_s[val_idx]

    input_dim = X_tr_s.shape[1]
    encoder = CATEEncoder(input_dim, LATENT_DIM, dropout=ENCODER_DROPOUT).to(device)
    predictor = OutcomeHead(
        LATENT_DIM,
        num_treatments=num_treatments,
        use_output_clip=USE_OUTPUT_CLIP,
        clip_val=calculated_clip_val,
        dropout=HEAD_DROPOUT,
    ).to(device)
    treat_clf = TreatmentClassifier(LATENT_DIM, num_treatments=num_treatments, dropout=CLF_DROPOUT).to(device)

    opt_main = optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=LR_MAIN, weight_decay=MAIN_WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=EPOCHS, eta_min=1e-6)
    opt_treat_clf = optim.AdamW(treat_clf.parameters(), lr=LR_TREAT_CLF, weight_decay=CLF_WD)

    crossfit_teachers = None
    crossfit_partition_audit = {}
    crossfit_base_seed = int(hyperparams.get('crossfit_base_seed', 120_000_007 + sim_idx))
    if PAIR_TEACHER_MODE == 'cross_fitted_teacher':
        crossfit_teachers, crossfit_partition_audit = build_persistent_crossfit_teachers(
            x_train_raw=X_tr_s[train_idx],
            treatment_train=T_tr_s[train_idx],
            outcome_train_raw=Y_tr_s[train_idx],
            n_splits=int(hyperparams.get('cross_fit_folds', 5)),
            base_seed=crossfit_base_seed,
            device=device,
            latent_dim=LATENT_DIM,
            encoder_dropout=ENCODER_DROPOUT,
            head_dropout=HEAD_DROPOUT,
            num_treatments=num_treatments,
            use_output_clip=USE_OUTPUT_CLIP,
            outcome_clip_factor=clip_factor,
            learning_rate=LR_MAIN,
            weight_decay=MAIN_WD,
            training_horizon=EPOCHS,
        )
        crossfit_partition_audit.update({
            'crossfit_validation_units_used_for_teacher_fit': 0,
            'crossfit_test_units_used_for_teacher_fit': 0,
            'crossfit_validation_test_excluded': True,
            'crossfit_teacher_architecture': 'CATEEncoder+OutcomeHead',
            'crossfit_teacher_loss': 'factual_supervised_smooth_l1_only',
            'crossfit_teacher_initialization': 'independent_seeded_no_main_weight_sharing',
            'crossfit_teacher_refresh': 'persistent_lockstep_predictions_at_standard_refresh_epochs',
        })

    # These hashes are audit-only. Cross-fit initialization is wrapped in an
    # isolated RNG context, so all three conditions must agree within a paired
    # replication before any training batch is consumed.
    main_initialization_hash = torch_module_state_hash(encoder, predictor, treat_clf)
    main_rng_hash_before_epoch0 = torch_rng_state_hash()
    pair_rng_hash_before_epoch0 = numpy_generator_state_hash(ds_train.rng)

    early_stopper = EarlyStoppingPEHE(patience=PATIENCE)
    final_val_pehe = 999.0
    static_pair_initialized = False

    last_epoch_loss_mi = 0.0
    last_epoch_loss_ctr = 0.0
    last_epoch_loss_sup = 0.0
    last_epoch_loss_total = 0.0
    best_epoch = -1
    best_epoch_any = -1
    best_val_any = float('inf')
    best_epoch_revised_window = -1
    best_val_revised_window = float('inf')
    n_mi_active_epochs_before_stop = 0
    refresh_id = 0
    previous_positive_mask = None
    refresh_diagnostics = []
    epoch_diagnostics = []
    mi_selection_diagnostics = []
    mi_gradient_diagnostics = []
    loss_trajectory = []
    best_active_support_indices = np.array([], dtype=np.int64)
    best_mi_endpoint_multiplicity = np.zeros(n_train, dtype=np.int64)
    best_sampled_positive_degree = np.zeros(n_train, dtype=np.int64)
    latest_teacher_audit = dict(crossfit_partition_audit)

    for epoch in range(EPOCHS):
        ds_train.set_epoch(epoch)
        epoch_mi_unit_indices = []
        epoch_all_pair_degree = np.zeros(n_train, dtype=np.int64)
        epoch_positive_pair_degree = np.zeros(n_train, dtype=np.int64)

        RAMP = 80
        lambda_ctr = 0.0 if (epoch < WARMUP_EPOCHS or not USE_CONTRASTIVE or PAIR_MODE == 'none') else min(ALPHA, (epoch - WARMUP_EPOCHS) / RAMP * ALPHA)

        if MI_MODE == MI_NO:
            lambda_mi = 0.0
        elif MI_USE_SCHEDULE:
            lambda_mi = 0.0 if epoch < MI_START_EPOCH else LAMBDA_MI_POS
        else:
            lambda_mi = LAMBDA_MI_POS

        if lambda_mi > 0:
            n_mi_active_epochs_before_stop += 1

        encoder.train()
        predictor.train()
        treat_clf.train()

        epoch_loss_mi = 0.0
        epoch_loss_ctr = 0.0
        epoch_loss_sup = 0.0
        epoch_loss_total = 0.0
        n_batches = 0
        n_optimization_batches = 0

        for batch_id, batch in enumerate(dl_train):
            x1, y1, t1, x2, y2, t2, label, idx_a, idx_b = [b.to(device) for b in batch]

            if x1.shape[0] == 0:
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
                loss_sup_value = float(loss_sup.item())
                epoch_loss_sup += loss_sup_value
                epoch_loss_total += loss_sup_value
                n_optimization_batches += 1
                continue

            n_batches += 1
            label = label.float()
            t1_idx = t1.view(-1).long()
            t2_idx = t2.view(-1).long()
            y1_r = y1.view(-1, 1)
            y2_r = y2.view(-1, 1)

            z1 = encoder(x1)
            z2 = encoder(x2)
            pos_mask = (label == 1)
            balanced_positions = None
            if MI_MODE == MI_BALANCED_LOCAL and lambda_mi > 0:
                balanced_positions = balanced_endpoint_positions(
                    torch.cat((t1_idx[pos_mask], t2_idx[pos_mask])),
                    sim_idx, epoch, batch_id,
                )
            idx_a_np = idx_a.detach().cpu().numpy().astype(np.int64)
            idx_b_np = idx_b.detach().cpu().numpy().astype(np.int64)
            pos_mask_np = pos_mask.detach().cpu().numpy().astype(bool)
            np.add.at(epoch_all_pair_degree, idx_a_np, 1)
            np.add.at(epoch_all_pair_degree, idx_b_np, 1)
            np.add.at(epoch_positive_pair_degree, idx_a_np[pos_mask_np], 1)
            np.add.at(epoch_positive_pair_degree, idx_b_np[pos_mask_np], 1)

            independent_unit_indices = None
            z_independent = None
            t_independent = None
            if lambda_mi > 0 and MI_MODE == MI_TRUE_GLOBAL:
                independent_unit_indices = sample_independent_unit_indices(
                    n_units=n_train,
                    batch_size=BATCH_SIZE,
                    base_seed=sim_idx,
                    epoch=epoch,
                    batch_id=batch_id,
                )
                independent_x = torch.tensor(
                    X_train_processed[independent_unit_indices],
                    dtype=torch.float32,
                    device=device,
                )
                t_independent = torch.tensor(
                    T_tr_s[train_idx][independent_unit_indices],
                    dtype=torch.float32,
                    device=device,
                ).view(-1).long()
                z_independent = encoder(independent_x)

            z_mi_det = None
            t_mi_det = None
            if lambda_mi > 0:
                z_mi_det, t_mi_det, _ = select_mi_subset(
                    MI_MODE,
                    z1.detach(),
                    z2.detach(),
                    t1_idx,
                    t2_idx,
                    pos_mask,
                    idx_a=idx_a,
                    idx_b=idx_b,
                    z_independent=z_independent.detach() if z_independent is not None else None,
                    t_independent=t_independent,
                    independent_unit_indices=independent_unit_indices,
                    balanced_positions=balanced_positions,
                )
            diagnose_mi_batch = bool(
                MI_GRADIENT_DIAGNOSTICS
                and len(mi_gradient_diagnostics) < MI_GRADIENT_MAX_BATCHES
                and lambda_mi > 0
                and z_mi_det is not None
                and z_mi_det.shape[0] > POS_MIN_COUNT
            )
            if diagnose_mi_batch:
                clf_ce_before, clf_acc_before = _diagnostic_classifier_scores(
                    treat_clf, z_mi_det, t_mi_det, num_treatments,
                    balanced_weighted=MI_MODE == MI_WEIGHTED_BALANCED_LOCAL,
                )
            if lambda_mi > 0 and z_mi_det is not None and z_mi_det.shape[0] > POS_MIN_COUNT:
                for _ in range(TREAT_CLF_STEPS):
                    opt_treat_clf.zero_grad()
                    if MI_MODE == MI_WEIGHTED_BALANCED_LOCAL:
                        clf_loss = balanced_selected_support_ce(treat_clf(z_mi_det), t_mi_det)
                    else:
                        clf_loss = treatment_classifier_loss(treat_clf, z_mi_det, t_mi_det, num_treatments)
                    clf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(treat_clf.parameters(), CLIP_NORM)
                    opt_treat_clf.step()
            if diagnose_mi_batch:
                clf_ce_after, clf_acc_after = _diagnostic_classifier_scores(
                    treat_clf, z_mi_det, t_mi_det, num_treatments,
                    balanced_weighted=MI_MODE == MI_WEIGHTED_BALANCED_LOCAL,
                )
                balanced_audit_extra = {}
                if hyperparams.get('instrumentation_balanced_mi_audit', False):
                    original_t = torch.cat((t1_idx[pos_mask], t2_idx[pos_mask]))
                    n0 = int((t_mi_det == 0).sum().item())
                    n1 = int((t_mi_det == 1).sum().item())
                    weighted = MI_MODE == MI_WEIGHTED_BALANCED_LOCAL
                    balanced_audit_extra = {
                        'treated_fraction_before': float(original_t.float().mean().item()),
                        'n_original_endpoints': int(original_t.numel()),
                        'n_class0_endpoints': n0,
                        'n_class1_endpoints': n1,
                        'class0_total_weight': 0.5 if weighted else n0 / (n0 + n1),
                        'class1_total_weight': 0.5 if weighted else n1 / (n0 + n1),
                        'class0_per_endpoint_weight': 0.5 / n0 if weighted else 1.0 / (n0 + n1),
                        'class1_per_endpoint_weight': 0.5 / n1 if weighted else 1.0 / (n0 + n1),
                        **_diagnostic_binary_classifier_extra(treat_clf, z_mi_det, t_mi_det),
                    }

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
            z_mi = None
            t_mi = None
            mi_unit_indices = np.array([], dtype=np.int64)
            if lambda_mi > 0:
                z_mi, t_mi, mi_unit_indices = select_mi_subset(
                    MI_MODE,
                    z1,
                    z2,
                    t1_idx,
                    t2_idx,
                    pos_mask,
                    idx_a=idx_a,
                    idx_b=idx_b,
                    z_independent=z_independent,
                    t_independent=t_independent,
                    independent_unit_indices=independent_unit_indices,
                    balanced_positions=balanced_positions,
                )
            if lambda_mi > 0 and z_mi is not None and z_mi.shape[0] > POS_MIN_COUNT:
                if MI_MODE == MI_WEIGHTED_BALANCED_LOCAL:
                    mi_lb = z_mi.new_tensor(np.log(2.0)) - balanced_selected_support_ce(
                        treat_clf(z_mi), t_mi
                    )
                elif MI_MODE == MI_BALANCED_LOCAL:
                    # Diagnostic balanced selected-support adversarial objective;
                    # this is not population mutual information.
                    mi_lb = z_mi.new_tensor(np.log(2.0)) - treatment_classifier_loss(
                        treat_clf, z_mi, t_mi, num_treatments
                    )
                else:
                    mi_lb = variational_mi_lower_bound(treat_clf, z_mi, t_mi, num_treatments)
                loss_mi = torch.clamp(mi_lb, -5.0, 5.0)
                epoch_mi_unit_indices.append(np.asarray(mi_unit_indices, dtype=np.int64))

            loss_main = loss_sup + lambda_ctr * loss_ctr + lambda_mi * loss_mi
            if diagnose_mi_batch:
                encoder_parameters = tuple(encoder.parameters())
                grad_sup = _diagnostic_encoder_gradient_norm(
                    loss_sup, encoder_parameters
                )
                grad_ctr = _diagnostic_encoder_gradient_norm(
                    lambda_ctr * loss_ctr, encoder_parameters
                )
                grad_mi = _diagnostic_encoder_gradient_norm(
                    lambda_mi * loss_mi, encoder_parameters
                )
                grad_total = _diagnostic_encoder_gradient_norm(
                    loss_main, encoder_parameters
                )
                mi_entropy = empirical_entropy_from_labels(
                    t_mi.detach(), num_treatments
                )
                if MI_MODE == MI_WEIGHTED_BALANCED_LOCAL:
                    mi_entropy = z_mi.new_tensor(np.log(2.0))
                mi_unit_indices_np = np.asarray(mi_unit_indices, dtype=np.int64)
                n_mi_endpoints = int(mi_unit_indices_np.size)
                n_unique_mi_endpoints = int(np.unique(mi_unit_indices_np).size)
                raw_mi = float(mi_lb.detach().item())
                mi_gradient_diagnostics.append({
                    **balanced_audit_extra,
                    'replication_id': int(sim_idx),
                    'epoch': int(epoch),
                    'batch_id': int(batch_id),
                    'loss_sup': float(loss_sup.detach().item()),
                    'loss_ctr': float(loss_ctr.detach().item()),
                    'loss_mi': float(loss_mi.detach().item()),
                    'loss_total': float(loss_main.detach().item()),
                    'lambda_ctr': float(lambda_ctr),
                    'lambda_mi': float(lambda_mi),
                    'grad_sup': grad_sup,
                    'grad_ctr_weighted': grad_ctr,
                    'grad_mi_weighted': grad_mi,
                    'grad_total': grad_total,
                    'ratio_mi_sup': _diagnostic_ratio(grad_mi, grad_sup),
                    'ratio_mi_ctr': _diagnostic_ratio(grad_mi, grad_ctr),
                    'ratio_ctr_sup': _diagnostic_ratio(grad_ctr, grad_sup),
                    'clf_ce_before': clf_ce_before,
                    'clf_ce_after': clf_ce_after,
                    'clf_acc_before': clf_acc_before,
                    'clf_acc_after': clf_acc_after,
                    'mi_entropy': float(mi_entropy.item()),
                    'mi_ce': float(mi_entropy.item()) - raw_mi,
                    'mi_raw_objective': raw_mi,
                    'mi_clipped_objective': float(loss_mi.detach().item()),
                    'mi_was_clipped': bool(raw_mi < -5.0 or raw_mi > 5.0),
                    'n_mi_endpoints': n_mi_endpoints,
                    'n_unique_mi_endpoints': n_unique_mi_endpoints,
                    'duplicate_fraction': (
                        1.0 - n_unique_mi_endpoints / n_mi_endpoints
                    ),
                    'treated_fraction': float(t_mi.detach().float().mean().item()),
                })
            loss_main.backward()
            main_preclip_norm = torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), CLIP_NORM)
            opt_main.step()
            if diagnose_mi_batch and hyperparams.get('instrumentation_balanced_mi_audit', False):
                mi_gradient_diagnostics[-1]['main_gradient_was_clipped'] = bool(
                    float(main_preclip_norm) > CLIP_NORM
                )
            if (diagnose_mi_batch and hyperparams.get('instrumentation_mi_stop_after_batches', False)
                    and len(mi_gradient_diagnostics) >= MI_GRADIENT_MAX_BATCHES):
                return {
                    '_mi_gradient_diagnostics': mi_gradient_diagnostics,
                    'split_hash': current_split_hash,
                    'main_initialization_hash': main_initialization_hash,
                    'main_rng_hash_before_epoch0': main_rng_hash_before_epoch0,
                    'final_model_state_hash': torch_module_state_hash(encoder, predictor, treat_clf),
                }

            for p in treat_clf.parameters():
                p.requires_grad = True

            epoch_loss_mi += float(loss_mi.item())
            epoch_loss_ctr += float(loss_ctr.item())
            epoch_loss_sup += float(loss_sup.item())
            epoch_loss_total += float(loss_main.item())
            n_optimization_batches += 1

        scheduler.step()
        if crossfit_teachers is not None:
            train_crossfit_teachers_one_epoch(
                teachers=crossfit_teachers,
                treatment_train=T_tr_s[train_idx],
                epoch=epoch,
                base_seed=crossfit_base_seed,
                batch_size=BATCH_SIZE,
                device=device,
                huber_beta=HUBER_BETA,
                clip_norm=CLIP_NORM,
            )
        last_epoch_loss_mi = epoch_loss_mi / max(1, n_batches)
        last_epoch_loss_ctr = epoch_loss_ctr / max(1, n_batches)
        last_epoch_loss_sup = epoch_loss_sup / max(1, n_optimization_batches)
        last_epoch_loss_total = epoch_loss_total / max(1, n_optimization_batches)

        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            z_val = encoder(X_val_t)
            mu_val_all = predictor(z_val)
            if validation_criterion == 'oracle_pehe':
                ite_val_pred_norm = (mu_val_all[:, 1] - mu_val_all[:, 0]).cpu().numpy().ravel()
                ite_val_pred = ite_val_pred_norm * y_std
                val_pehe = np.sqrt(np.mean((gt_val_ite - ite_val_pred) ** 2))
                val_score = val_pehe
            else:
                # Synthetic diagnostic only: factual validation with the same
                # normalized-outcome Huber loss as training; no oracle selection.
                validation_t = torch.as_tensor(T_tr_s[val_idx], dtype=torch.long, device=device)
                validation_y = torch.as_tensor(Y_tr_norm_all[val_idx], dtype=torch.float32, device=device)
                factual_pred = mu_val_all.gather(1, validation_t.reshape(-1, 1)).reshape(-1)
                val_score = float(F.smooth_l1_loss(
                    factual_pred, validation_y.reshape(-1), beta=HUBER_BETA
                ).item())
                val_pehe = float('nan')

        if not np.isfinite(val_score):
            val_score = 999.0
            if validation_criterion == 'oracle_pehe':
                val_pehe = val_score

        if val_score < best_val_any - early_stopper.min_delta:
            best_val_any = float(val_score)
            best_epoch_any = int(epoch)
        if (
            epoch >= REVISED_REFERENCE_START_EPOCH
            and val_score < best_val_revised_window - early_stopper.min_delta
        ):
            best_val_revised_window = float(val_score)
            best_epoch_revised_window = int(epoch)

        active_support_indices = ds_train.sampled_active_positive_support_indices()
        if epoch_mi_unit_indices:
            selected_mi_indices = np.concatenate(epoch_mi_unit_indices)
        else:
            selected_mi_indices = np.array([], dtype=np.int64)
        if MI_MODE == MI_CURRENT_GLOBAL:
            relevant_pair_degree = epoch_all_pair_degree
        else:
            relevant_pair_degree = epoch_positive_pair_degree
        current_mi_multiplicity = np.zeros(n_train, dtype=np.int64)
        if lambda_mi > 0:
            mi_row, current_mi_multiplicity = summarize_mi_selection(
                selected_unit_indices=selected_mi_indices,
                n_train=n_train,
                treatment=T_tr_s[train_idx],
                method=MI_MODE,
                tau_true=true_tau_train,
                pair_degree=relevant_pair_degree,
            )
            mi_selection_diagnostics.append({
                "dataset": "IHDP",
                "replication_id": int(sim_idx),
                "data_seed": int(sim_idx),
                "model_seed": int(sim_idx),
                "epoch": int(epoch),
                "method": MI_MODE,
                "mi_pair_degree_definition": (
                    "all_sampled_pair_endpoint_degree"
                    if MI_MODE == MI_CURRENT_GLOBAL
                    else "sampled_positive_pair_endpoint_degree"
                ),
                "config_hash": config_hash,
                "split_hash": current_split_hash,
                **mi_row,
            })

        checkpoint_eligible = checkpoint_is_eligible(epoch, CHECKPOINT_START_EPOCH)
        if checkpoint_eligible:
            previous_best_pehe = early_stopper.best_pehe
            early_stopper(val_score, encoder, predictor)
            if early_stopper.best_pehe < previous_best_pehe:
                best_epoch = epoch
                best_active_support_indices = active_support_indices.copy()
                best_mi_endpoint_multiplicity = current_mi_multiplicity.copy()
                best_sampled_positive_degree = epoch_positive_pair_degree.copy()
            final_val_pehe = early_stopper.best_pehe

        sampled_support = (
            ds_train.sampled_active_positive_support_fraction()
            if INSTRUMENTATION_ENABLED
            else float("nan")
        )
        loss_row = {
            "epoch": int(epoch),
            "train_loss_sup": last_epoch_loss_sup,
            "train_loss_ctr": last_epoch_loss_ctr,
            "train_loss_mi": last_epoch_loss_mi,
            "train_loss_total": last_epoch_loss_total,
            "lambda_ctr": float(lambda_ctr),
            "lambda_mi": float(lambda_mi),
            "val_pehe": float(val_pehe),
            "validation_score": float(val_score),
            "validation_criterion": validation_criterion,
        }
        loss_trajectory.append(loss_row)
        if INSTRUMENTATION_ENABLED:
            epoch_diagnostics.append({
                "dataset": "IHDP",
                "replication_id": int(sim_idx),
                "data_seed": int(sim_idx),
                "model_seed": int(sim_idx),
                "config_hash": config_hash,
                "split_hash": current_split_hash,
                **loss_row,
                "sampled_active_positive_support_fraction": sampled_support,
                "checkpoint_protocol": CHECKPOINT_PROTOCOL,
                "checkpoint_start_epoch": int(CHECKPOINT_START_EPOCH),
                "checkpoint_eligible": bool(checkpoint_eligible),
            })

        if early_stopper.early_stop:
            break

        should_update = (epoch >= WARMUP_EPOCHS and ITE_UPDATE_FREQ > 0 and (epoch % ITE_UPDATE_FREQ == 0))
        if should_update and PAIR_MODE in ['dynamic_ite', 'static_ite']:
            with torch.no_grad():
                x_tr_curr = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
                z_full = encoder(x_tr_curr)
                mu_full = predictor(z_full)
                new_mu0 = mu_full[:, 0].cpu().numpy().ravel()
                new_mu1 = mu_full[:, 1].cpu().numpy().ravel()

                did_refresh = False
                teacher_refresh_audit = {}
                if PAIR_MODE == 'dynamic_ite' and USE_DYNAMIC_UPDATE:
                    if PAIR_TEACHER_MODE in {'self_updating', 'standard_artemis'}:
                        ds_train.update_ite_estimates(new_mu0, new_mu1)
                        did_refresh = True
                    elif PAIR_TEACHER_MODE == 'cross_fitted_ridge':
                        ds_train.update_ite_estimates(cross_fitted_mu0, cross_fitted_mu1)
                        did_refresh = True
                    elif PAIR_TEACHER_MODE == 'cross_fitted_teacher':
                        oof_mu0, oof_mu1, teacher_refresh_audit = (
                            predict_crossfit_potential_outcomes(
                                teachers=crossfit_teachers,
                                n_training_units=n_train,
                                main_y_mean=y_mean,
                                main_y_std=y_std,
                                device=device,
                            )
                        )
                        ds_train.update_ite_estimates(oof_mu0, oof_mu1)
                        latest_teacher_audit = {
                            **crossfit_partition_audit,
                            **teacher_refresh_audit,
                        }
                        did_refresh = True
                    elif PAIR_TEACHER_MODE in {'oracle', 'oracle_pairing'}:
                        ds_train.update_ite_estimates(
                            (mu0_tr_s[train_idx] - y_mean) / y_std,
                            (mu1_tr_s[train_idx] - y_mean) / y_std,
                        )
                        did_refresh = True
                    elif PAIR_TEACHER_MODE == 'frozen_after_warmup' and not static_pair_initialized:
                        ds_train.update_ite_estimates(new_mu0, new_mu1)
                        static_pair_initialized = True
                        did_refresh = True
                elif PAIR_MODE == 'static_ite' and not static_pair_initialized:
                    ds_train.update_ite_estimates(new_mu0, new_mu1)
                    static_pair_initialized = True
                    did_refresh = True

                if did_refresh and INSTRUMENTATION_ENABLED:
                    refresh_row, previous_positive_mask = compute_refresh_diagnostics(
                        dataset=ds_train,
                        audit_pair_panel=audit_pair_panel,
                        true_tau=true_tau_train,
                        y_std=y_std,
                        epoch=epoch,
                        refresh_id=refresh_id,
                        previous_positive_mask=previous_positive_mask,
                        val_metric=val_score,
                        active_support_indices=active_support_indices,
                        local_mi_unique_indices=(
                            np.flatnonzero(current_mi_multiplicity > 0)
                            if MI_MODE == MI_CURRENT_LOCAL else None
                        ),
                    )
                    refresh_row.update({
                        "dataset": "IHDP",
                        "replication_id": int(sim_idx),
                        "data_seed": int(sim_idx),
                        "model_seed": int(sim_idx),
                        "config_hash": config_hash,
                        "split_hash": current_split_hash,
                        "audit_pair_panel_seed": int(audit_pair_panel_seed),
                        "audit_pair_panel_hash": current_audit_pair_panel_hash,
                        "pair_teacher_mode": PAIR_TEACHER_MODE,
                        "pair_teacher_diagnostic_label": (
                            "ORACLE DIAGNOSTIC - NOT A REALISTIC METHOD"
                            if PAIR_TEACHER_MODE in {'oracle', 'oracle_pairing'}
                            else PAIR_TEACHER_MODE
                        ),
                        **latest_teacher_audit,
                    })
                    refresh_diagnostics.append(refresh_row)
                    refresh_id += 1

    early_stopper.restore_best_weights(encoder, predictor)
    final_model_state_hash = torch_module_state_hash(encoder, predictor, treat_clf)
    encoder.eval()
    predictor.eval()

    X_train_t = torch.as_tensor(X_train_processed, dtype=torch.float32, device=device)
    X_val_t_best = torch.as_tensor(X_val_processed, dtype=torch.float32, device=device)
    X_te_t = torch.as_tensor(X_test_processed, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_train_best = encoder(X_train_t).cpu().numpy()
        z_val_best = encoder(X_val_t_best).cpu().numpy()
        z_test_best = encoder(X_te_t).cpu().numpy()
        mu_train_norm_best = predictor(torch.as_tensor(z_train_best, dtype=torch.float32, device=device)).cpu().numpy()
        mu_val_norm_best = predictor(torch.as_tensor(z_val_best, dtype=torch.float32, device=device)).cpu().numpy()
        mu_test_norm_best = predictor(torch.as_tensor(z_test_best, dtype=torch.float32, device=device)).cpu().numpy()

    mu_train_best = mu_train_norm_best * y_std + y_mean
    mu_val_best = mu_val_norm_best * y_std + y_mean
    mu_test_best = mu_test_norm_best * y_std + y_mean
    ite_pred = (mu_test_norm_best[:, 1] - mu_test_norm_best[:, 0]) * y_std

    probe_results = []
    probe_metrics = {
        'treatment_probe_auc': float('nan'),
        'effect_probe_r2': float('nan'),
        'treatment_probe_accuracy': float('nan'),
        'treatment_probe_cross_entropy': float('nan'),
        'effect_probe_mae': float('nan'),
        'semantic_alignment': float('nan'),
        'representation_probe_protocol': 'disabled',
        'treatment_probe_model': 'disabled',
        'effect_probe_model': 'disabled',
        'representation_probe_seed': int(REPRESENTATION_PROBE_SEED),
        'representation_probe_train_n': 0,
        'representation_probe_eval_n': 0,
    }
    if REPRESENTATION_PROBE_SUITE_ENABLED:
        probe_results, suite_summary = evaluate_probe_suite(
            z_train=z_train_best,
            treatment_train=T_tr_s[train_idx],
            z_validation=z_val_best,
            treatment_validation=T_tr_s[val_idx],
            z_test=z_test_best,
            treatment_test=T_te_s,
            seed=REPRESENTATION_PROBE_SEED,
            tau_train=true_tau_train,
            tau_validation=gt_val_ite,
            tau_test=mu1_te_s - mu0_te_s,
        )
        probe_metrics.update({
            'treatment_probe_auc': suite_summary.get('treatment_probe_AUC', float('nan')),
            'effect_probe_r2': suite_summary.get('effect_probe_R2', float('nan')),
            'treatment_probe_accuracy': suite_summary.get('treatment_probe_accuracy', float('nan')),
            'treatment_probe_cross_entropy': suite_summary.get('treatment_probe_cross_entropy', float('nan')),
            'effect_probe_mae': suite_summary.get('effect_probe_MAE', float('nan')),
            'semantic_alignment': suite_summary.get('semantic_alignment', float('nan')),
            'representation_probe_protocol': 'fixed_train_validation_test_splits_full_probe_suite',
            'treatment_probe_model': 'linear+shallow_mlp+strong_mlp',
            'effect_probe_model': 'StandardScaler+Ridge(alpha=1.0)',
            'representation_probe_train_n': int(z_train_best.shape[0]),
            'representation_probe_eval_n': int(z_test_best.shape[0]),
        })
    elif REPRESENTATION_PROBE_ENABLED:
        probe_metrics = evaluate_frozen_representation_probes(
            encoder=encoder,
            x_probe_train=X_train_processed,
            treatment_probe_train=T_tr_s[train_idx],
            tau_probe_train=true_tau_train,
            x_probe_eval=X_test_processed,
            treatment_probe_eval=T_te_s,
            tau_probe_eval=mu1_te_s - mu0_te_s,
            device=device,
            probe_seed=REPRESENTATION_PROBE_SEED,
        )

    final_pair_arrays = {
        'graph_support_mask': np.zeros(n_train, dtype=bool),
        'positive_pair_degree': np.zeros(n_train, dtype=np.int64),
        'audit_idx_a': audit_pair_panel[0],
        'audit_idx_b': audit_pair_panel[1],
        'predicted_positive_mask': np.zeros(audit_pair_panel[0].size, dtype=bool),
        'pseudo_effect_distance': np.asarray([], dtype=np.float64),
        'true_effect_distance': np.asarray([], dtype=np.float64),
        'oracle_positive_mask': np.asarray([], dtype=bool),
    }
    if SAVE_ARTIFACTS_ENABLED:
        final_threshold_norm = compute_tau_threshold(
            mu_train_norm_best[:, 0],
            mu_train_norm_best[:, 1],
            perc=PERC_THR,
            rng=np.random.default_rng(910_000_019 + sim_idx),
        )
        _, final_pair_arrays = compute_pair_audit(
            pseudo_tau=(mu_train_norm_best[:, 1] - mu_train_norm_best[:, 0]) * y_std,
            treatment=T_tr_s[train_idx],
            panel=audit_pair_panel,
            predicted_threshold=final_threshold_norm * y_std,
            percentile=PERC_THR,
            true_tau=true_tau_train,
            active_support_indices=best_active_support_indices,
            local_mi_unique_indices=(
                np.flatnonzero(best_mi_endpoint_multiplicity > 0)
                if MI_MODE == MI_CURRENT_LOCAL else None
            ),
        )

    predictions_artifact = {
        'indices': np.arange(X_te_s.shape[0], dtype=np.int64),
        'T': T_te_s,
        'Y': Y_te_s,
        'mu0_hat': mu_test_best[:, 0],
        'mu1_hat': mu_test_best[:, 1],
        'tau_hat': ite_pred,
        'tau_true': mu1_te_s - mu0_te_s,
    }
    embeddings_artifact = {
        'train_indices': train_idx,
        'validation_indices': val_idx,
        'test_indices': np.arange(X_te_s.shape[0], dtype=np.int64),
        'X_train': X_train_processed,
        'X_validation': X_val_processed,
        'X_test': X_test_processed,
        'T_train': T_tr_s[train_idx],
        'T_validation': T_tr_s[val_idx],
        'T_test': T_te_s,
        'Y_train': Y_tr_s[train_idx],
        'Y_validation': Y_tr_s[val_idx],
        'Y_test': Y_te_s,
        'Z_train': z_train_best,
        'Z_validation': z_val_best,
        'Z_test': z_test_best,
        'mu0_hat_train': mu_train_best[:, 0],
        'mu1_hat_train': mu_train_best[:, 1],
        'mu0_hat_validation': mu_val_best[:, 0],
        'mu1_hat_validation': mu_val_best[:, 1],
        'tau_true_train': true_tau_train,
        'tau_true_validation': gt_val_ite,
        'graph_support_membership': final_pair_arrays['graph_support_mask'],
        'support_membership': final_pair_arrays['graph_support_mask'],
        'active_support_membership': np.isin(np.arange(n_train), best_active_support_indices),
        'positive_pair_degree': final_pair_arrays['positive_pair_degree'],
        'sampled_positive_pair_degree': best_sampled_positive_degree,
        'mi_endpoint_multiplicity': best_mi_endpoint_multiplicity,
        'audit_idx_a': final_pair_arrays['audit_idx_a'],
        'audit_idx_b': final_pair_arrays['audit_idx_b'],
        'audit_predicted_positive_mask': final_pair_arrays['predicted_positive_mask'],
        'audit_pseudo_effect_distance': final_pair_arrays['pseudo_effect_distance'],
        'audit_true_effect_distance': final_pair_arrays['true_effect_distance'],
        'audit_oracle_positive_mask': final_pair_arrays['oracle_positive_mask'],
    }

    y_true_te = np.stack([mu0_te_s, mu1_te_s], axis=1)
    pehe = sqrt_PEHE_with_diff(y_true_te, ite_pred)
    ate_err = eps_ATE_diff(mu1_te_s - mu0_te_s, ite_pred)
    runtime_seconds = time.perf_counter() - run_started_at

    return {
        'val_pehe': final_val_pehe if validation_criterion == 'oracle_pehe' else float('nan'),
        'best_val_metric': float(final_val_pehe),
        'validation_criterion': validation_criterion,
        'test_pehe': pehe,
        'ate_err': ate_err,
        'epochs': epoch + 1,
        'best_epoch': int(best_epoch),
        'best_epoch_any': int(best_epoch_any),
        'best_val_any': float(best_val_any),
        'best_epoch_revised_window': int(best_epoch_revised_window),
        'best_val_revised_window': float(best_val_revised_window),
        'best_val_pehe': float(final_val_pehe) if validation_criterion == 'oracle_pehe' else float('nan'),
        'last_epoch': int(epoch),
        'mi_start_epoch': int(MI_START_EPOCH),
        'checkpoint_protocol': CHECKPOINT_PROTOCOL,
        'checkpoint_start_epoch': int(CHECKPOINT_START_EPOCH),
        'n_mi_active_epochs_before_stop': int(n_mi_active_epochs_before_stop),
        'best_epoch_before_mi_start': bool(best_epoch < MI_START_EPOCH),
        'runtime_seconds': float(runtime_seconds),
        'config_hash': config_hash,
        'split_hash': current_split_hash,
        'final_model_state_hash': final_model_state_hash,
        'instrumentation_enabled': bool(INSTRUMENTATION_ENABLED),
        'pair_teacher_mode': PAIR_TEACHER_MODE,
        'main_initialization_hash': main_initialization_hash,
        'main_rng_hash_before_epoch0': main_rng_hash_before_epoch0,
        'pair_rng_hash_before_epoch0': pair_rng_hash_before_epoch0,
        **latest_teacher_audit,
        'last_epoch_loss_mi': last_epoch_loss_mi,
        'last_epoch_loss_ctr': last_epoch_loss_ctr,
        'last_epoch_loss_sup': last_epoch_loss_sup,
        'last_epoch_loss_total': last_epoch_loss_total,
        **probe_metrics,
        '_loss_trajectory': loss_trajectory,
        '_refresh_diagnostics': refresh_diagnostics,
        '_epoch_diagnostics': epoch_diagnostics,
        '_mi_selection_diagnostics': mi_selection_diagnostics,
        '_mi_gradient_diagnostics': mi_gradient_diagnostics,
        '_probe_results': probe_results,
        '_predictions_artifact': predictions_artifact,
        '_embeddings_artifact': embeddings_artifact,
    }

# ==============================================================================
# EXPERIMENT RUNNERS
# ==============================================================================
_GIT_COMMIT_CACHE = None


def current_git_commit() -> str:
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is None:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        _GIT_COMMIT_CACHE = completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"
    return _GIT_COMMIT_CACHE


def subset_num_sims(X_tr):
    total = X_tr.shape[-1] if X_tr.ndim == 3 else 1
    return min(N_SIMS, total)


def run_setting(
    setting_name: str,
    params: Dict[str, Any],
    X_tr,
    T_tr,
    YF_tr,
    mu0_tr,
    mu1_tr,
    X_te,
    T_te,
    YF_te,
    mu0_te,
    mu1_te,
    n_sims: int,
    return_diagnostics: bool = False,
):
    total_avail = X_tr.shape[-1] if X_tr.ndim == 3 else 1
    n_sims = min(n_sims, total_avail)

    print("\n" + "=" * 80)
    print(f"RUNNING SETTING: {setting_name}")
    print("=" * 80)

    results_storage = []
    refresh_storage = []
    epoch_storage = []
    mi_selection_storage = []
    mi_gradient_storage = []
    probe_storage = []
    for i in range(n_sims):
        if X_tr.ndim == 3:
            train_data = (X_tr[:, :, i], T_tr[:, i], YF_tr[:, i], mu0_tr[:, i], mu1_tr[:, i])
            test_data = (X_te[:, :, i], T_te[:, i], YF_te[:, i], mu0_te[:, i], mu1_te[:, i])
        else:
            train_data = (X_tr, T_tr, YF_tr, mu0_tr, mu1_tr)
            test_data = (X_te, T_te, YF_te, mu0_te, mu1_te)

        res = train_single_simulation(i, train_data, test_data, DEVICE, params)
        scenario = str(params.get('instrumentation_scenario', 'canonical'))
        run_id = make_run_id(
            dataset='IHDP',
            scenario=scenario,
            replication_id=i,
            model_seed=i,
            method=setting_name,
            config_hash=res['config_hash'],
        )
        prediction_artifact_path = None
        embedding_artifact_path = None
        if params.get('instrumentation_save_artifacts', False):
            prediction_artifact_path, embedding_artifact_path = save_run_artifacts(
                output_dir=Path(OUT_DIR),
                run_id=run_id,
                predictions=res['_predictions_artifact'],
                embeddings=res['_embeddings_artifact'],
            )

        row = normalize_per_run_row({
            'dataset': 'IHDP',
            'scenario': scenario,
            'replication_id': i,
            'data_seed': i,
            'model_seed': i,
            'method': setting_name,
            'run_id': run_id,
            'git_commit': current_git_commit(),
            'setting': setting_name,
            'sim_id': i,
            'test_pehe': res['test_pehe'],
            'ate_err': res['ate_err'],
            'PEHE': res['test_pehe'],
            'ATE_error': res['ate_err'],
            'epochs': res['epochs'],
            'best_epoch': res['best_epoch'],
            'best_epoch_any': res['best_epoch_any'],
            'best_epoch_revised_window': res['best_epoch_revised_window'],
            'best_val_pehe': res['best_val_pehe'],
            'val_PEHE': res['best_val_pehe'],
            'last_epoch': res['last_epoch'],
            'mi_start_epoch': res['mi_start_epoch'],
            'checkpoint_protocol': res['checkpoint_protocol'],
            'checkpoint_start_epoch': res['checkpoint_start_epoch'],
            'n_mi_active_epochs_before_stop': res['n_mi_active_epochs_before_stop'],
            'n_mi_active_epochs': res['n_mi_active_epochs_before_stop'],
            'early_stopped': bool(res['epochs'] < int(params.get('epochs', 400))),
            'best_epoch_before_mi_start': res['best_epoch_before_mi_start'],
            'runtime_seconds': res['runtime_seconds'],
            'config_hash': res['config_hash'],
            'split_hash': res['split_hash'],
            'final_model_state_hash': res['final_model_state_hash'],
            'val_pehe_early_stop': res['val_pehe'],
            'best_val_metric': res['val_pehe'],
            'test_metric': res['test_pehe'],
            'last_epoch_loss_mi': res['last_epoch_loss_mi'],
            'last_epoch_loss_ctr': res['last_epoch_loss_ctr'],
            'last_epoch_loss_sup': res['last_epoch_loss_sup'],
            'last_epoch_loss_total': res['last_epoch_loss_total'],
            'treatment_probe_auc': res['treatment_probe_auc'],
            'treatment_probe_AUC': res['treatment_probe_auc'],
            'effect_probe_r2': res['effect_probe_r2'],
            'effect_probe_R2': res['effect_probe_r2'],
            'treatment_probe_accuracy': res.get('treatment_probe_accuracy', float('nan')),
            'treatment_probe_cross_entropy': res.get('treatment_probe_cross_entropy', float('nan')),
            'effect_probe_MAE': res.get('effect_probe_mae', float('nan')),
            'semantic_alignment': res.get('semantic_alignment', float('nan')),
            'representation_probe_protocol': res['representation_probe_protocol'],
            'treatment_probe_model': res['treatment_probe_model'],
            'effect_probe_model': res['effect_probe_model'],
            'representation_probe_seed': res['representation_probe_seed'],
            'representation_probe_train_n': res['representation_probe_train_n'],
            'representation_probe_eval_n': res['representation_probe_eval_n'],
            'pair_teacher_mode': res.get('pair_teacher_mode'),
            'main_initialization_hash': res.get('main_initialization_hash'),
            'main_rng_hash_before_epoch0': res.get('main_rng_hash_before_epoch0'),
            'pair_rng_hash_before_epoch0': res.get('pair_rng_hash_before_epoch0'),
            'crossfit_folds': res.get('crossfit_folds', np.nan),
            'crossfit_fold_seed': res.get('crossfit_fold_seed', np.nan),
            'crossfit_oof_coverage_fraction': res.get(
                'crossfit_oof_coverage_fraction', np.nan
            ),
            'crossfit_oof_exactly_once_fraction': res.get(
                'crossfit_oof_exactly_once_fraction', np.nan
            ),
            'crossfit_oof_exactly_once': res.get('crossfit_oof_exactly_once', np.nan),
            'crossfit_no_fit_holdout_overlap': res.get(
                'crossfit_no_fit_holdout_overlap', np.nan
            ),
            'crossfit_validation_units_used_for_teacher_fit': res.get(
                'crossfit_validation_units_used_for_teacher_fit', np.nan
            ),
            'crossfit_test_units_used_for_teacher_fit': res.get(
                'crossfit_test_units_used_for_teacher_fit', np.nan
            ),
            'crossfit_validation_test_excluded': res.get(
                'crossfit_validation_test_excluded', np.nan
            ),
            'crossfit_teacher_optimization_steps_total': res.get(
                'crossfit_teacher_optimization_steps_total', np.nan
            ),
            'crossfit_teacher_architecture': res.get('crossfit_teacher_architecture'),
            'crossfit_teacher_loss': res.get('crossfit_teacher_loss'),
            'crossfit_teacher_initialization': res.get('crossfit_teacher_initialization'),
            'crossfit_teacher_refresh': res.get('crossfit_teacher_refresh'),
            'prediction_artifact_path': str(prediction_artifact_path) if prediction_artifact_path else None,
            'embedding_artifact_path': str(embedding_artifact_path) if embedding_artifact_path else None,
        })
        for k, v in params.items():
            row[f'param_{k}'] = v
        results_storage.append(row)

        if params.get('instrumentation_mi_gradient_diagnostics', False):
            for diagnostic_row in res['_mi_gradient_diagnostics']:
                mi_gradient_storage.append({
                    'run_id': run_id,
                    'method': setting_name,
                    **diagnostic_row,
                })

        for diagnostic_row in res['_refresh_diagnostics']:
            refresh_storage.append({
                'run_id': run_id,
                'scenario': scenario,
                'method': setting_name,
                'setting': setting_name,
                **diagnostic_row,
            })
        for diagnostic_row in res['_epoch_diagnostics']:
            epoch_storage.append({
                'run_id': run_id,
                'scenario': scenario,
                'method': setting_name,
                'setting': setting_name,
                **diagnostic_row,
            })
        for diagnostic_row in res['_mi_selection_diagnostics']:
            mi_selection_storage.append({
                'run_id': run_id,
                'scenario': scenario,
                'setting': setting_name,
                **diagnostic_row,
            })
        for diagnostic_row in res['_probe_results']:
            probe_storage.append({
                'run_id': run_id,
                'dataset': 'IHDP',
                'scenario': scenario,
                'replication_id': i,
                'data_seed': i,
                'model_seed': i,
                'method': setting_name,
                'split_hash': res['split_hash'],
                'probe_seed': res['representation_probe_seed'],
                **diagnostic_row,
            })

        print(
            f"[{setting_name}] [Sim {i + 1}/{n_sims}] PEHE: {res['test_pehe']:.4f} | "
            f"ATE: {res['ate_err']:.4f} | Ep: {res['epochs']}",
            flush=True,
        )

    if params.get('instrumentation_mi_gradient_diagnostics', False):
        gradient_path = Path(OUT_DIR) / 'mi_gradient_diagnostics.csv'
        pd.DataFrame(mi_gradient_storage).to_csv(
            gradient_path, index=False, sep=';'
        )
        print(f'Saved MI gradient diagnostics to: {gradient_path}')

    df = pd.DataFrame(results_storage)
    agg = pd.DataFrame([{
        'setting': setting_name,
        'n_sims': n_sims,
        'mean_pehe': df['test_pehe'].mean(),
        'std_pehe': df['test_pehe'].std(),
        'mean_ate_err': df['ate_err'].mean(),
        'std_ate_err': df['ate_err'].std(),
        'mean_best_val_pehe': df['best_val_pehe'].mean(),
        'mean_epochs': df['epochs'].mean(),
        'mean_treatment_probe_auc': df['treatment_probe_auc'].mean(),
        'mean_effect_probe_r2': df['effect_probe_r2'].mean(),
        'mean_last_epoch_loss_mi': df['last_epoch_loss_mi'].mean(),
        'mean_last_epoch_loss_ctr': df['last_epoch_loss_ctr'].mean(),
    }])
    if return_diagnostics:
        return (
            df,
            agg,
            pd.DataFrame(refresh_storage),
            pd.DataFrame(epoch_storage),
            pd.DataFrame(mi_selection_storage),
            pd.DataFrame(probe_storage),
        )
    return df, agg


def build_mi_ablation_configs(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs = {}
    for mode in (
        MI_NO,
        MI_CURRENT_LOCAL,
        MI_CURRENT_GLOBAL,
        MI_TRUE_GLOBAL,
        MI_DEDUPLICATED_SUPPORT,
    ):
        params = copy.deepcopy(best_params)
        params['mi_mode'] = mode
        params['mi_use_schedule'] = True
        configs[mode] = params
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


def run_experiment_group(experiment_name: str, configs: Dict[str, Dict[str, Any]], X_tr, T_tr, YF_tr, mu0_tr, mu1_tr, X_te, T_te, YF_te, mu0_te, mu1_te, n_sims: int):
    all_rows = []
    all_agg = []
    all_refresh = []
    all_epoch = []
    all_mi_selection = []
    all_probe = []
    failed = []

    for setting_name, params in configs.items():
        try:
            df, agg, refresh_df, epoch_df, mi_selection_df, probe_df = run_setting(
                setting_name=setting_name,
                params=params,
                X_tr=X_tr, T_tr=T_tr, YF_tr=YF_tr, mu0_tr=mu0_tr, mu1_tr=mu1_tr,
                X_te=X_te, T_te=T_te, YF_te=YF_te, mu0_te=mu0_te, mu1_te=mu1_te,
                n_sims=n_sims,
                return_diagnostics=True,
            )
            all_rows.append(df)
            all_agg.append(agg)
            if not refresh_df.empty:
                all_refresh.append(refresh_df)
            if not epoch_df.empty:
                all_epoch.append(epoch_df)
            if not mi_selection_df.empty:
                all_mi_selection.append(mi_selection_df)
            if not probe_df.empty:
                all_probe.append(probe_df)
        except Exception as e:
            LOGGER.exception(f"Setting failed: {setting_name}")
            failed.append({'setting': setting_name, 'error': repr(e)})

    if not all_rows:
        raise RuntimeError(f"All settings failed for {experiment_name}")

    df_all = pd.concat(all_rows, axis=0, ignore_index=True)
    df_agg = pd.concat(all_agg, axis=0, ignore_index=True)

    if failed:
        fail_path = os.path.join(OUT_DIR, f"{experiment_name}_failed_settings.csv")
        pd.DataFrame(failed).to_csv(fail_path, index=False, sep=';')
        print(f"Saved failed settings to: {fail_path}")

    save_experiment_results(experiment_name, df_all, df_agg)

    canonical_outputs_enabled = any(
        bool(params.get('instrumentation_canonical_outputs', False))
        for params in configs.values()
    )
    if canonical_outputs_enabled:
        canonical_path = os.path.join(OUT_DIR, "per_run_results.csv")
        df_all.to_csv(canonical_path, index=False, sep=';')
        print(f"Saved canonical per-run results to: {canonical_path}")

    if all_refresh:
        refresh_path = os.path.join(OUT_DIR, f"{experiment_name}_refresh_diagnostics.csv")
        pd.concat(all_refresh, axis=0, ignore_index=True).to_csv(refresh_path, index=False, sep=';')
        print(f"Saved refresh diagnostics to: {refresh_path}")
    if all_epoch:
        epoch_path = os.path.join(OUT_DIR, f"{experiment_name}_epoch_diagnostics.csv")
        pd.concat(all_epoch, axis=0, ignore_index=True).to_csv(epoch_path, index=False, sep=';')
        print(f"Saved epoch diagnostics to: {epoch_path}")
        if canonical_outputs_enabled:
            canonical_epoch_path = os.path.join(OUT_DIR, "epoch_diagnostics.csv")
            pd.concat(all_epoch, axis=0, ignore_index=True).to_csv(canonical_epoch_path, index=False, sep=';')
            print(f"Saved canonical epoch diagnostics to: {canonical_epoch_path}")
    if all_mi_selection:
        mi_selection_path = os.path.join(OUT_DIR, "mi_selection_diagnostics.csv")
        pd.concat(all_mi_selection, axis=0, ignore_index=True).to_csv(mi_selection_path, index=False, sep=';')
        print(f"Saved MI selection diagnostics to: {mi_selection_path}")
    if all_probe:
        probe_path = os.path.join(OUT_DIR, "probe_results.csv")
        pd.concat(all_probe, axis=0, ignore_index=True).to_csv(probe_path, index=False, sep=';')
        print(f"Saved probe results to: {probe_path}")

    if canonical_outputs_enabled and all_refresh:
        refresh_all = pd.concat(all_refresh, axis=0, ignore_index=True)
        canonical_refresh_path = os.path.join(OUT_DIR, "refresh_diagnostics.csv")
        refresh_all.to_csv(canonical_refresh_path, index=False, sep=';')
        pair_quality_path = os.path.join(OUT_DIR, "pair_quality.csv")
        support_path = os.path.join(OUT_DIR, "support_diagnostics.csv")
        refresh_all.to_csv(pair_quality_path, index=False, sep=';')
        refresh_all.to_csv(support_path, index=False, sep=';')
        print(f"Saved canonical refresh diagnostics to: {canonical_refresh_path}")
        print(f"Saved pair quality diagnostics to: {pair_quality_path}")
        print(f"Saved support diagnostics to: {support_path}")

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
    print(f"Simulazioni per MI ablation: {n_sims} / {X_tr.shape[-1] if X_tr.ndim == 3 else 1}")

    mi_configs = build_mi_ablation_configs(BEST_PARAMS)
    run_experiment_group(
        experiment_name="mi_ablation",
        configs=mi_configs,
        X_tr=X_tr, T_tr=T_tr, YF_tr=YF_tr, mu0_tr=mu0_tr, mu1_tr=mu1_tr,
        X_te=X_te, T_te=T_te, YF_te=YF_te, mu0_te=mu0_te, mu1_te=mu1_te,
        n_sims=n_sims,
    )


if __name__ == "__main__":
    main()
