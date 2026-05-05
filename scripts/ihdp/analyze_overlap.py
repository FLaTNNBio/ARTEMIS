import os
import json
import copy
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Reuse user's training components from test_gmi.py
import test_gmi as base

LOGGER = logging.getLogger("IHDP_LATENT_DIAGNOSTICS")
logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# USER CONFIG
# ==============================================================================
N_SIMS = 100
OUT_DIR = "latent_diagnostics_ihdp"
os.makedirs(OUT_DIR, exist_ok=True)

ANALYZE_SPLIT = "test"   # "test" or "train"
SETTINGS_TO_COMPARE = [
    "full_model",
    "no_contrastive",
    "no_local_mi",
    "supervised_only",
]

K_NEIGHBORS = 10
TAU_BINS = 4
PAIR_SAMPLE_SIZE = 20000
LOW_SUPPORT_QUANTILE = 0.25
HARD_SUPPORT_QUANTILE = 0.10
RBF_GAMMA = None          # None -> median heuristic
USE_TRUE_TAU_FOR_ALIGNMENT = True
USE_TRUE_TAU_FOR_SUPPORT_ERROR = True

# Optional external treatment probe
TRY_EXTERNAL_TREATMENT_PROBE = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# METRICS
# ==============================================================================
def pdist2_numpy(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    if Y is None:
        Y = X
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    return np.maximum(d2, 0.0)


def linear_mmd(z0: np.ndarray, z1: np.ndarray) -> float:
    m0 = np.mean(z0, axis=0)
    m1 = np.mean(z1, axis=0)
    return float(np.linalg.norm(m0 - m1))


def _median_heuristic_gamma(Z: np.ndarray, max_points: int = 1000) -> float:
    if Z.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(Z.shape[0], size=max_points, replace=False)
        Z = Z[idx]
    d2 = pdist2_numpy(Z)
    tri = d2[np.triu_indices_from(d2, k=1)]
    tri = tri[np.isfinite(tri)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return 1.0
    med = np.median(tri)
    return float(1.0 / max(med, 1e-8))


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = pdist2_numpy(X, Y)
    return np.exp(-gamma * d2)


def rbf_mmd(z0: np.ndarray, z1: np.ndarray, gamma: float = None) -> float:
    Z = np.concatenate([z0, z1], axis=0)
    if gamma is None:
        gamma = _median_heuristic_gamma(Z)
    K00 = rbf_kernel(z0, z0, gamma)
    K11 = rbf_kernel(z1, z1, gamma)
    K01 = rbf_kernel(z0, z1, gamma)
    return float(K00.mean() + K11.mean() - 2.0 * K01.mean())


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    sx = np.std(rx)
    sy = np.std(ry)
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def semantic_alignment(z: np.ndarray, tau: np.ndarray, sample_pairs: int = 20000, seed: int = 0) -> float:
    n = z.shape[0]
    if n < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    m = min(sample_pairs, n * max(n - 1, 1))
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n - 1, size=m)
    j = np.where(j >= i, j + 1, j)
    dz = np.linalg.norm(z[i] - z[j], axis=1)
    dt = np.abs(tau[i] - tau[j])
    return spearman_rank_corr(dz, dt)


def knn_overlap(z: np.ndarray, t: np.ndarray, k: int = 10) -> float:
    n = z.shape[0]
    if n <= 1:
        return np.nan
    k_eff = min(k, n - 1)
    d2 = pdist2_numpy(z)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    opp = (t[nn_idx] != t[:, None]).mean(axis=1)
    return float(np.mean(opp))


def nearest_opposite_treatment_distance(z: np.ndarray, t: np.ndarray) -> float:
    z0 = z[t == 0]
    z1 = z[t == 1]
    if len(z0) == 0 or len(z1) == 0:
        return np.nan
    d01 = np.sqrt(pdist2_numpy(z0, z1))
    nn0 = np.min(d01, axis=1)
    nn1 = np.min(d01, axis=0)
    return float(np.mean(np.concatenate([nn0, nn1])))


def local_support_scores(z: np.ndarray, t: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    n = z.shape[0]
    if n <= 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    k_eff = min(k, n - 1)
    d2 = pdist2_numpy(z)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
    overlap = (t[nn_idx] != t[:, None]).mean(axis=1)

    opp_dist = np.full(n, np.nan, dtype=np.float64)
    for val in [0, 1]:
        idx_a = np.where(t == val)[0]
        idx_b = np.where(t != val)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        d = np.sqrt(pdist2_numpy(z[idx_a], z[idx_b]))
        opp_dist[idx_a] = np.min(d, axis=1)
    return overlap, opp_dist


def support_gap_ratio(errors: np.ndarray, support_score: np.ndarray, low_quantile: float = 0.25) -> float:
    mask = np.isfinite(errors) & np.isfinite(support_score)
    errors = errors[mask]
    support_score = support_score[mask]
    if errors.size < 10:
        return np.nan
    thr = np.quantile(support_score, low_quantile)
    low = errors[support_score <= thr]
    high = errors[support_score > thr]
    if low.size == 0 or high.size == 0:
        return np.nan
    return float(np.mean(low) / max(np.mean(high), 1e-8))


def hard_gap_ratio(errors: np.ndarray, support_score: np.ndarray, hard_quantile: float = 0.10) -> float:
    mask = np.isfinite(errors) & np.isfinite(support_score)
    errors = errors[mask]
    support_score = support_score[mask]
    if errors.size < 10:
        return np.nan
    thr = np.quantile(support_score, hard_quantile)
    hard = errors[support_score <= thr]
    rest = errors[support_score > thr]
    if hard.size == 0 or rest.size == 0:
        return np.nan
    return float(np.mean(hard) / max(np.mean(rest), 1e-8))


def local_metrics_by_tau_bins(z: np.ndarray, t: np.ndarray, tau: np.ndarray, k: int = 10, n_bins: int = 4) -> Dict[str, float]:
    bins = pd.qcut(tau, q=n_bins, duplicates="drop")
    aucs = []
    mmds = []
    overlaps = []
    nn_dists = []

    for _, idx in pd.Series(np.arange(len(tau))).groupby(bins, observed=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if idx.size < max(20, k + 2):
            continue
        t_bin = t[idx]
        if len(np.unique(t_bin)) < 2:
            continue
        z_bin = z[idx]
        z0 = z_bin[t_bin == 0]
        z1 = z_bin[t_bin == 1]
        if len(z0) < 3 or len(z1) < 3:
            continue
        if TRY_EXTERNAL_TREATMENT_PROBE and SKLEARN_AVAILABLE:
            auc_val = external_probe_auc(z_bin, t_bin)
            if np.isfinite(auc_val):
                aucs.append(auc_val)
        mmds.append(rbf_mmd(z0, z1, gamma=RBF_GAMMA))
        overlaps.append(knn_overlap(z_bin, t_bin, k=min(k, idx.size - 1)))
        nn_dists.append(nearest_opposite_treatment_distance(z_bin, t_bin))

    return {
        "local_probe_auc_mean": float(np.nanmean(aucs)) if len(aucs) else np.nan,
        "local_rbf_mmd_mean": float(np.nanmean(mmds)) if len(mmds) else np.nan,
        "local_knn_overlap_mean": float(np.nanmean(overlaps)) if len(overlaps) else np.nan,
        "local_opp_nn_dist_mean": float(np.nanmean(nn_dists)) if len(nn_dists) else np.nan,
    }


def external_probe_auc(z: np.ndarray, t: np.ndarray) -> float:
    if not (TRY_EXTERNAL_TREATMENT_PROBE and SKLEARN_AVAILABLE):
        return np.nan
    if len(np.unique(t)) < 2 or z.shape[0] < 20:
        return np.nan
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            z, t, test_size=0.35, random_state=0, stratify=t
        )
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, prob))
    except Exception:
        return np.nan


# ==============================================================================
# TRAINING THAT RETURNS LATENT REPRESENTATIONS
# ==============================================================================
def train_single_simulation_with_latents(
    sim_idx: int,
    data_train: Tuple,
    data_test: Tuple,
    device: str,
    hyperparams: Dict[str, Any],
    analyze_split: str = "test",
) -> Dict[str, Any]:
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

    num_treatments = base.infer_num_treatments(T_tr_s)

    n_total = X_tr_s.shape[0]
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    rng = np.random.default_rng(sim_idx)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    cont_indices = base.get_continuous_indices(X_tr_s)

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

    ds_train = base.DynamicContrastiveCausalDS(
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
    encoder = base.CATEEncoder(input_dim, LATENT_DIM, dropout=ENCODER_DROPOUT).to(device)
    predictor = base.OutcomeHead(
        LATENT_DIM,
        num_treatments=num_treatments,
        use_output_clip=USE_OUTPUT_CLIP,
        clip_val=calculated_clip_val,
        dropout=HEAD_DROPOUT
    ).to(device)
    treat_clf = base.TreatmentClassifier(LATENT_DIM, num_treatments=num_treatments, dropout=CLF_DROPOUT).to(device)

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

    early_stopper = base.EarlyStoppingPEHE(patience=PATIENCE)
    final_val_pehe = 999.0
    static_pair_initialized = False

    for epoch in range(EPOCHS):
        ds_train.set_epoch(epoch)

        RAMP = 80
        lambda_ctr = 0.0 if (epoch < WARMUP_EPOCHS or not USE_CONTRASTIVE or PAIR_MODE == 'none') else min(ALPHA, (epoch - WARMUP_EPOCHS) / RAMP * ALPHA)
        lambda_mi = 0.0 if (epoch < MI_START_EPOCH or not USE_LOCAL_MI or PAIR_MODE == 'none') else LAMBDA_MI_POS

        encoder.train()
        predictor.train()
        treat_clf.train()

        for batch in dl_train:
            x1, y1, t1, x2, y2, t2, label = [b.to(device) for b in batch]

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
                    clf_loss = base.treatment_classifier_loss(treat_clf, z_pos_det, t_pos_det, num_treatments)
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
                loss_ctr = base.contrastive_loss(z1, z2, label, margin=MARGIN)

            loss_mi = torch.tensor(0.0, device=device)
            if lambda_mi > 0 and pos_mask.sum().item() > POS_MIN_COUNT:
                z_pos = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)
                t_pos = torch.cat([t1_idx[pos_mask], t2_idx[pos_mask]], dim=0)
                mi_lb = base.variational_mi_lower_bound(treat_clf, z_pos, t_pos, num_treatments)
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

    # Performance on test
    X_te_t = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_te = encoder(X_te_t)
        mu_te_all = predictor(z_te)
        ite_pred_test = (mu_te_all[:, 1] - mu_te_all[:, 0]).cpu().numpy().ravel() * y_std
    y_true_te = np.stack([mu0_te_s, mu1_te_s], axis=1)
    pehe = base.sqrt_PEHE_with_diff(y_true_te, ite_pred_test)
    ate_err = base.eps_ATE_diff(mu1_te_s - mu0_te_s, ite_pred_test)

    if analyze_split == "train":
        X_an = X_train_processed
        T_an = T_tr_s[train_idx].astype(int)
        mu0_an = mu0_tr_s[train_idx]
        mu1_an = mu1_tr_s[train_idx]
    else:
        X_an = X_test_processed
        T_an = T_te_s.astype(int)
        mu0_an = mu0_te_s
        mu1_an = mu1_te_s

    X_an_t = torch.tensor(X_an, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_an = encoder(X_an_t).cpu().numpy()
        mu_an_all = predictor(X_an_t.new_tensor(z_an) if False else encoder(X_an_t))  # no-op branch avoided below
    with torch.no_grad():
        mu_an_all = predictor(encoder(X_an_t)).cpu().numpy()
    tau_hat_an = (mu_an_all[:, 1] - mu_an_all[:, 0]) * y_std
    tau_true_an = (mu1_an - mu0_an)

    return {
        "val_pehe": final_val_pehe,
        "test_pehe": pehe,
        "ate_err": ate_err,
        "epochs": epoch + 1,
        "z": z_an,
        "t": T_an,
        "tau_hat": tau_hat_an,
        "tau_true": tau_true_an,
    }


# ==============================================================================
# EXPERIMENTS
# ==============================================================================
def build_settings(best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    all_cfg = base.build_component_ablation_configs(best_params)
    all_cfg["full_model"] = copy.deepcopy(best_params)
    return {k: all_cfg[k] for k in SETTINGS_TO_COMPARE if k in all_cfg}


def compute_all_diagnostics(z: np.ndarray, t: np.ndarray, tau_true: np.ndarray, tau_hat: np.ndarray, seed: int = 0) -> Dict[str, float]:
    tau_for_alignment = tau_true if USE_TRUE_TAU_FOR_ALIGNMENT else tau_hat
    err = np.abs((tau_true if USE_TRUE_TAU_FOR_SUPPORT_ERROR else tau_hat) - tau_hat)

    z0 = z[t == 0]
    z1 = z[t == 1]
    out = {}
    out["linear_mmd"] = linear_mmd(z0, z1)
    out["rbf_mmd"] = rbf_mmd(z0, z1, gamma=RBF_GAMMA)
    out["semantic_alignment"] = semantic_alignment(z, tau_for_alignment, sample_pairs=PAIR_SAMPLE_SIZE, seed=seed)
    out["knn_overlap"] = knn_overlap(z, t, k=K_NEIGHBORS)
    out["opp_treat_nn_dist"] = nearest_opposite_treatment_distance(z, t)

    support_overlap, opp_dist = local_support_scores(z, t, k=K_NEIGHBORS)
    out["support_gap_overlap"] = support_gap_ratio(err, support_overlap, low_quantile=LOW_SUPPORT_QUANTILE)
    out["hard_gap_overlap"] = hard_gap_ratio(err, support_overlap, hard_quantile=HARD_SUPPORT_QUANTILE)

    # Using negative distance so lower distance = lower score = low support.
    support_from_dist = -opp_dist
    out["support_gap_oppnn"] = support_gap_ratio(err, support_from_dist, low_quantile=LOW_SUPPORT_QUANTILE)
    out["hard_gap_oppnn"] = hard_gap_ratio(err, support_from_dist, hard_quantile=HARD_SUPPORT_QUANTILE)

    if TRY_EXTERNAL_TREATMENT_PROBE and SKLEARN_AVAILABLE:
        out["probe_auc"] = external_probe_auc(z, t)
    else:
        out["probe_auc"] = np.nan

    out.update(local_metrics_by_tau_bins(z, t, tau_true, k=K_NEIGHBORS, n_bins=TAU_BINS))
    return out


def run_all():
    loader = base.AbstractCausalLoader.get_loader('IHDP')
    loaded_data = loader.load()
    X_tr, T_tr, YF_tr, _, mu0_tr, mu1_tr, X_te, T_te, YF_te, _, mu0_te, mu1_te = loaded_data

    total_avail = X_tr.shape[-1] if X_tr.ndim == 3 else 1
    n_sims = min(N_SIMS, total_avail)

    settings = build_settings(base.BEST_PARAMS)
    LOGGER.info("Running latent diagnostics on %d simulations for settings: %s", n_sims, list(settings.keys()))

    rows = []
    for setting_name, params in settings.items():
        for i in range(n_sims):
            if X_tr.ndim == 3:
                train_data = (X_tr[:, :, i], T_tr[:, i], YF_tr[:, i], mu0_tr[:, i], mu1_tr[:, i])
                test_data = (X_te[:, :, i], T_te[:, i], YF_te[:, i], mu0_te[:, i], mu1_te[:, i])
            else:
                train_data = (X_tr, T_tr, YF_tr, mu0_tr, mu1_tr)
                test_data = (X_te, T_te, YF_te, mu0_te, mu1_te)

            res = train_single_simulation_with_latents(
                sim_idx=i,
                data_train=train_data,
                data_test=test_data,
                device=DEVICE,
                hyperparams=params,
                analyze_split=ANALYZE_SPLIT,
            )

            diags = compute_all_diagnostics(
                z=res["z"],
                t=res["t"],
                tau_true=res["tau_true"],
                tau_hat=res["tau_hat"],
                seed=i,
            )

            row = {
                "setting": setting_name,
                "sim_id": i,
                "split": ANALYZE_SPLIT,
                "test_pehe": res["test_pehe"],
                "ate_err": res["ate_err"],
                "val_pehe": res["val_pehe"],
                "epochs": res["epochs"],
            }
            row.update(diags)
            rows.append(row)

            LOGGER.info(
                "[%s] Sim %d/%d | PEHE %.4f | ATE %.4f | Align %.4f | kNN %.4f | OppNN %.4f | Probe AUC %.4f",
                setting_name,
                i + 1,
                n_sims,
                res["test_pehe"],
                res["ate_err"],
                row.get("semantic_alignment", np.nan),
                row.get("knn_overlap", np.nan),
                row.get("opp_treat_nn_dist", np.nan),
                row.get("probe_auc", np.nan),
            )

    df = pd.DataFrame(rows)
    per_sim_path = os.path.join(OUT_DIR, "latent_diagnostics_per_sim.csv")
    df.to_csv(per_sim_path, index=False, sep=';')

    metric_cols = [c for c in df.columns if c not in {"setting", "sim_id", "split"}]
    summary_rows = []
    for setting_name, g in df.groupby("setting"):
        row = {"setting": setting_name, "n_sims": len(g)}
        for c in metric_cols:
            row[f"mean_{c}"] = g[c].mean()
            row[f"std_{c}"] = g[c].std()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "latent_diagnostics_summary.csv")
    summary.to_csv(summary_path, index=False, sep=';')

    notes = {
        "analyze_split": ANALYZE_SPLIT,
        "n_sims": int(n_sims),
        "settings": list(settings.keys()),
        "k_neighbors": K_NEIGHBORS,
        "tau_bins": TAU_BINS,
        "pair_sample_size": PAIR_SAMPLE_SIZE,
        "low_support_quantile": LOW_SUPPORT_QUANTILE,
        "hard_support_quantile": HARD_SUPPORT_QUANTILE,
        "use_true_tau_for_alignment": USE_TRUE_TAU_FOR_ALIGNMENT,
        "use_true_tau_for_support_error": USE_TRUE_TAU_FOR_SUPPORT_ERROR,
        "sklearn_available": SKLEARN_AVAILABLE,
        "metrics_interpretation": {
            "linear_mmd": "lower is better",
            "rbf_mmd": "lower is better",
            "semantic_alignment": "higher is better",
            "knn_overlap": "higher is better",
            "opp_treat_nn_dist": "lower is better",
            "support_gap_overlap": "closer to 1 / lower is better",
            "hard_gap_overlap": "closer to 1 / lower is better",
            "support_gap_oppnn": "closer to 1 / lower is better",
            "hard_gap_oppnn": "closer to 1 / lower is better",
            "probe_auc": "closer to 0.5 is better",
            "local_probe_auc_mean": "closer to 0.5 is better",
            "local_rbf_mmd_mean": "lower is better",
            "local_knn_overlap_mean": "higher is better",
            "local_opp_nn_dist_mean": "lower is better",
        }
    }
    notes_path = os.path.join(OUT_DIR, "latent_diagnostics_notes.json")
    with open(notes_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    print("\n" + "=" * 100)
    print("LATENT DIAGNOSTICS SUMMARY")
    print("=" * 100)
    view_cols = [
        "setting",
        "mean_test_pehe",
        "mean_ate_err",
        "mean_linear_mmd",
        "mean_rbf_mmd",
        "mean_semantic_alignment",
        "mean_knn_overlap",
        "mean_opp_treat_nn_dist",
        "mean_support_gap_overlap",
        "mean_hard_gap_overlap",
        "mean_probe_auc",
        "mean_local_probe_auc_mean",
        "mean_local_rbf_mmd_mean",
        "mean_local_knn_overlap_mean",
        "mean_local_opp_nn_dist_mean",
    ]
    existing_cols = [c for c in view_cols if c in summary.columns]
    print(summary[existing_cols].to_string(index=False))
    print(f"\nSaved per-simulation results to: {per_sim_path}")
    print(f"Saved summary results to: {summary_path}")
    print(f"Saved notes to: {notes_path}")


if __name__ == "__main__":
    run_all()
