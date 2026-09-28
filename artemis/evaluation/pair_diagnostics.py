from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .support_diagnostics import mask_jaccard, support_transition, treatment_support_summary


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or left.size != right.size or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(np.asarray(left).reshape(-1)).rank(method="average").to_numpy()
    right_rank = pd.Series(np.asarray(right).reshape(-1)).rank(method="average").to_numpy()
    return _safe_pearson(left_rank, right_rank)


def build_fixed_pair_panel(
    n_units: int, panel_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if n_units < 2 or panel_size <= 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty
    all_a, all_b = np.triu_indices(int(n_units), k=1)
    if panel_size >= all_a.size:
        return all_a.astype(np.int64), all_b.astype(np.int64)
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(all_a.size, size=int(panel_size), replace=False)
    return all_a[chosen].astype(np.int64), all_b[chosen].astype(np.int64)


def _classification_counts(predicted: np.ndarray, oracle: np.ndarray):
    tp = int(np.logical_and(predicted, oracle).sum())
    fp = int(np.logical_and(predicted, ~oracle).sum())
    fn = int(np.logical_and(~predicted, oracle).sum())
    precision = float(tp / (tp + fp)) if tp + fp else float("nan")
    recall = float(tp / (tp + fn)) if tp + fn else float("nan")
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else float("nan")
    return tp, fp, fn, precision, recall, f1


def compute_pair_audit(
    pseudo_tau: np.ndarray,
    treatment: np.ndarray,
    panel: Tuple[np.ndarray, np.ndarray],
    predicted_threshold: float,
    percentile: float,
    true_tau: Optional[np.ndarray] = None,
    previous_positive_mask: Optional[np.ndarray] = None,
    previous_support_mask: Optional[np.ndarray] = None,
    active_support_indices: Optional[np.ndarray] = None,
    local_mi_unique_indices: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    pseudo_tau = np.asarray(pseudo_tau, dtype=np.float64).reshape(-1)
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    idx_a, idx_b = (np.asarray(panel[0], dtype=np.int64), np.asarray(panel[1], dtype=np.int64))
    pseudo_distance = np.abs(pseudo_tau[idx_a] - pseudo_tau[idx_b])
    positive = pseudo_distance < float(predicted_threshold)
    negative = ~positive
    positive_endpoints = np.concatenate([idx_a[positive], idx_b[positive]])
    graph_support_mask = np.zeros(pseudo_tau.size, dtype=bool)
    graph_support_mask[np.unique(positive_endpoints)] = True
    degree = np.bincount(positive_endpoints, minlength=pseudo_tau.size).astype(np.int64)

    active_mask = np.zeros(pseudo_tau.size, dtype=bool)
    if active_support_indices is not None:
        active_mask[np.asarray(active_support_indices, dtype=np.int64)] = True
    local_mi_mask = np.zeros(pseudo_tau.size, dtype=bool)
    local_mi_defined = local_mi_unique_indices is not None
    if local_mi_unique_indices is not None:
        local_mi_mask[np.asarray(local_mi_unique_indices, dtype=np.int64)] = True

    positive_ta = treatment[idx_a[positive]]
    positive_tb = treatment[idx_b[positive]]
    row: Dict[str, float] = {
        "positive_edge_fraction": float(np.mean(positive)) if positive.size else float("nan"),
        "n_positive_pairs_audit": int(positive.sum()),
        "n_negative_pairs_audit": int(negative.sum()),
        "graph_support_fraction": float(np.mean(graph_support_mask)),
        "sampled_active_positive_support_fraction": float(np.mean(active_mask)),
        "current_local_mi_unique_fraction": float(np.mean(local_mi_mask)) if local_mi_defined else float("nan"),
        "n_graph_support_units": int(graph_support_mask.sum()),
        "n_active_positive_support_units": int(active_mask.sum()),
        "n_current_local_mi_unique_units": int(local_mi_mask.sum()) if local_mi_defined else float("nan"),
        "positive_edge_jaccard": mask_jaccard(previous_positive_mask, positive),
        "same_treatment_pair_fraction": float(np.mean(positive_ta == positive_tb)) if positive_ta.size else float("nan"),
        "cross_treatment_pair_fraction": float(np.mean(positive_ta != positive_tb)) if positive_ta.size else float("nan"),
        "positive_degree_mean": float(np.mean(degree)),
        "positive_degree_median": float(np.median(degree)),
        "positive_degree_std": float(np.std(degree)),
        "positive_degree_q90": float(np.quantile(degree, 0.90)),
        "positive_degree_q99": float(np.quantile(degree, 0.99)),
        "positive_degree_max": int(np.max(degree)) if degree.size else 0,
        "fraction_degree_zero": float(np.mean(degree == 0)),
    }
    row.update(support_transition(previous_support_mask, graph_support_mask))
    row.update(treatment_support_summary(treatment, graph_support_mask, "graph_support"))
    row.update(treatment_support_summary(treatment, active_mask, "active_support"))
    for ta in np.unique(treatment):
        for tb in np.unique(treatment):
            row[f"n_treatment_pair_{ta}_{tb}"] = int(
                np.logical_and(positive_ta == ta, positive_tb == tb).sum()
            )

    true_distance = np.asarray([], dtype=np.float64)
    oracle_positive = np.asarray([], dtype=bool)
    if true_tau is not None:
        true_tau = np.asarray(true_tau, dtype=np.float64).reshape(-1)
        true_distance = np.abs(true_tau[idx_a] - true_tau[idx_b])
        oracle_threshold = float(np.percentile(true_distance, percentile)) if true_distance.size else float("nan")
        oracle_positive = true_distance < oracle_threshold
        predicted_true_gaps = true_distance[positive]
        row.update({
            "oracle_positive_threshold": oracle_threshold,
            "pseudo_true_distance_pearson": _safe_pearson(pseudo_distance, true_distance),
            "pseudo_true_distance_spearman": _safe_spearman(pseudo_distance, true_distance),
            "true_effect_gap_mean": float(np.mean(predicted_true_gaps)) if predicted_true_gaps.size else float("nan"),
            "true_effect_gap_median": float(np.median(predicted_true_gaps)) if predicted_true_gaps.size else float("nan"),
            "true_effect_gap_q75": float(np.quantile(predicted_true_gaps, 0.75)) if predicted_true_gaps.size else float("nan"),
            "true_effect_gap_q90": float(np.quantile(predicted_true_gaps, 0.90)) if predicted_true_gaps.size else float("nan"),
            "true_effect_gap_max": float(np.max(predicted_true_gaps)) if predicted_true_gaps.size else float("nan"),
        })
        _, fp, _, precision, recall, f1 = _classification_counts(positive, oracle_positive)
        false_positive_distance = true_distance[np.logical_and(positive, ~oracle_positive)]
        row.update({
            "pair_precision_oracle": precision,
            "pair_recall_oracle": recall,
            "pair_f1_oracle": f1,
            "n_false_positive_pairs": fp,
            "false_positive_true_distance_mean": float(np.mean(false_positive_distance)) if false_positive_distance.size else float("nan"),
            "false_positive_true_distance_q90": float(np.quantile(false_positive_distance, 0.90)) if false_positive_distance.size else float("nan"),
        })

    arrays = {
        "audit_idx_a": idx_a,
        "audit_idx_b": idx_b,
        "pseudo_effect_distance": pseudo_distance,
        "predicted_positive_mask": positive,
        "graph_support_mask": graph_support_mask,
        "positive_pair_degree": degree,
        "true_effect_distance": true_distance,
        "oracle_positive_mask": oracle_positive,
    }
    return row, arrays
