from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_corr(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if rank:
        left = pd.Series(left).rank(method="average").to_numpy()
        right = pd.Series(right).rank(method="average").to_numpy()
    if left.size < 2 or left.size != right.size or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def summarize_mi_selection(
    selected_unit_indices: np.ndarray,
    n_train: int,
    treatment: np.ndarray,
    method: str,
    tau_true: Optional[np.ndarray] = None,
    pair_degree: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    selected = np.asarray(selected_unit_indices, dtype=np.int64).reshape(-1)
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    multiplicity = np.bincount(selected, minlength=int(n_train)).astype(np.int64)
    unique = np.flatnonzero(multiplicity > 0)
    nonzero_multiplicity = multiplicity[unique]
    row: Dict[str, float] = {
        "n_mi_endpoints_total": int(selected.size),
        "n_mi_unique_units": int(unique.size),
        "mi_unique_fraction_train": float(unique.size / n_train) if n_train else float("nan"),
        "endpoint_multiplicity_mean": float(np.mean(nonzero_multiplicity)) if unique.size else float("nan"),
        "endpoint_multiplicity_median": float(np.median(nonzero_multiplicity)) if unique.size else float("nan"),
        "endpoint_multiplicity_q90": float(np.quantile(nonzero_multiplicity, 0.90)) if unique.size else float("nan"),
        "endpoint_multiplicity_max": int(np.max(nonzero_multiplicity)) if unique.size else 0,
    }
    for treatment_value in np.unique(treatment):
        row[f"selected_treatment_fraction_{treatment_value}"] = (
            float(np.mean(treatment[selected] == treatment_value))
            if selected.size else float("nan")
        )

    if tau_true is not None:
        tau_true = np.asarray(tau_true, dtype=np.float64).reshape(-1)
        selected_tau = tau_true[selected]
        unselected_tau = tau_true[multiplicity == 0]
        row.update({
            "tau_mean_selected": float(np.mean(selected_tau)) if selected_tau.size else float("nan"),
            "tau_std_selected": float(np.std(selected_tau)) if selected_tau.size else float("nan"),
            "tau_q10_selected": float(np.quantile(selected_tau, 0.10)) if selected_tau.size else float("nan"),
            "tau_q90_selected": float(np.quantile(selected_tau, 0.90)) if selected_tau.size else float("nan"),
            "tau_mean_unselected": float(np.mean(unselected_tau)) if unselected_tau.size else float("nan"),
            "tau_std_unselected": float(np.std(unselected_tau)) if unselected_tau.size else float("nan"),
        })

    if pair_degree is not None:
        pair_degree = np.asarray(pair_degree, dtype=np.float64).reshape(-1)
        row.update({
            "endpoint_multiplicity_pair_degree_pearson": _safe_corr(multiplicity, pair_degree),
            "endpoint_multiplicity_pair_degree_spearman": _safe_corr(multiplicity, pair_degree, rank=True),
        })
    row["selection_independent_of_pair_sampler_by_construction"] = method == "TrueGlobalMI"
    row["uniform_full_population_selection_empirically_verified"] = bool(
        method == "TrueGlobalMI"
        and unique.size == n_train
        and nonzero_multiplicity.size > 0
        and np.all(nonzero_multiplicity == nonzero_multiplicity[0])
    )
    return row, multiplicity
