from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def mask_jaccard(left: Optional[np.ndarray], right: np.ndarray) -> float:
    if left is None:
        return float("nan")
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def treatment_support_summary(
    treatment: np.ndarray,
    support_mask: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    support_mask = np.asarray(support_mask, dtype=bool).reshape(-1)
    selected = treatment[support_mask]
    values = np.unique(treatment)
    result: Dict[str, float] = {}
    shares = []
    for value in values:
        share = float(np.mean(selected == value)) if selected.size else float("nan")
        result[f"treatment_fraction_{value}_in_{prefix}"] = share
        shares.append(share)
    result[f"min_treatment_share_in_{prefix}"] = (
        float(np.nanmin(shares)) if selected.size and shares else float("nan")
    )
    return result


def support_transition(
    previous_mask: Optional[np.ndarray], current_mask: np.ndarray
) -> Dict[str, float]:
    current_mask = np.asarray(current_mask, dtype=bool)
    if previous_mask is None:
        return {
            "graph_support_jaccard": float("nan"),
            "fraction_units_entering_support": float("nan"),
            "fraction_units_leaving_support": float("nan"),
        }
    previous_mask = np.asarray(previous_mask, dtype=bool)
    return {
        "graph_support_jaccard": mask_jaccard(previous_mask, current_mask),
        "fraction_units_entering_support": float(
            np.mean(np.logical_and(current_mask, ~previous_mask))
        ),
        "fraction_units_leaving_support": float(
            np.mean(np.logical_and(previous_mask, ~current_mask))
        ),
    }
