from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata, wilcoxon


def _holm_adjust(p_values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running_max = 0.0
    for rank_index, original_index in enumerate(order):
        candidate = min(1.0, (values.size - rank_index) * values[original_index])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank_index] = running_max
    adjusted = np.empty(values.size, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted <= 0.05, adjusted


def paired_summary(
    differences: Iterable[float],
    bootstrap_seed: int = 20260915,
    bootstrap_resamples: int = 50_000,
    tie_tolerance: float = 1e-12,
) -> Dict[str, float]:
    values = np.asarray(list(differences), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one paired difference is required")
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, values.size, size=(int(bootstrap_resamples), values.size))
    boot_means = values[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    nonzero = values[np.abs(values) > tie_tolerance]
    if nonzero.size:
        try:
            test = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
        except TypeError:
            test = wilcoxon(values, zero_method="wilcox", alternative="two-sided", mode="auto")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    else:
        statistic, p_value = 0.0, 1.0
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    cohens_dz = float(np.mean(values) / standard_deviation) if standard_deviation > 0 else float("nan")
    return {
        "n_blocks": int(values.size),
        "wins": int(np.sum(values < -tie_tolerance)),
        "ties": int(np.sum(np.abs(values) <= tie_tolerance)),
        "losses": int(np.sum(values > tie_tolerance)),
        "mean_paired_difference": float(np.mean(values)),
        "median_paired_difference": float(np.median(values)),
        "bootstrap_ci_95_low": float(ci_low),
        "bootstrap_ci_95_high": float(ci_high),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value_two_sided": p_value,
        "paired_cohens_dz": cohens_dz,
    }


def aligned_friedman_holm(
    scores: pd.DataFrame,
    block_column: str,
    method_column: str,
    metric_column: str,
    lower_is_better: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Aligned ranks by independent block; never treat optimizer seeds as blocks."""
    wide = scores.pivot(index=block_column, columns=method_column, values=metric_column).dropna()
    if wide.shape[0] < 2 or wide.shape[1] < 3:
        raise ValueError("Aligned Friedman requires >=2 complete blocks and >=3 methods")
    centered = wide.sub(wide.mean(axis=1), axis=0)
    aligned_values = centered.to_numpy().reshape(-1)
    ranks = rankdata(aligned_values, method="average").reshape(centered.shape)
    if not lower_is_better:
        ranks = ranks.max() + 1 - ranks
    rank_frame = pd.DataFrame(ranks, index=wide.index, columns=wide.columns)
    n_blocks, n_methods = rank_frame.shape
    method_rank_sums = rank_frame.sum(axis=0).to_numpy()
    block_rank_sums = rank_frame.sum(axis=1).to_numpy()
    numerator = (n_methods - 1) * (
        np.sum(method_rank_sums ** 2)
        - n_methods * n_blocks ** 2 * (n_methods * n_blocks + 1) ** 2 / 4
    )
    denominator = (
        n_methods * n_blocks * (n_methods * n_blocks + 1)
        * (2 * n_methods * n_blocks + 1) / 6
        - np.sum(block_rank_sums ** 2) / n_methods
    )
    statistic = float(numerator / denominator) if denominator > 0 else float("nan")
    omnibus_p = float(chi2.sf(statistic, df=n_methods - 1))

    methods = list(wide.columns)
    pair_rows = []
    raw_p = []
    pairs = []
    for first_index in range(len(methods)):
        for second_index in range(first_index + 1, len(methods)):
            first, second = methods[first_index], methods[second_index]
            test = wilcoxon(wide[first], wide[second], alternative="two-sided")
            pairs.append((first, second))
            raw_p.append(float(test.pvalue))
    rejected, adjusted = _holm_adjust(raw_p)
    for (first, second), p_raw, p_adjusted, reject in zip(pairs, raw_p, adjusted, rejected):
        pair_rows.append({
            "method_a": first,
            "method_b": second,
            "p_value_raw": p_raw,
            "p_value_holm": float(p_adjusted),
            "reject_holm_0_05": bool(reject),
        })
    mean_ranks = rank_frame.mean(axis=0).rename("mean_aligned_rank").reset_index()
    return mean_ranks, pd.DataFrame(pair_rows), {
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(omnibus_p),
        "n_blocks": int(n_blocks),
        "n_methods": int(n_methods),
    }
