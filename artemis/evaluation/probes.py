from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TREATMENT_PROBE_FACTORIES = {
    "linear": lambda seed: LogisticRegression(C=1.0, max_iter=2_000, solver="lbfgs", random_state=seed),
    "shallow_mlp": lambda seed: MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-4, max_iter=500, random_state=seed),
    "strong_mlp": lambda seed: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=500, random_state=seed),
}


def _treatment_metrics(model, z: np.ndarray, treatment: np.ndarray) -> Dict[str, float]:
    treatment = np.asarray(treatment).reshape(-1).astype(int)
    probabilities = model.predict_proba(z)
    prediction = model.predict(z)
    if probabilities.shape[1] == 2 and np.unique(treatment).size == 2:
        auc = float(roc_auc_score(treatment, probabilities[:, 1]))
    else:
        auc = float(roc_auc_score(treatment, probabilities, multi_class="ovr", average="macro"))
    return {
        "treatment_probe_AUC": auc,
        "treatment_probe_accuracy": float(accuracy_score(treatment, prediction)),
        "treatment_probe_cross_entropy": float(log_loss(treatment, probabilities, labels=model.classes_)),
    }


def semantic_alignment(
    z: np.ndarray,
    tau_true: np.ndarray,
    seed: int,
    n_pairs: int = 20_000,
) -> float:
    from scipy.stats import spearmanr

    z = np.asarray(z)
    tau_true = np.asarray(tau_true, dtype=np.float64).reshape(-1)
    if z.shape[0] < 2:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    idx_a = rng.integers(0, z.shape[0], size=int(n_pairs))
    idx_b = rng.integers(0, z.shape[0] - 1, size=int(n_pairs))
    idx_b += idx_b >= idx_a
    latent_distance = np.linalg.norm(z[idx_a] - z[idx_b], axis=1)
    effect_distance = np.abs(tau_true[idx_a] - tau_true[idx_b])
    return float(spearmanr(latent_distance, effect_distance).statistic)


def evaluate_probe_suite(
    z_train: np.ndarray,
    treatment_train: np.ndarray,
    z_validation: np.ndarray,
    treatment_validation: np.ndarray,
    z_test: np.ndarray,
    treatment_test: np.ndarray,
    seed: int,
    tau_train: Optional[np.ndarray] = None,
    tau_validation: Optional[np.ndarray] = None,
    tau_test: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    linear_test_summary: Dict[str, float] = {}
    for probe_name, factory in TREATMENT_PROBE_FACTORIES.items():
        model = make_pipeline(StandardScaler(), factory(int(seed)))
        model.fit(z_train, np.asarray(treatment_train).reshape(-1).astype(int))
        for split, z_eval, t_eval in (
            ("validation", z_validation, treatment_validation),
            ("test", z_test, treatment_test),
        ):
            metrics = _treatment_metrics(model, z_eval, t_eval)
            row = {"probe_target": "treatment", "probe_model": probe_name, "probe_split": split, **metrics}
            rows.append(row)
            if probe_name == "linear" and split == "test":
                linear_test_summary = metrics.copy()

    effect_summary = {
        "effect_probe_R2": float("nan"),
        "effect_probe_MAE": float("nan"),
        "semantic_alignment": float("nan"),
    }
    if tau_train is not None and tau_validation is not None and tau_test is not None:
        effect_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        effect_model.fit(z_train, np.asarray(tau_train).reshape(-1))
        for split, z_eval, tau_eval in (
            ("validation", z_validation, tau_validation),
            ("test", z_test, tau_test),
        ):
            tau_eval = np.asarray(tau_eval).reshape(-1)
            prediction = effect_model.predict(z_eval)
            metrics = {
                "effect_probe_R2": float(r2_score(tau_eval, prediction)),
                "effect_probe_MAE": float(mean_absolute_error(tau_eval, prediction)),
            }
            rows.append({"probe_target": "oracle_tau", "probe_model": "linear_ridge", "probe_split": split, **metrics})
            if split == "test":
                effect_summary.update(metrics)
        effect_summary["semantic_alignment"] = semantic_alignment(z_test, tau_test, seed=int(seed) + 17)

    summary = {**linear_test_summary, **effect_summary}
    return rows, summary
