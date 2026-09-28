from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping

import numpy as np


CANONICAL_PER_RUN_COLUMNS = (
    "dataset", "scenario", "replication_id", "data_seed", "model_seed",
    "split_hash", "config_hash", "method", "PEHE", "ATE_error",
    "best_epoch", "best_epoch_any", "best_epoch_revised_window", "last_epoch",
    "checkpoint_protocol", "checkpoint_start_epoch", "mi_start_epoch",
    "n_mi_active_epochs", "early_stopped", "runtime_seconds",
    "best_val_metric", "test_metric", "git_commit", "run_id",
)


def make_run_id(
    dataset: str,
    scenario: str,
    replication_id: int,
    model_seed: int,
    method: str,
    config_hash: str,
) -> str:
    readable = "_".join(
        re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()
        for value in (dataset, scenario, f"r{replication_id}", f"s{model_seed}", method)
    )
    payload = json.dumps(
        {
            "dataset": dataset,
            "scenario": scenario,
            "replication_id": int(replication_id),
            "model_seed": int(model_seed),
            "method": method,
            "config_hash": config_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{readable}_{hashlib.sha256(payload).hexdigest()[:12]}"


def normalize_per_run_row(values: Mapping[str, Any]) -> Dict[str, Any]:
    row = {column: np.nan for column in CANONICAL_PER_RUN_COLUMNS}
    row.update(dict(values))
    missing_identity = [
        column for column in ("dataset", "scenario", "replication_id", "method", "run_id")
        if row[column] is None or (isinstance(row[column], float) and np.isnan(row[column]))
    ]
    if missing_identity:
        raise ValueError(f"Missing canonical run identity fields: {missing_identity}")
    return row
