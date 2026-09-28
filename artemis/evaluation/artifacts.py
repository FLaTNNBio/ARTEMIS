from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def _npz_ready(values: Mapping[str, object]):
    return {key: np.asarray(value) for key, value in values.items() if value is not None}


def save_run_artifacts(
    output_dir: Path,
    run_id: str,
    predictions: Mapping[str, object],
    embeddings: Mapping[str, object],
):
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "predictions"
    embedding_dir = output_dir / "embeddings"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{run_id}.npz"
    embedding_path = embedding_dir / f"{run_id}.npz"
    np.savez_compressed(prediction_path, **_npz_ready(predictions))
    np.savez_compressed(embedding_path, **_npz_ready(embeddings))
    return prediction_path, embedding_path


def validate_run_artifact(path: Path, required_arrays):
    path = Path(path)
    with np.load(path, allow_pickle=False) as artifact:
        missing = sorted(set(required_arrays).difference(artifact.files))
        if missing:
            raise ValueError(f"Artifact {path} is missing arrays: {missing}")
        return {name: tuple(artifact[name].shape) for name in required_arrays}
