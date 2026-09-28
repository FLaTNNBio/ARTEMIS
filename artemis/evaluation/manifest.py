from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, arguments: Iterable[str]) -> Optional[str]:
    command = [
        "git", "-c", f"safe.directory={repo_root.as_posix()}", *arguments
    ]
    completed = subprocess.run(
        command, cwd=repo_root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_environment() -> Dict[str, Any]:
    versions: Dict[str, Optional[str]] = {}
    for module_name in ("numpy", "pandas", "scipy", "sklearn", "torch"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": versions,
    }


def build_run_manifest(
    repo_root: Path,
    config: Mapping[str, Any],
    dataset_paths: Iterable[Path],
    device: str,
    started_at: str,
    ended_at: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    status = _git(repo_root, ["status", "--porcelain"])
    manifest = {
        "git_commit": _git(repo_root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
        "git_status_porcelain": status,
        "config": dict(config),
        "command": list(sys.argv),
        "working_directory": os.getcwd(),
        "environment": collect_environment(),
        "device": device,
        "dataset_hashes": {
            str(Path(path)): sha256_file(path) for path in dataset_paths
        },
        "start_timestamp_utc": started_at,
        "end_timestamp_utc": ended_at,
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, default=str) + "\n", encoding="utf-8")
