"""Tiny file-based model registry.

    artifacts/models/<version>/model.joblib    fitted scikit-learn Pipeline
    artifacts/models/<version>/metadata.json   version, metrics, fingerprint, ...
    artifacts/current.json                     pointer: which version is served

Versions are written to a temporary folder and renamed into place, and the pointer is
replaced with ``os.replace``, so readers never see a half-written model. Old versions
are never deleted, which is what makes rollback possible.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib


class ModelNotFoundError(FileNotFoundError):
    pass


def _models_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / "models"


def _pointer(artifacts_dir: Path) -> Path:
    return artifacts_dir / "current.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_version() -> str:
    return f"v{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_version(pipeline, metadata: dict, artifacts_dir: Path) -> str:
    version = metadata["model_version"]
    final = _models_dir(artifacts_dir) / version
    tmp = _models_dir(artifacts_dir) / f".tmp-{version}"
    tmp.mkdir(parents=True)
    joblib.dump(pipeline, tmp / "model.joblib")
    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(tmp, final)
    return version


def update_metadata(version: str, artifacts_dir: Path, **fields) -> dict:
    path = _models_dir(artifacts_dir) / version / "metadata.json"
    meta = {**json.loads(path.read_text(encoding="utf-8")), **fields}
    _atomic_write_json(path, meta)
    return meta


def read_pointer(artifacts_dir: Path) -> dict | None:
    path = _pointer(artifacts_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def current_version(artifacts_dir: Path) -> str | None:
    pointer = read_pointer(artifacts_dir)
    return pointer["current"] if pointer else None


def load_version(version: str, artifacts_dir: Path):
    folder = _models_dir(artifacts_dir) / version
    if not (folder / "model.joblib").exists():
        raise ModelNotFoundError(f"model version '{version}' not found in {folder.parent}")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    # joblib uses pickle: only load files this project wrote itself, never uploaded ones.
    return joblib.load(folder / "model.joblib"), metadata


def load_current(artifacts_dir: Path):
    version = current_version(artifacts_dir)
    if version is None:
        raise ModelNotFoundError("no promoted model yet - run `python scripts/setup.py` first")
    return load_version(version, artifacts_dir)


def promote(version: str, artifacts_dir: Path, reason: str) -> dict:
    if not (_models_dir(artifacts_dir) / version / "model.joblib").exists():
        raise ModelNotFoundError(f"cannot promote missing version '{version}'")
    old = read_pointer(artifacts_dir) or {"current": None, "history": []}
    event = {"version": version, "promoted_at": utc_now(), "reason": reason}
    pointer = {
        "current": version,
        "previous": old["current"],
        "promoted_at": event["promoted_at"],
        "reason": reason,
        "history": [*old.get("history", []), event],
    }
    _atomic_write_json(_pointer(artifacts_dir), pointer)
    return pointer


def rollback(artifacts_dir: Path) -> dict:
    pointer = read_pointer(artifacts_dir)
    if not pointer or not pointer.get("previous"):
        raise ModelNotFoundError("no previous model version to roll back to")
    return promote(pointer["previous"], artifacts_dir, reason=f"rollback from {pointer['current']}")


def list_versions(artifacts_dir: Path) -> list[dict]:
    root = _models_dir(artifacts_dir)
    if not root.exists():
        return []
    metas = [
        json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith(".") and (d / "metadata.json").exists()
    ]
    return sorted(metas, key=lambda m: m["trained_at"])
