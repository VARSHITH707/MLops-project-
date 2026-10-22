"""Loads the served model once and reloads it automatically after a promotion/rollback."""

from __future__ import annotations

import threading
import time
import uuid

import pandas as pd

from src.models import registry
from src.models.registry import ModelNotFoundError, utc_now

_lock = threading.Lock()
_cache: dict = {"key": None, "model": None, "meta": None}


def get_model(settings):
    """Return (pipeline, metadata) for the version currently named in artifacts/current.json."""
    version = registry.current_version(settings.artifacts_dir)
    if version is None:
        raise ModelNotFoundError("no trained model is available - run `python scripts/setup.py`")
    key = (str(settings.artifacts_dir), version)
    with _lock:
        if _cache["key"] != key:
            model, meta = registry.load_version(version, settings.artifacts_dir)
            _cache.update(key=key, model=model, meta=meta)
        return _cache["model"], _cache["meta"]


def risk_category(probability: float, threshold: float, medium: float) -> str:
    if probability >= threshold:
        return "high"
    return "medium" if probability >= medium else "low"


def predict(records: list[dict], settings) -> list[dict]:
    model, meta = get_model(settings)
    start = time.perf_counter()
    proba = model.predict_proba(pd.DataFrame(records))[:, 1]
    latency = (time.perf_counter() - start) * 1000 / len(records)
    threshold = meta["threshold"]
    medium = settings.params["risk"]["medium"]
    now = utc_now()
    return [
        {
            "id": uuid.uuid4().hex,
            "ts": now,
            "model_version": meta["model_version"],
            "machine_id": rec.get("machine_id"),
            "prediction": int(p >= threshold),
            "probability": float(p),
            "risk": risk_category(float(p), threshold, medium),
            "threshold": threshold,
            "latency_ms": latency,
            "features": rec,
        }
        for rec, p in zip(records, proba, strict=True)
    ]
