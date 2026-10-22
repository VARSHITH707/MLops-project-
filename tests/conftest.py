"""Shared fixtures. Every test runs in its own temporary PM_HOME, never the real project data."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import PROJECT_ROOT, get_settings

API_KEY = "test-admin-key"

SAMPLE = {
    "machine_id": "M-007",
    "machine_type": "L",
    "air_temperature": 302.5,
    "process_temperature": 310.9,
    "rotational_speed": 1380,
    "torque": 58.4,
    "tool_wear": 215,
    "operating_hours": 15200,
    "maintenance_history_count": 5,
}


@pytest.fixture
def settings(tmp_path: Path, monkeypatch):
    """Isolated, small and fast configuration."""
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
    params["data"]["n_samples"] = 2000
    params["train"]["random_forest"]["n_estimators"] = 40
    params["monitoring"]["drift_min_samples"] = 30
    params["retraining"]["min_new_labels"] = 50
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.setenv("PM_HOME", str(tmp_path))
    monkeypatch.setenv("PM_PARAMS_FILE", str(tmp_path / "params.yaml"))
    monkeypatch.setenv("PM_DISABLE_MLFLOW", "1")
    monkeypatch.setenv("PM_ADMIN_API_KEY", API_KEY)
    monkeypatch.delenv("PM_SCHEDULER_ENABLED", raising=False)
    return get_settings()


@pytest.fixture
def trained(settings):
    """Run the full first-time pipeline (generate -> validate -> split -> train + promote)."""
    from scripts import setup

    assert setup.main([]) == 0
    return settings


@pytest.fixture
def client(trained):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
