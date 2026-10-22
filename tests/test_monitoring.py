import numpy as np
import pandas as pd

from app import monitoring
from src.data.generate_data import simulate_new_observations
from tests.conftest import API_KEY

CFG = {"drift_min_samples": 30, "psi_bins": 10, "psi_warn": 0.1, "psi_alert": 0.25}


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert monitoring.psi(ref, ref) < 1e-9
    assert monitoring.psi(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert monitoring.psi(ref, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25


def test_psi_categorical():
    ref = pd.Series(["L"] * 50 + ["M"] * 30 + ["H"] * 20)
    assert monitoring.psi(ref, ref, categorical=True) < 1e-9
    assert monitoring.psi(ref, pd.Series(["H"] * 100), categorical=True) > 0.25


def test_drift_report_levels():
    ref = simulate_new_observations(2000, seed=1)
    assert monitoring.drift_report(ref, simulate_new_observations(500, seed=2), CFG)["status"] == "stable"
    drifted = monitoring.drift_report(ref, simulate_new_observations(500, seed=2, drift=2.0), CFG)
    assert drifted["status"] == "significant"
    assert "air_temperature" in drifted["drifted_features"]
    assert monitoring.drift_report(ref, ref.head(5), CFG)["status"] == "insufficient_data"


def test_performance_needs_enough_labels(tmp_path):
    db = tmp_path / "m.db"
    assert monitoring.performance(db, min_labels=5)["status"] == "insufficient_data"


def test_performance_matches_logged_decisions(tmp_path):
    db = tmp_path / "m.db"
    feats = simulate_new_observations(4, seed=0).iloc[0].to_dict()
    rows = [
        {"model_version": "v1", "prediction": int(p >= 0.4), "probability": p, "risk": "low",
         "threshold": 0.4, "latency_ms": 1.0, "features": feats}
        for p in (0.9, 0.5, 0.3, 0.1)
    ]
    ids = monitoring.log_predictions(db, rows)
    monitoring.record_outcomes(db, list(zip(ids, [1, 0, 1, 0], strict=True)), "simulated")
    perf = monitoring.performance(db, min_labels=4)
    cm = perf["metrics"]["confusion_matrix"]
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert perf["outcome_sources"] == ["simulated"]


def test_monitoring_endpoints(client):
    assert client.get("/monitoring/drift").json()["status"] == "insufficient_data"
    r = client.post("/admin/simulate", json={"n": 100, "seed": 3, "drift": 2.0},
                    headers={"X-API-Key": API_KEY})
    assert r.status_code == 200 and r.json()["simulated_outcomes_recorded"] == 100
    assert client.get("/monitoring/drift").json()["status"] == "significant"
    perf = client.get("/monitoring/performance").json()
    assert perf["status"] == "ok" and perf["labelled"] == 100
    summary = client.get("/monitoring/summary").json()
    assert summary["total_predictions"] == 100 and summary["labelled_outcomes"] == 100


def test_errors_are_recorded(client):
    client.post("/predict", json={"machine_type": "Z"})
    assert client.get("/monitoring/summary").json()["api_errors"] >= 1
