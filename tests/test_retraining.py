import pandas as pd
import pytest

from app import retraining
from src.data.generate_data import simulate_new_observations
from src.models import registry
from src.pipelines import retrain_pipeline
from src.pipelines.retrain_pipeline import RetrainInProgressError, RetrainLock, decide, run_retraining
from tests.conftest import API_KEY

H = {"X-API-Key": API_KEY}
CFG = {"min_f1": 0.5, "min_f1_gain": 0.0, "max_recall_drop": 0.02}


def test_decide_rules():
    cur = {"f1": 0.70, "recall": 0.85}
    assert decide(cur, {"f1": 0.72, "recall": 0.86}, CFG)[0]
    assert not decide(cur, {"f1": 0.65, "recall": 0.90}, CFG)[0]  # worse F1
    assert not decide(cur, {"f1": 0.75, "recall": 0.80}, CFG)[0]  # recall dropped too much
    assert not decide(None, {"f1": 0.40, "recall": 0.9}, CFG)[0]  # below absolute minimum
    assert decide(None, {"f1": 0.60, "recall": 0.9}, CFG)[0]


def test_lock_blocks_second_run_and_releases(tmp_path):
    path = tmp_path / ".lock"
    with RetrainLock(path, 60):
        with pytest.raises(RetrainInProgressError):
            RetrainLock(path, 60).acquire()
    assert not path.exists()
    RetrainLock(path, 60).acquire().release()


def test_stale_lock_is_reclaimed(tmp_path):
    path = tmp_path / ".lock"
    path.write_text("old")
    lock = RetrainLock(path, timeout_seconds=-1).acquire()  # anything is "stale"
    assert lock.held
    lock.release()


def test_candidate_promoted_and_previous_preserved(trained, monkeypatch):
    s = trained
    old = registry.current_version(s.artifacts_dir)
    monkeypatch.setattr(retrain_pipeline, "decide", lambda *a: (True, ["forced accept for test"]))
    report = run_retraining(s, "test", simulate_new_observations(100, seed=9))
    assert report["status"] == "promoted"
    assert registry.current_version(s.artifacts_dir) == report["candidate_version"] != old
    assert registry.read_pointer(s.artifacts_dir)["previous"] == old
    registry.load_version(old, s.artifacts_dir)  # old model still loadable
    registry.rollback(s.artifacts_dir)
    assert registry.current_version(s.artifacts_dir) == old


def test_candidate_rejected_keeps_current(trained, monkeypatch):
    s = trained
    old = registry.current_version(s.artifacts_dir)
    monkeypatch.setattr(retrain_pipeline, "decide", lambda *a: (False, ["forced reject for test"]))
    report = run_retraining(s, "test", simulate_new_observations(100, seed=9))
    assert report["status"] == "rejected"
    assert registry.current_version(s.artifacts_dir) == old
    _, meta = registry.load_version(report["candidate_version"], s.artifacts_dir)
    assert meta["status"] == "rejected"
    assert retraining.history(s)[0]["status"] == "rejected"


def test_invalid_new_data_does_not_touch_model(trained):
    s = trained
    old = registry.current_version(s.artifacts_dir)
    bad = simulate_new_observations(20, seed=1)
    bad.loc[0, "torque"] = -100
    report = run_retraining(s, "test", bad)
    assert report["status"] == "failed_validation"
    assert registry.current_version(s.artifacts_dir) == old
    assert not retrain_pipeline.lock_path(s).exists()


def test_training_error_releases_lock(trained, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(retrain_pipeline, "train_and_register", boom)
    report = run_retraining(trained, "test", pd.DataFrame())
    assert report["status"] == "failed"
    assert not retrain_pipeline.lock_path(trained).exists()


def test_triggers_and_api_flow(client, trained):
    assert not client.get("/retrain/status").json()["triggers"]["should_retrain"]
    client.post("/admin/simulate", json={"n": 80, "seed": 4}, headers=H)
    triggers = client.get("/retrain/status").json()["triggers"]
    assert triggers["should_retrain"] and triggers["new_labels"] == 80

    r = client.post("/admin/retrain", json={"reason": "test", "wait": True}, headers=H)
    assert r.status_code == 202 and r.json()["status"] in {"promoted", "rejected"}
    assert r.json()["current_metrics"] is not None  # compared against the served model
    status = client.get("/retrain/status").json()
    assert status["last_run"]["run_id"] == r.json()["run_id"]
    assert status["triggers"]["new_labels"] == 0  # counter resets after a run


def test_retrain_conflict_while_locked(client, trained):
    with RetrainLock(retrain_pipeline.lock_path(trained), 60):
        assert client.post("/admin/retrain", json={}, headers=H).status_code == 409


def test_rollback_without_previous_is_409(client):
    assert client.post("/admin/rollback", headers=H).status_code == 409


def test_scheduler_runs_in_one_process_only(trained, monkeypatch):
    monkeypatch.setenv("PM_SCHEDULER_ENABLED", "1")
    from app.config import get_settings

    s = get_settings()
    monkeypatch.setattr(retraining, "_scheduler_loop", lambda *a: None)
    assert retraining.start_scheduler(s) is True
    assert retraining.start_scheduler(s) is False  # second "worker" backs off
