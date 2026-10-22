"""Retraining triggers, background execution, status, and the optional scheduler."""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from app import monitoring
from src.models import registry
from src.models.registry import utc_now
from src.pipelines.retrain_pipeline import (
    RetrainInProgressError,
    RetrainLock,
    lock_path,
    run_retraining,
)

log = logging.getLogger(__name__)


def history(settings, limit: int = 20) -> list[dict]:
    path = settings.reports_dir / "retraining_history.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:]][::-1]


def _last_run_time(settings) -> str | None:
    runs = history(settings, limit=1)
    return runs[0]["finished_at"] if runs else None


def check_triggers(settings) -> dict:
    """Decide whether retraining is warranted. Drift alone only raises an advisory."""
    cfg = settings.params["retraining"]
    mon = settings.params["monitoring"]
    new_labels = len(monitoring.labelled_frame(settings.db_path, since=_last_run_time(settings)))
    reasons = []
    if new_labels >= cfg["min_new_labels"]:
        reasons.append(f"{new_labels} new labelled outcomes (>= {cfg['min_new_labels']})")

    current = registry.current_version(settings.artifacts_dir)
    perf = monitoring.performance(settings.db_path, mon["performance_min_labels"], current)
    if perf["status"] == "ok" and current:
        _, meta = registry.load_version(current, settings.artifacts_dir)
        drop = meta["metrics"]["f1"] - perf["metrics"]["f1"]
        perf["f1_drop_vs_training"] = round(drop, 4)
        if drop >= cfg["degradation_f1_drop"]:
            reasons.append(f"live F1 is {drop:.3f} below training F1 (>= {cfg['degradation_f1_drop']})")
    return {"should_retrain": bool(reasons), "reasons": reasons, "new_labels": new_labels, "performance": perf}


def is_running(settings) -> bool:
    return lock_path(settings).exists()


def status(settings) -> dict:
    runs = history(settings)
    return {
        "in_progress": is_running(settings),
        "last_run": runs[0] if runs else None,
        "history": runs,
        "triggers": check_triggers(settings),
        "scheduler_enabled": settings.scheduler_enabled,
    }


def start(settings, reason: str, wait: bool = False) -> dict:
    """Acquire the lock now (so a second caller gets an immediate error), then run."""
    lock = RetrainLock(lock_path(settings), settings.params["retraining"]["lock_timeout_seconds"]).acquire()
    try:
        new_data = monitoring.labelled_frame(settings.db_path)
    except Exception:
        lock.release()
        raise
    if wait:
        return run_retraining(settings, reason, new_data, lock)
    threading.Thread(
        target=run_retraining, args=(settings, reason, new_data, lock), daemon=True, name="retraining"
    ).start()
    return {"status": "started", "trigger": reason, "started_at": utc_now()}


def _scheduler_loop(settings, lock_file) -> None:
    interval = settings.params["retraining"]["scheduler_interval_seconds"]
    while True:
        os.utime(lock_file)  # heartbeat: proves this worker still owns the scheduler
        try:
            triggers = check_triggers(settings)
            if triggers["should_retrain"]:
                start(settings, "scheduled: " + "; ".join(triggers["reasons"]), wait=True)
        except RetrainInProgressError:
            log.info("Scheduler skipped: retraining already running")
        except Exception:  # noqa: BLE001 - keep the scheduler alive
            log.exception("Scheduled trigger check failed")
        time.sleep(interval)


def start_scheduler(settings) -> bool:
    """Start the condition-based scheduler in at most one process (lock file + heartbeat)."""
    if not settings.scheduler_enabled:
        return False
    interval = settings.params["retraining"]["scheduler_interval_seconds"]
    lock = RetrainLock(settings.artifacts_dir / ".scheduler.lock", timeout_seconds=3 * interval)
    try:
        lock.acquire()
    except RetrainInProgressError:
        log.info("Another worker owns the retraining scheduler")
        return False
    threading.Thread(target=_scheduler_loop, args=(settings, lock.path), daemon=True, name="scheduler").start()
    log.info("Retraining scheduler started (every %ss)", interval)
    return True
