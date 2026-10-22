"""Controlled retraining: train a candidate, compare it with the served model on the
SAME evaluation data, and promote it only if the acceptance rules pass.

Evaluation set = original held-out test set + a share of the new labelled rows.
Training set   = original training set + the rest of the new labelled rows.
The previous model is never deleted, so a promotion can always be rolled back.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from src.data.validate_data import DataValidationError, validate_dataframe
from src.features.build_features import split_dataset
from src.models import registry
from src.models.evaluate import evaluate_model
from src.models.registry import ModelNotFoundError, utc_now
from src.models.train import train_and_register

log = logging.getLogger(__name__)


class RetrainInProgressError(RuntimeError):
    pass


class RetrainLock:
    """Cross-process lock: a file created with O_EXCL. Stale locks expire after ``timeout``."""

    def __init__(self, path: Path, timeout_seconds: float):
        self.path = path
        self.timeout = timeout_seconds
        self.held = False

    def acquire(self) -> RetrainLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() - self.path.stat().st_mtime > self.timeout:
                    log.warning("Removing stale retraining lock %s", self.path)
                    self.path.unlink(missing_ok=True)
                    continue
                raise RetrainInProgressError("a retraining run is already in progress") from None
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
            self.held = True
            return self
        raise RetrainInProgressError("could not acquire retraining lock")

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self) -> RetrainLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def lock_path(settings) -> Path:
    return settings.artifacts_dir / ".retrain.lock"


def decide(current: dict | None, candidate: dict, cfg: dict) -> tuple[bool, list[str]]:
    """Acceptance rules. Returns (accepted, reasons)."""
    reasons = []
    if candidate["f1"] < cfg["min_f1"]:
        reasons.append(f"candidate F1 {candidate['f1']:.3f} below minimum {cfg['min_f1']}")
    if current is not None:
        if candidate["f1"] < current["f1"] + cfg["min_f1_gain"]:
            reasons.append(
                f"candidate F1 {candidate['f1']:.3f} not >= current {current['f1']:.3f} + {cfg['min_f1_gain']}"
            )
        if candidate["recall"] < current["recall"] - cfg["max_recall_drop"]:
            reasons.append(
                f"candidate recall {candidate['recall']:.3f} dropped more than "
                f"{cfg['max_recall_drop']} below current {current['recall']:.3f}"
            )
    accepted = not reasons
    if accepted:
        reasons.append("all acceptance checks passed")
    return accepted, reasons


def _save_report(settings, report: dict) -> Path:
    folder = settings.reports_dir / "retraining"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with open(settings.reports_dir / "retraining_history.jsonl", "a", encoding="utf-8") as fh:
        summary = {k: report.get(k) for k in (
            "run_id", "started_at", "finished_at", "trigger", "status", "candidate_version",
            "current_version", "decision_reasons", "new_labelled_rows")}
        fh.write(json.dumps(summary, default=str) + "\n")
    return path


def run_retraining(settings, trigger: str, new_data: pd.DataFrame, lock: RetrainLock | None = None) -> dict:
    """Run one retraining cycle. ``lock`` may be pre-acquired by the caller."""
    cfg = settings.params["retraining"]
    lock = lock or RetrainLock(lock_path(settings), cfg["lock_timeout_seconds"]).acquire()
    report: dict = {
        "run_id": f"retrain-{registry.new_version()[1:]}",
        "started_at": utc_now(),
        "trigger": trigger,
        "new_labelled_rows": len(new_data),
    }
    try:
        base_train = pd.read_csv(settings.train_csv)
        base_test = pd.read_csv(settings.test_csv)
        if not new_data.empty:
            validate_dataframe(new_data, check_duplicates=False)
            new_data = new_data[base_train.columns]
            if len(new_data) >= 10:
                new_train, new_eval = split_dataset(new_data, cfg["new_data_eval_share"], seed=0)
            else:
                new_train, new_eval = new_data, new_data.iloc[0:0]
        else:
            new_train = new_eval = base_train.iloc[0:0]
        train_df = pd.concat([base_train, new_train], ignore_index=True)
        eval_df = pd.concat([base_test, new_eval], ignore_index=True)
        report.update(train_rows=len(train_df), eval_rows=len(eval_df))

        # Duplicates were checked on the base data; new rows are keyed by unique prediction id.
        candidate = train_and_register(
            train_df, eval_df, settings, run_name=report["run_id"], check_duplicates=False
        )
        cand_version = candidate["model_version"]
        report["candidate_version"] = cand_version
        report["candidate_metrics"] = candidate["metrics"]

        try:
            current_model, current_meta = registry.load_current(settings.artifacts_dir)
            report["current_version"] = current_meta["model_version"]
            report["current_metrics"] = evaluate_model(current_model, eval_df, current_meta["threshold"])
        except ModelNotFoundError:
            report["current_version"] = None
            report["current_metrics"] = None

        accepted, reasons = decide(report["current_metrics"], candidate["metrics"], cfg)
        report["decision_reasons"] = reasons
        if accepted:
            registry.promote(cand_version, settings.artifacts_dir, reason=f"retraining: {trigger}")
            registry.update_metadata(cand_version, settings.artifacts_dir, status="promoted")
            report["status"] = "promoted"
            log.info("Promoted %s (previous %s kept for rollback)", cand_version, report["current_version"])
        else:
            registry.update_metadata(cand_version, settings.artifacts_dir, status="rejected",
                                     rejection_reasons=reasons)
            report["status"] = "rejected"
            log.info("Rejected candidate %s: %s", cand_version, reasons)
    except DataValidationError as exc:
        report.update(status="failed_validation", decision_reasons=exc.errors)
        log.error("Retraining aborted, new data invalid: %s", exc)
    except Exception as exc:  # noqa: BLE001 - recorded in the report, served model untouched
        report.update(status="failed", decision_reasons=[f"{type(exc).__name__}: {exc}"])
        log.exception("Retraining failed; served model unchanged")
    finally:
        report["finished_at"] = utc_now()
        lock.release()
    report["report_path"] = str(_save_report(settings, report))
    return report
