"""Prediction logging (SQLite), drift detection (PSI) and live performance.

Drift method - Population Stability Index (PSI), per feature:
    bins come from quantiles of the training (reference) data;
    PSI = sum((cur% - ref%) * ln(cur% / ref%)) over the bins.
    < 0.1 stable, 0.1-0.25 moderate shift, >= 0.25 significant shift.
Limits: it checks one feature at a time, needs enough samples, and drift only says
"inputs look different" - it does not prove the model became less accurate. Accuracy
can only be measured once real outcomes are recorded (see ``performance``).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generate_data import TARGET
from src.data.validate_data import FEATURE_COLUMNS
from src.models.evaluate import compute_metrics
from src.models.registry import utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    model_version TEXT NOT NULL,
    machine_id TEXT,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    risk TEXT NOT NULL,
    threshold REAL NOT NULL,
    latency_ms REAL NOT NULL,
    features TEXT NOT NULL,
    actual INTEGER,
    outcome_ts TEXT,
    outcome_source TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions (ts);
CREATE TABLE IF NOT EXISTS api_errors (
    ts TEXT NOT NULL, path TEXT NOT NULL, status INTEGER NOT NULL, detail TEXT
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_predictions(db_path: Path, rows: list[dict]) -> list[str]:
    ids = []
    with closing(connect(db_path)) as conn, conn:
        for row in rows:
            pid = row.get("id") or uuid.uuid4().hex
            ids.append(pid)
            conn.execute(
                "INSERT INTO predictions (id, ts, model_version, machine_id, prediction, probability,"
                " risk, threshold, latency_ms, features) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    pid, row.get("ts") or utc_now(), row["model_version"], row.get("machine_id"),
                    int(row["prediction"]), float(row["probability"]), row["risk"],
                    float(row["threshold"]), float(row["latency_ms"]),
                    json.dumps({c: row["features"][c] for c in FEATURE_COLUMNS}),
                ),
            )
    return ids


def record_outcomes(db_path: Path, outcomes: list[tuple[str, int]], source: str) -> int:
    """Attach real (or clearly-labelled simulated) outcomes to logged predictions."""
    now = utc_now()
    with closing(connect(db_path)) as conn, conn:
        updated = 0
        for prediction_id, actual in outcomes:
            cur = conn.execute(
                "UPDATE predictions SET actual=?, outcome_ts=?, outcome_source=? WHERE id=?",
                (int(actual), now, source, prediction_id),
            )
            updated += cur.rowcount
        return updated


def record_outcome(db_path: Path, prediction_id: str, actual: int, source: str) -> bool:
    return record_outcomes(db_path, [(prediction_id, actual)], source) == 1


def log_error(db_path: Path, path: str, status: int, detail: str) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.execute("INSERT INTO api_errors VALUES (?,?,?,?)", (utc_now(), path, status, detail[:500]))


def recent_predictions(db_path: Path, limit: int = 50) -> list[dict]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "features": json.loads(r["features"])} for r in rows]


def _frame(rows: list[sqlite3.Row]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "id", "ts", "machine_id", "actual"])
    feats = pd.DataFrame([json.loads(r["features"]) for r in rows])
    meta = pd.DataFrame([{k: r[k] for k in r.keys() if k != "features"} for r in rows])
    return pd.concat([meta, feats], axis=1)


def prediction_frame(db_path: Path, limit: int = 1000) -> pd.DataFrame:
    with closing(connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return _frame(rows)


def labelled_frame(db_path: Path, since: str | None = None) -> pd.DataFrame:
    """Logged predictions with outcomes, as rows shaped like the training data."""
    sql = "SELECT * FROM predictions WHERE actual IS NOT NULL"
    args: tuple = ()
    if since:
        sql += " AND outcome_ts > ?"
        args = (since,)
    with closing(connect(db_path)) as conn:
        df = _frame(conn.execute(sql + " ORDER BY ts", args).fetchall())
    if df.empty:
        return df
    df[TARGET] = df["actual"].astype(int)
    df["machine_id"] = df["machine_id"].fillna("unknown")
    df["timestamp"] = df["ts"]
    return df


def activity_summary(db_path: Path) -> dict:
    with closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, AVG(latency_ms) lat, AVG(probability) p, SUM(prediction) pos,"
            " SUM(actual IS NOT NULL) labelled, MAX(ts) last FROM predictions"
        ).fetchone()
        by_risk = dict(conn.execute("SELECT risk, COUNT(*) FROM predictions GROUP BY risk").fetchall())
        errors = conn.execute("SELECT COUNT(*) FROM api_errors").fetchone()[0]
        recent_errors = [dict(r) for r in conn.execute(
            "SELECT * FROM api_errors ORDER BY ts DESC LIMIT 10").fetchall()]
    return {
        "total_predictions": row["n"],
        "predicted_failures": int(row["pos"] or 0),
        "avg_probability": row["p"],
        "avg_latency_ms": row["lat"],
        "labelled_outcomes": int(row["labelled"] or 0),
        "last_prediction_at": row["last"],
        "by_risk": by_risk,
        "api_errors": errors,
        "recent_errors": recent_errors,
    }


def psi(reference: pd.Series, current: pd.Series, bins: int = 10, categorical: bool = False) -> float:
    eps = 1e-6
    if categorical:
        cats = sorted(set(reference) | set(current))
        ref = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        cur = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(reference.astype(float), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(reference.astype(float), edges)[0] / len(reference)
        cur = np.histogram(current.astype(float), edges)[0] / len(current)
    ref, cur = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict:
    if len(current) < cfg["drift_min_samples"]:
        return {"status": "insufficient_data", "samples": len(current),
                "required": cfg["drift_min_samples"], "features": {}}
    features = {}
    for col in FEATURE_COLUMNS:
        value = psi(reference[col], current[col], cfg["psi_bins"], categorical=col == "machine_type")
        level = "significant" if value >= cfg["psi_alert"] else "moderate" if value >= cfg["psi_warn"] else "stable"
        features[col] = {
            "psi": round(value, 4),
            "level": level,
            **({} if col == "machine_type" else {
                "reference_mean": round(float(reference[col].mean()), 3),
                "current_mean": round(float(current[col].mean()), 3),
            }),
        }
    levels = {f["level"] for f in features.values()}
    status = "significant" if "significant" in levels else "moderate" if "moderate" in levels else "stable"
    return {
        "status": status,
        "samples": len(current),
        "drifted_features": [c for c, f in features.items() if f["level"] != "stable"],
        "features": features,
        "method": "PSI per feature vs training data",
        "note": "Drift means inputs changed; it does not by itself prove accuracy dropped.",
    }


def performance(db_path: Path, min_labels: int, model_version: str | None = None) -> dict:
    """Live precision/recall/F1 using recorded outcomes (each row's own threshold)."""
    sql = "SELECT probability, threshold, actual, outcome_source FROM predictions WHERE actual IS NOT NULL"
    args: tuple = ()
    if model_version:
        sql += " AND model_version = ?"
        args = (model_version,)
    with closing(connect(db_path)) as conn:
        rows = conn.execute(sql, args).fetchall()
    if len(rows) < min_labels:
        return {"status": "insufficient_data", "labelled": len(rows), "required": min_labels}
    proba = np.array([r["probability"] for r in rows])
    thr = np.array([r["threshold"] for r in rows])
    actual = np.array([r["actual"] for r in rows])
    # Shift probabilities so a single threshold of 0.5 reproduces each row's decision.
    metrics = compute_metrics(actual, np.clip(proba - thr + 0.5, 0, 1), 0.5)
    metrics["roc_auc"] = compute_metrics(actual, proba, 0.5)["roc_auc"]
    metrics.pop("threshold")
    sources = sorted({r["outcome_source"] for r in rows})
    return {"status": "ok", "labelled": len(rows), "outcome_sources": sources, "metrics": metrics}
