"""FastAPI prediction service.  Run:  python -m uvicorn app.main:app --port 8000
Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import Counter
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app import monitoring, predictor, retraining
from app.config import Settings, get_settings
from app.schemas import (
    BatchRequest,
    BatchResponse,
    OutcomeRequest,
    PredictionResponse,
    RetrainRequest,
    SensorReading,
    SimulateRequest,
)
from src.data.generate_data import TARGET, simulate_new_observations
from src.data.validate_data import FEATURE_COLUMNS
from src.models import registry
from src.models.registry import ModelNotFoundError
from src.pipelines.retrain_pipeline import RetrainInProgressError

log = logging.getLogger("api")
STARTED = time.time()
REQUESTS: Counter = Counter()


def require_admin(settings: Settings = Depends(get_settings), x_api_key: str | None = Header(None)) -> None:
    if not settings.admin_api_key:
        raise HTTPException(503, "admin actions disabled: set the PM_ADMIN_API_KEY environment variable")
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.admin_api_key):
        raise HTTPException(401, "missing or invalid X-API-Key header")


def _response(row: dict) -> PredictionResponse:
    return PredictionResponse(
        prediction_id=row["id"],
        failure_predicted=bool(row["prediction"]),
        failure_probability=round(row["probability"], 4),
        risk_category=row["risk"],
        threshold=row["threshold"],
        model_version=row["model_version"],
        timestamp=row["ts"],
        latency_ms=round(row["latency_ms"], 3),
    )


def _predict_and_log(records: list[dict], settings: Settings) -> list[dict]:
    try:
        rows = predictor.predict(records, settings)
    except ModelNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    monitoring.log_predictions(settings.db_path, rows)
    return rows


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings = get_settings()
        monitoring.connect(settings.db_path).close()  # create tables
        retraining.start_scheduler(settings)
        yield

    app = FastAPI(
        title="Predictive Maintenance API",
        description="Estimates machine failure risk (next 24 operating hours) from sensor readings. "
        "Educational simulation trained on synthetic data - not a certified safety system.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        response = await call_next(request)
        REQUESTS[(request.url.path, response.status_code)] += 1
        if response.status_code >= 400 and request.url.path not in ("/favicon.ico",):
            try:
                monitoring.log_error(get_settings().db_path, request.url.path, response.status_code, "")
            except Exception:  # noqa: BLE001 - never fail a request because logging failed
                log.exception("could not record API error")
        return response

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": f"internal error: {type(exc).__name__}"})

    @app.get("/health", tags=["service"])
    def health(settings: Settings = Depends(get_settings)):
        version = registry.current_version(settings.artifacts_dir)
        return {
            "status": "ok" if version else "degraded",
            "model_loaded": version is not None,
            "model_version": version,
            "uptime_seconds": round(time.time() - STARTED, 1),
            "detail": None if version else "no trained model - run `python scripts/setup.py`",
        }

    @app.get("/model-info", tags=["service"])
    def model_info(settings: Settings = Depends(get_settings)):
        try:
            _, meta = predictor.get_model(settings)
        except ModelNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "metadata": meta,
            "pointer": registry.read_pointer(settings.artifacts_dir),
            "versions": [
                {k: v.get(k) for k in ("model_version", "model_type", "trained_at", "status", "run_name")}
                | {"f1": v["metrics"]["f1"], "recall": v["metrics"]["recall"]}
                for v in registry.list_versions(settings.artifacts_dir)
            ],
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(reading: SensorReading, settings: Settings = Depends(get_settings)):
        return _response(_predict_and_log([reading.model_dump()], settings)[0])

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
    def predict_batch(batch: BatchRequest, settings: Settings = Depends(get_settings)):
        rows = _predict_and_log([r.model_dump() for r in batch.readings], settings)
        return BatchResponse(predictions=[_response(r) for r in rows])

    @app.post("/outcomes", tags=["monitoring"])
    def record_outcome(body: OutcomeRequest, settings: Settings = Depends(get_settings)):
        if not monitoring.record_outcome(settings.db_path, body.prediction_id, body.actual_failure, body.source):
            raise HTTPException(404, f"prediction '{body.prediction_id}' not found")
        return {"recorded": True, "prediction_id": body.prediction_id, "source": body.source}

    @app.get("/predictions/recent", tags=["monitoring"])
    def recent(limit: int = 50, settings: Settings = Depends(get_settings)):
        return monitoring.recent_predictions(settings.db_path, max(1, min(limit, 500)))

    @app.get("/monitoring/summary", tags=["monitoring"])
    def summary(settings: Settings = Depends(get_settings)):
        return monitoring.activity_summary(settings.db_path)

    @app.get("/monitoring/drift", tags=["monitoring"])
    def drift(window: int = 500, settings: Settings = Depends(get_settings)):
        if not settings.train_csv.exists():
            raise HTTPException(503, "reference (training) data not found - run `python scripts/setup.py`")
        reference = pd.read_csv(settings.train_csv, usecols=FEATURE_COLUMNS)
        current = monitoring.prediction_frame(settings.db_path, limit=max(1, min(window, 5000)))
        return monitoring.drift_report(reference, current, settings.params["monitoring"])

    @app.get("/monitoring/performance", tags=["monitoring"])
    def performance(settings: Settings = Depends(get_settings)):
        return monitoring.performance(
            settings.db_path,
            settings.params["monitoring"]["performance_min_labels"],
            registry.current_version(settings.artifacts_dir),
        )

    @app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
    def metrics(settings: Settings = Depends(get_settings)):
        """Prometheus text format."""
        s = monitoring.activity_summary(settings.db_path)
        lines = [
            "# TYPE pm_http_requests_total counter",
            *(f'pm_http_requests_total{{path="{p}",status="{c}"}} {n}' for (p, c), n in sorted(REQUESTS.items())),
            "# TYPE pm_predictions_total counter",
            f"pm_predictions_total {s['total_predictions']}",
            f"pm_predicted_failures_total {s['predicted_failures']}",
            f"pm_labelled_outcomes_total {s['labelled_outcomes']}",
            f"pm_api_errors_total {s['api_errors']}",
            f"pm_avg_latency_ms {s['avg_latency_ms'] or 0:.4f}",
            f"pm_model_loaded {int(registry.current_version(settings.artifacts_dir) is not None)}",
            f"pm_uptime_seconds {time.time() - STARTED:.1f}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/retrain/status", tags=["retraining"])
    def retrain_status(settings: Settings = Depends(get_settings)):
        return retraining.status(settings)

    @app.post("/admin/retrain", tags=["admin"], dependencies=[Depends(require_admin)], status_code=202)
    def retrain(body: RetrainRequest, settings: Settings = Depends(get_settings)):
        try:
            return retraining.start(settings, f"manual: {body.reason}", wait=body.wait)
        except RetrainInProgressError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/admin/rollback", tags=["admin"], dependencies=[Depends(require_admin)])
    def rollback(settings: Settings = Depends(get_settings)):
        try:
            return registry.rollback(settings.artifacts_dir)
        except ModelNotFoundError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/admin/simulate", tags=["admin"], dependencies=[Depends(require_admin)])
    def simulate(body: SimulateRequest, settings: Settings = Depends(get_settings)):
        """Generate SIMULATED readings, predict them, and (optionally) record SIMULATED outcomes."""
        df = simulate_new_observations(body.n, body.seed, body.drift)
        rows = _predict_and_log(df[["machine_id", *FEATURE_COLUMNS]].to_dict("records"), settings)
        if body.record_outcomes:
            outcomes = [(row["id"], int(actual)) for row, actual in zip(rows, df[TARGET], strict=True)]
            monitoring.record_outcomes(settings.db_path, outcomes, "simulated")
        return {
            "simulated_readings": len(rows),
            "simulated_outcomes_recorded": len(rows) if body.record_outcomes else 0,
            "drift": body.drift,
            "predicted_failures": sum(r["prediction"] for r in rows),
            "note": "Readings and outcomes are SIMULATED for demonstration.",
        }

    return app


app = create_app()
