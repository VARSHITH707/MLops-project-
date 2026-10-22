"""Train the baseline (Logistic Regression) and candidate (Random Forest) models,
pick the better one on held-out data, save it to the registry and log it to MLflow."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.generate_data import TARGET
from src.data.validate_data import FEATURE_COLUMNS, validate_dataframe
from src.features.build_features import ENGINEERED, build_preprocessor
from src.models import registry
from src.models.evaluate import evaluate_model

log = logging.getLogger(__name__)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the data content, so every model records exactly what it saw."""
    hashed = pd.util.hash_pandas_object(df.sort_index(axis=1), index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def build_models(train_cfg: dict) -> dict[str, Pipeline]:
    seed = train_cfg["seed"]
    lr = train_cfg["logistic_regression"]
    rf = train_cfg["random_forest"]
    classifiers = {
        "logistic_regression": LogisticRegression(
            C=lr["C"], max_iter=lr["max_iter"], class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf["n_estimators"],
            max_depth=rf["max_depth"],
            min_samples_leaf=rf["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    return {name: Pipeline([*build_preprocessor(), ("model", clf)]) for name, clf in classifiers.items()}


def _log_to_mlflow(settings, result: dict, version_dir: Path, run_name: str) -> str | None:
    """Best effort: a tracking failure must not break training."""
    if not settings.mlflow_enabled:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        name = "predictive-maintenance"
        if mlflow.get_experiment_by_name(name) is None:
            artifact_root = (settings.home / "mlruns" / "artifacts").resolve().as_uri()
            mlflow.create_experiment(name, artifact_location=artifact_root)
        mlflow.set_experiment(name)
        run_id = None
        for model_type, res in result["candidates"].items():
            with mlflow.start_run(run_name=f"{run_name}-{model_type}") as run:
                mlflow.set_tags(
                    {
                        "model_type": model_type,
                        "selected": str(model_type == result["model_type"]),
                        "dataset_fingerprint": result["dataset_fingerprint"],
                        "model_version": result["model_version"],
                    }
                )
                mlflow.log_params({f"{model_type}.{k}": v for k, v in res["hyperparameters"].items()})
                mlflow.log_params({"threshold": result["threshold"], "train_rows": result["train_rows"]})
                mlflow.log_metrics(
                    {k: v for k, v in res["metrics"].items() if isinstance(v, float)}
                    | {"training_seconds": res["training_seconds"]}
                )
                if model_type == result["model_type"]:
                    mlflow.log_artifacts(str(version_dir), artifact_path="model")
                    run_id = run.info.run_id
        return run_id
    except Exception as exc:  # noqa: BLE001 - tracking is optional
        log.warning("MLflow logging failed, continuing without it: %s", exc)
        return None


def train_and_register(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    settings,
    run_name: str = "train",
    check_duplicates: bool = True,
) -> dict:
    """Fit both model types, keep the best on ``test_df``, save it (NOT promoted)."""
    validate_dataframe(train_df, check_duplicates=check_duplicates)
    validate_dataframe(test_df, check_duplicates=check_duplicates)
    if train_df[TARGET].nunique() < 2:
        raise ValueError("training data must contain both failure and non-failure rows")

    cfg = settings.params["train"]
    threshold = cfg["threshold"]
    metric = cfg["selection_metric"]
    candidates: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in build_models(cfg).items():
        start = time.perf_counter()
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET])
        seconds = round(time.perf_counter() - start, 3)
        metrics = evaluate_model(pipeline, test_df, threshold)
        fitted[name] = pipeline
        candidates[name] = {
            "metrics": metrics,
            "training_seconds": seconds,
            "hyperparameters": cfg[name],
        }
        log.info("%s: f1=%.3f recall=%.3f precision=%.3f roc_auc=%.3f (%.2fs)",
                 name, metrics["f1"], metrics["recall"], metrics["precision"],
                 metrics["roc_auc"] or float("nan"), seconds)

    best = max(candidates, key=lambda n: candidates[n]["metrics"][metric])
    version = registry.new_version()
    metadata = {
        "model_version": version,
        "model_type": best,
        "trained_at": registry.utc_now(),
        "dataset_fingerprint": dataset_fingerprint(train_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_names": FEATURE_COLUMNS,
        "engineered_features": ENGINEERED,
        "target": TARGET,
        "random_seed": cfg["seed"],
        "threshold": threshold,
        "selection_metric": metric,
        "metrics": candidates[best]["metrics"],
        "training_seconds": candidates[best]["training_seconds"],
        "hyperparameters": cfg[best],
        "status": "candidate",
        "run_name": run_name,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    registry.save_version(fitted[best], metadata, settings.artifacts_dir)
    result = {**metadata, "candidates": candidates}
    run_id = _log_to_mlflow(settings, result, settings.artifacts_dir / "models" / version, run_name)
    if run_id:
        registry.update_metadata(version, settings.artifacts_dir, mlflow_run_id=run_id)
    result["mlflow_run_id"] = run_id
    return result


def write_training_report(result: dict, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_version": result["model_version"],
        "model_type": result["model_type"],
        **{k: v for k, v in result["metrics"].items() if not isinstance(v, dict)},
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    path = reports_dir / "training_report.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
