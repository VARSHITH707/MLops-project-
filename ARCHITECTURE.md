# Architecture

## 1. Goals and constraints
- Runs on a normal laptop, fully offline once dependencies are installed.
- Every step is reproducible (fixed seeds, DVC-tracked data, hashed datasets).
- The served model can only change through an explicit, auditable action (initial training, guarded promotion or rollback).
- Simple enough to explain in a viva: no Kubernetes, Kafka, Spark or paid cloud services.

## 2. Components

```
┌──────────────────────────── offline / batch ─────────────────────────────┐
│ params.yaml ─► DVC: generate ─► validate ─► prepare ─► train              │
│                   data/raw     reports/     data/processed   │            │
│                                                              ▼            │
│                     artifacts/models/<version>/{model.joblib,metadata.json}│
│                     artifacts/current.json  (pointer: current/previous)   │
│                     MLflow  mlruns/mlflow.db + mlruns/artifacts           │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │ load current version (cached, auto-reload)
┌──────────────────────────── online / serving ────────────────────────────┐
│ Streamlit :8501 ──HTTP──► FastAPI :8000                                   │
│                           ├─ /predict, /predict/batch ─► predictor.py     │
│                           ├─ /outcomes, /monitoring/*, /metrics           │
│                           └─ /admin/* (X-API-Key) ─► retraining.py        │
│                                    │                       │              │
│                                    ▼                       ▼              │
│                     SQLite logs/monitoring.db   retrain_pipeline.py       │
│                     (predictions, outcomes,     lock → validate → train   │
│                      api_errors)                → compare → promote/reject│
└──────────────────────────────────────────────────────────────────────────┘
```

| Layer | Module | Responsibility |
|---|---|---|
| Config | `app/config.py`, `params.yaml`, env vars | one place for paths, parameters and secrets; `PM_HOME` relocates every path (used by tests and Docker) |
| Data | `src/data/generate_data.py` | synthetic dataset + drifted "new" observations |
| Validation | `src/data/validate_data.py`, `app/schemas.py` | dataset rules / per-request rules (Pydantic) |
| Features | `src/features/build_features.py` | engineered features, `ColumnTransformer`, stratified split |
| Training | `src/models/train.py`, `evaluate.py` | fit LR + RF, select by F1, metrics, MLflow logging |
| Registry | `src/models/registry.py` | versioned folders, atomic pointer, promote, rollback |
| Serving | `app/main.py`, `app/predictor.py` | HTTP API, cached model, risk categories |
| Monitoring | `app/monitoring.py` | SQLite logging, PSI drift, live performance, error log |
| Retraining | `src/pipelines/retrain_pipeline.py`, `app/retraining.py` | lock, triggers, candidate evaluation, decision, reports, scheduler |
| UI | `app/dashboard.py` | API client only, no model of its own |

## 3. Key flows

### 3.1 Prediction
`POST /predict` → Pydantic validation (422 on error) → `predictor.get_model()` (reads `current.json`; reloads only if the version changed) → `Pipeline.predict_proba` → threshold and risk band → row inserted into SQLite → response containing the `prediction_id`.

### 3.2 Initial training (`scripts/setup.py` or `dvc repro`)
generate → validate (exit 1 on any error) → stratified split → fit both models on train → evaluate on test → save best as version (status `candidate`) → log to MLflow → promote (initial run, or `--promote`) → `reports/metrics.json`.

### 3.3 Retraining
```
trigger (manual / ≥N new labels / F1 degradation)
  └─► acquire lock (O_EXCL file; 409 if held; stale after 30 min)
       └─► labelled rows from SQLite → validate
            └─► train = base train + 70% new;  eval = base test + 30% new
                 └─► candidate = train_and_register(train, eval)
                      └─► current metrics = evaluate(current model, eval)   ← same data
                           └─► decide(): F1 ≥ min, F1 ≥ current + gain, recall drop ≤ max
                                ├─ pass → registry.promote (atomic) → status promoted
                                └─ fail → metadata status rejected → current keeps serving
  finally: write report + history line, release lock
```

### 3.4 Rollback
`registry.rollback()` promotes `pointer.previous`. Versions are never deleted, so it is always possible as long as a previous version exists (409 otherwise).

## 4. Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| Synthetic data from a documented risk formula | offline and reproducible; the label logic can be explained | metrics don't transfer to real machines |
| Feature engineering inside the sklearn `Pipeline` | identical transformations in training and serving; scaler fitted on train only | the feature function must be importable wherever the model is loaded |
| Select by F1, threshold 0.35 | rare positives; a missed failure costs more than a false alarm | more false alarms |
| File-based registry + atomic pointer | transparent, trivially reversible, no server needed | single writer; not a multi-team registry |
| MLflow on SQLite, best-effort logging | MLflow 3 favours DB backends; training must not depend on tracking | tracking is local unless the URI is changed |
| DVC stages with `persist: true` model outputs | `dvc repro` reproduces the pipeline without deleting retrained versions | `dvc status` reports the model folder as changed after retraining (expected) |
| Custom PSI instead of Evidently | ~40 transparent lines, no heavy dependency | univariate only |
| Candidate vs current on the **same** eval set | the only fair comparison | candidate type is selected on that same set (slight optimism) |
| File lock for retraining and the scheduler | works across processes and API workers; stdlib only | stale-lock timeout instead of real liveness detection |
| Shared API key for admin routes | explicit and configurable, no hardcoded secret | not per-user authentication |
| Dashboard calls the API only | one source of truth | dashboard is useless without the API (clear error shown) |

## 5. Failure behaviour

| Situation | Behaviour |
|---|---|
| No model trained | `/health` → `degraded`; `/predict`, `/model-info` → 503 with "run scripts/setup.py" |
| Invalid request | 422 listing each bad field; recorded in `api_errors` |
| MLflow unavailable | training logs a warning and continues |
| Retraining already running | 409 |
| New labelled data invalid | run ends `failed_validation`; model unchanged; lock released |
| Exception during retraining | run ends `failed`; model unchanged; lock released; report written |
| Candidate worse | `rejected`; candidate kept on disk for inspection |
| Process crash during retraining | lock reclaimed after `lock_timeout_seconds`; the pointer is never half-written |
| Several API workers with the scheduler on | only the worker holding `.scheduler.lock` runs it; the heartbeat keeps the lock fresh |

## 6. Security notes
- Admin routes are disabled unless `PM_ADMIN_API_KEY` is set. Keys are compared with `secrets.compare_digest`.
- No secrets in code, images or Git (`.env` is git- and docker-ignored; `.env.example` holds placeholders).
- `joblib` files are pickles: the app only loads models it wrote itself and never accepts uploaded model files.
- Containers run as a non-root user.
- Request size is bounded (batch ≤ 1000 readings; strings length-limited).

## 7. What would change for production
Time-based splits and real labels; a remote MLflow server and registry; a DVC remote; proper authentication; Postgres instead of SQLite; a real scheduler/orchestrator; multivariate drift and alerting; canary or shadow deployment before promotion.
