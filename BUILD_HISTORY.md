# BUILD_HISTORY — Predictive Maintenance MLOps Platform

This file is the running record of how the project was built. New timeline entries are appended; older entries are never rewritten.

---

## A. Project Overview

| Item | Value |
|---|---|
| Project name | Predictive Maintenance MLOps Platform |
| Problem | Industrial machines fail unexpectedly. Failures stop production, raise repair costs, and create safety risks. |
| Objective | Estimate the probability that a machine fails within the next 24 operating hours from its sensor readings, and wrap that model in a full MLOps lifecycle (data, validation, training, tracking, versioning, serving, monitoring, controlled retraining, rollback). |
| Data | Reproducible synthetic sensor data (fixed random seed). Educational simulation, not a certified safety system. |
| Stack | Python 3.11, pandas, NumPy, scikit-learn, joblib, Pydantic, MLflow, DVC, FastAPI, Uvicorn, Streamlit, SQLite, Pytest, Docker / Docker Compose, GitHub Actions, custom PSI drift detection |
| Deliverables | Working code, tests, CI workflow, Docker files, README, ARCHITECTURE, PROJECT_GUIDE, PROFESSOR_DEMO, VIVA_QUESTIONS, TROUBLESHOOTING, CLOUD_DEPLOYMENT, this history |

## B. Initial Environment (recorded 2026-10-23, local time UTC+5:30)

| Check | Result |
|---|---|
| OS | Windows 11 Home (build 10.0.26300) |
| Default `python` on PATH | Python 3.11.15, but it is another tool's virtualenv without `pip` |
| Default `pip` on PATH | belongs to Python 3.13 (not the required version) |
| Python 3.11 used | `uv`-managed CPython 3.11.15 → project venv `.venv` |
| Git | 2.45.2; workspace `Z:\MLOPS` was empty and not a Git repo |
| GitHub CLI | logged in; target repo `VARSHITH707/MLops-project-` exists, public, empty |
| Docker / Docker Compose | **Not installed** (`docker: command not found`) |
| Node | v24 (not needed) |
| Initial blockers | Docker unavailable → container deployment can be written but not executed on this machine |

## C. Architecture Decisions

| # | Decision | Reason | Alternatives | Consequences / limits |
|---|---|---|---|---|
| 1 | Project lives in `predictive-maintenance-mlops/`, which is also the Git repo root | Prompt asks for a dedicated folder; the GitHub repo holds only this project | Repo root at `Z:\MLOPS` | The pushed repository root is the project itself |
| 2 | Dedicated `.venv` built from uv's CPython 3.11.15 | Machine's default `pip` is Python 3.13 | Use 3.13 | Commands in docs use `.venv\Scripts\python.exe` |
| 3 | Synthetic data with a hand-written failure-risk formula | No dataset purchase or download; fully reproducible | AI4I public dataset | Metrics only show that the pipeline works, not real-world accuracy |
| 4 | Local file-based model registry (`artifacts/models/<version>/` + `artifacts/current.json` pointer, written atomically) | Simple, transparent, easy to roll back | MLflow Model Registry | One pointer file; fine for a single host or shared volume |
| 5 | MLflow with a SQLite backend (`mlruns/mlflow.db`) | MLflow 3 recommends a database backend over the file store; works offline | File store, remote server | Tracking is local unless `MLFLOW_TRACKING_URI` is changed. Training still succeeds if MLflow fails (logged warning) |
| 6 | Custom PSI drift detection instead of Evidently | Transparent (~40 lines), no heavy dependency, easy to explain in a viva | Evidently | Per-feature only; no multivariate drift |
| 7 | Retraining compares candidate vs current model on the **same** evaluation set and promotes only if acceptance rules pass | Prompt forbids promoting just because training finished | Always promote / human approval | Rules are configurable in `params.yaml` |
| 8 | Admin endpoints protected by `X-API-Key` header checked against `PM_ADMIN_API_KEY` env var; disabled if the variable is unset | No hardcoded secret | OAuth | Simple shared-key auth, suitable for a demo |
| 9 | Background scheduler disabled by default and guarded by a lock file so only one API worker runs it | Avoid duplicate schedulers with multiple workers | APScheduler, cron | Lock goes stale after 3 missed intervals and is then reclaimed |

## D. Build Timeline

### 2026-10-23 ~01:30 IST — Phase 1: Inspect the workspace
- Ran: `gh auth status`, `git --version`, `python --version`, `pip list`, `where python`, `py -0p`, `docker --version`.
- Found the Python/pip mismatch and the missing Docker described in section B.
- Next: create the venv and the project structure.

### 2026-10-23 ~01:35 IST — Phase 2: Environment + structure
- Ran: `<uv cpython 3.11.15>\python.exe -m venv .venv` → OK (pip 24.0, Python 3.11).
- Ran: `uv pip install --python .venv/Scripts/python.exe numpy pandas scikit-learn joblib pydantic pyyaml mlflow dvc fastapi "uvicorn[standard]" streamlit requests pytest httpx ruff` → OK. Resolved versions pinned in `requirements*.txt`.
- Ran: `git init -b main`, created the folder layout and this file.
- Next: data generation and validation.

### 2026-10-23 01:40–01:46 IST — Phases 3–5: Data, validation, features, training
- Created: `params.yaml`, `app/config.py`, `src/data/generate_data.py`, `src/data/validate_data.py`, `src/features/build_features.py`, `src/models/evaluate.py`, `src/models/registry.py`, `src/models/train.py`, `scripts/train_model.py`, `scripts/setup.py`.
- Ran: `.venv\Scripts\python.exe scripts/setup.py` → exit 0. 8000 rows, failure rate 12.96%. **Problem:** the model was weak (RF F1 0.445, ROC-AUC 0.787) because the simulated labels were too noisy to make a useful demo.
- Fix: tuned the generator so the hidden risk score is sharper (coefficients ×2.5, intercept −8). A throwaway tuning script compared sharpness 2.0 / 2.5 / 3.0 → picked 2.5 (failure rate ≈15.6%, RF AUC ≈0.94, still not trivially separable).
- Re-ran `scripts/setup.py` → exit 0. Failure rate 15.65%. LR: F1 0.493, recall 0.916, AUC 0.901. RF: F1 0.690, recall 0.852, precision 0.580, AUC 0.935 → RF selected and promoted. Artifacts `artifacts/models/<version>/model.joblib`, `metadata.json`, `artifacts/current.json`, `reports/metrics.json`, `reports/training_report.json` all present.
- Observation: the first MLflow use takes ~45 s on this machine (import + DB table creation). Tests disable MLflow with `PM_DISABLE_MLFLOW=1`.

### 2026-10-23 01:47–01:52 IST — Phases 7–9: API, monitoring, retraining
- Created: `app/schemas.py`, `app/predictor.py`, `app/monitoring.py`, `app/retraining.py`, `app/main.py`, `src/pipelines/retrain_pipeline.py`.
- In-process smoke test (FastAPI TestClient): `/health` ok, `/predict` → probability 0.8872 / high, invalid body → 422, `/admin/simulate` (300 rows, drift 1.5) ok, drift → significant, performance ok, `/admin/retrain` without key → 401.
- **Failure 1:** retraining returned `failed_validation`: "timestamp has unparseable values; 160 duplicate (machine_id, timestamp)". Causes: (a) pandas could not parse a column mixing `2026-01-01T00:00:00` and `...+00:00` formats; (b) logged batch predictions share one timestamp, so the duplicate rule (meant for raw sensor data) fired. Fixes: parse with `format="ISO8601", utc=True`; added `check_duplicates` flag, set to False for logged predictions, which are keyed by a unique prediction id.
- Re-ran → `promoted` (candidate F1 0.706 vs current 0.705 on the same 1,690-row eval set). `/admin/rollback` → previous version served again. Verified.

### 2026-10-23 01:52–01:57 IST — Phases 10–12: Dashboard, Docker, CI, tests, DVC
- Created: `app/dashboard.py`, `requirements*.txt`, `dvc.yaml`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `.gitignore`, `.dockerignore`, `.env.example`, `pyproject.toml`, `tests/*`.
- Dashboard fix: replaced `st.error(x) if err else st.success(y)` one-liners with if/else (Streamlit "magic" would render the return value). Replaced the deprecated `use_container_width` with `width="stretch"`.
- `pytest -q` → **50 passed** (13.6 s). `ruff check .` → 7 issues (unused import, zip without strict, long line) → fixed → "All checks passed!".
- Coverage run → 79% overall; dashboard 0%. Added `tests/test_dashboard.py` (Streamlit AppTest with its HTTP calls routed to the real FastAPI app) → 2 passed.
- `dvc init`, `dvc repro` from a clean state → all 4 stages ran, identical metrics (F1 0.69044, AUC 0.93491), proving reproducibility. Second `dvc repro` → "Data and pipelines are up to date". `dvc metrics show` works.
- MLflow check: 2 runs (LR + RF) with metrics, params and the dataset fingerprint; the selected run holds `model/model.joblib` + `model/metadata.json`.

### 2026-10-23 01:57–02:02 IST — Phase 13: Live smoke tests (real servers)
- Started Uvicorn (:8000), Streamlit (:8501), MLflow UI (:5000).
- `curl /health` → ok; healthy sample → 0.0062 LOW; worn-tool sample → 0.8872 HIGH; `/docs` → 200; invalid body → 422; admin without key → 401.
- Browser: dashboard Overview showed live metrics from the API; Monitoring tab showed significant drift after simulating 300 drifted readings; the Retraining tab "Retrain now" → PROMOTED; MLflow UI listed the experiment and the retraining runs.
- UI fixes: the "Model" metric was truncated → moved to the text line; the admin actions now render above the status block so the status reflects the action just taken; the simulation message is human-readable.
- `scripts/check_system.py --api http://localhost:8000` → READY (Docker WARN). `scripts/run_demo.py` → candidate **rejected** (F1 0.727 < current 0.732), served model unchanged. This shows the promotion guard working on real output.
- Next: documentation, final review, commit, push.

### 2026-10-23 02:03–02:10 IST — Phases 14–16: Documentation, review, final verification
- Created: `README.md`, `PROJECT_GUIDE.md`, `ARCHITECTURE.md`, `PROFESSOR_DEMO.md`, `VIVA_QUESTIONS.md` (40 Q&A), `TROUBLESHOOTING.md`, `CLOUD_DEPLOYMENT.md`, `Makefile`, `scripts/check_system.py`, `scripts/run_demo.py`.
- Review: confirmed `python -m mlflow` and `python -m dvc` entry points used in the docs work; `dvc dag` prints without a pager; the docs' metric values match `reports/metrics.json`; the test count (52) matches `pytest` output.
- Change: added `reports/data_validation.json` as a dependency of the `prepare` stage, so the DAG reads generate → validate → prepare → train. `dvc repro` reran prepare + train with identical metrics.
- Final `ruff check .` → pass; `pytest -q` → 52 passed.
- Next: commit and push to GitHub, then check the Actions run.

### 2026-10-23 02:11–02:15 IST — Push and CI
- `git commit` (dba891a) and `git push -u origin main` → OK.
- GitHub Actions run 39850819908 → **success** (test 1m2s, docker 1m19s). Evidence is in the test log (section F).
- Annotation: actions/checkout@v4 and setup-python@v5 target the deprecated Node 20 → bumped to checkout@v5 and setup-python@v6. Updated the README's Docker verification note.

## E. Implementation Status

| Component | Status |
|---|---|
| Data generation | VERIFIED |
| Data validation | VERIFIED |
| Feature engineering | VERIFIED |
| Model training | VERIFIED |
| Model evaluation | VERIFIED |
| MLflow | VERIFIED |
| DVC | VERIFIED (local cache only; no remote configured) |
| FastAPI | VERIFIED |
| Streamlit | VERIFIED |
| Docker | VERIFIED for the image (CI: build → train in container → /health ok). `docker compose up` not run (Docker not installed locally) |
| Monitoring | VERIFIED |
| Drift detection | VERIFIED |
| Retraining | VERIFIED |
| Model promotion | VERIFIED |
| Rollback | VERIFIED |
| Tests | VERIFIED (52 passed) |
| CI | VERIFIED (GitHub Actions run 39850819908: test ✓ 1m2s, docker ✓ 1m19s) |
| Cloud guide | IMPLEMENTED (documentation only, nothing deployed) |
| Professor guide | IMPLEMENTED (steps exercised live during smoke tests) |

## F. Test and Verification Log

| Date (IST) | Command | Result | Relevant output |
|---|---|---|---|
| 10-23 01:43 | `python scripts/setup.py` (first generator) | exit 0, weak model | RF F1 0.445, AUC 0.787 |
| 10-23 01:46 | `python scripts/setup.py` (tuned generator) | exit 0 | failure rate 15.65%; RF F1 0.690, recall 0.852, AUC 0.935 |
| 10-23 01:49 | in-process API smoke test (TestClient) | retrain FAILED | `failed_validation`: timestamp parse + duplicates |
| 10-23 01:50 | same, after fix | PASS | promoted (cand F1 0.706 vs cur 0.705), rollback ok |
| 10-23 01:53 | `pytest -q` | PASS | 50 passed in 13.6 s |
| 10-23 01:54 | `ruff check .` | FAIL → fixed → PASS | 7 issues → "All checks passed!" |
| 10-23 01:54 | `pytest --cov` | PASS | 79% total (dashboard 0% → tests added) |
| 10-23 01:55 | `pytest tests/test_dashboard.py` | PASS | 2 passed |
| 10-23 01:56 | `dvc init && dvc repro` | PASS | 4 stages, identical metrics; 2nd repro "up to date" |
| 10-23 01:57 | MLflow `search_runs` | PASS | 2 runs, selected run has model/model.joblib |
| 10-23 01:58 | `curl` against Uvicorn :8000 | PASS | health ok; 0.8872 HIGH; 0.0062 LOW; /docs 200; 422; 401 |
| 10-23 02:00 | Browser: dashboard + MLflow UI | PASS | metrics, drift "significant", Retrain → PROMOTED |
| 10-23 02:02 | `scripts/check_system.py --api ...` | PASS | READY (docker WARN) |
| 10-23 02:02 | `scripts/run_demo.py` | PASS | candidate REJECTED (0.727 < 0.732), model unchanged |
| 10-23 02:08 | `ruff check . && pytest -q` | PASS | All checks passed; 52 passed in 13.9 s |
| 10-23 02:08 | `dvc repro` after adding validation report as prepare dep | PASS | linear DAG generate→validate→prepare→train; same metrics |
| 10-23 02:09 | `setup.py`, `setup.py` again, `setup.py --force` (temp PM_HOME) | PASS | trains; skips when model exists; rebuilds + promotes |
| 10-23 02:09 | `yaml.safe_load` of docker-compose.yml and ci.yml | PASS | services init/api/dashboard/mlflow; jobs test/docker |
| 10-23 02:12 | `git push -u origin main` → GitHub Actions run 39850819908 | PASS | `test` job: ruff "All checks passed!", "52 passed in 9.83s", mini pipeline outputs present; `docker` job: image built, model trained in container (RF F1 0.690, AUC 0.935, identical to Windows), `/health` → `model_loaded: true` |

## G. Problems and Resolutions

| Problem | Cause | Resolution |
|---|---|---|
| `python -m pip` → `No module named pip` | `python` on PATH is another tool's venv | Built a dedicated `.venv` from uv's CPython 3.11.15 |
| Model too weak to demo (F1 0.445) | Synthetic labels too noisy | Sharpened the risk score (×2.5) → F1 0.690, AUC 0.935 |
| Retraining `failed_validation` (timestamps, duplicates) | Mixed ISO formats; batch predictions share a timestamp | `format="ISO8601", utc=True`; `check_duplicates=False` for logged predictions (unique ids) |
| Streamlit could render return values of `x if c else y` lines | Streamlit "magic" displays bare expressions | Rewrote as explicit if/else |
| Dashboard "Model" metric truncated, raw dict message, stale status | Layout | Model type moved to text; readable message; admin actions above status |
| `preview_start` needed launch config at workspace root | Tooling | `Z:\MLOPS\.claude\launch.json` (outside the repo) |
| Docker not installed | Machine setup | Not resolved locally; CI `docker` job verifies the image on GitHub |

## H. Final Verification

**Verified by actually running:** data generation (reproducible), validation (every rule by tests), feature engineering, training and selection, evaluation metrics, artifact and metadata saving, MLflow logging + UI, DVC pipeline + lock + metrics, FastAPI (all endpoints via tests; core endpoints via curl against real Uvicorn), Streamlit (AppTest + manual browser run), SQLite logging, PSI drift, live performance, retraining lock, promotion, rejection, failed-validation path, crash path, rollback, single-instance scheduler, `check_system.py`, `run_demo.py`.

**Verified remotely (GitHub Actions run 39850819908):** lint, all 52 tests on Linux, the mini pipeline, Docker image build, training inside the container and the API health check.

**Not verified:** `docker compose up` with all four services together (init → api → dashboard → mlflow, volumes, health-check ordering), because Docker is not installed on this machine. The single-container path is verified by CI; the Compose file is YAML-validated only.

**Not verified at all:** cloud deployment (documentation only, by design: no cloud resources were created). Scheduler timing over real minutes (only its single-instance guard and trigger logic are tested; the loop body is exercised through `check_triggers` + `start`).

## I. Handover Notes

- **State:** all phases complete; code, tests and docs are committed on `main` and pushed to `VARSHITH707/MLops-project-`.
- **Resume:** `cd Z:\MLOPS\predictive-maintenance-mlops; .\.venv\Scripts\Activate.ps1; python scripts/check_system.py; python -m pytest -q`
- **Last completed phase:** 16 (final verification report).
- **Next action:** rehearse PROFESSOR_DEMO.md; optionally install Docker Desktop and run `docker compose up --build` to verify the full Compose stack.
- **Known blockers:** Docker unavailable locally. No DVC remote / remote MLflow configured (local only, as documented).
