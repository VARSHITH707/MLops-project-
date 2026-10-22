# Professor Demonstration (about 10 minutes)

Everything runs locally. No internet or cloud account is needed during the demo.

## Before the demo (do this the evening before, ~5 minutes)

```powershell
cd <project folder>
.\.venv\Scripts\Activate.ps1
python -m pytest -q                         # expect: 52 passed
# Optional clean slate: wipes old prediction logs and retraining history
Remove-Item logs\monitoring.db, reports\retraining_history.jsonl -ErrorAction SilentlyContinue
Remove-Item reports\retraining -Recurse -ErrorAction SilentlyContinue
python scripts/setup.py --force             # fresh data + model, promoted
python scripts/check_system.py              # expect: READY
```

Open **three PowerShell windows** (activate the venv in each) and start:

| Window | Command |
|---|---|
| A: API | `$env:PM_ADMIN_API_KEY = "demo-key"; python -m uvicorn app.main:app --port 8000` |
| B: Dashboard | `python -m streamlit run app/dashboard.py` |
| C: MLflow | `python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000` |

Open browser tabs: http://localhost:8501 (dashboard), http://localhost:8000/docs (API), http://localhost:5000 (MLflow), plus the GitHub repo's **Actions** tab. Keep a fourth PowerShell window open for commands.

---

## Script

### 1. Problem and objective (1 min)
**Show:** the README top section.
**Say:** "Factories lose money and risk safety when machines break unexpectedly. Sensors report temperature, speed, torque and tool wear. My system estimates the probability that a machine fails in the next 24 operating hours, so maintenance can be scheduled first. The model is only one piece. The project is the full MLOps lifecycle around it: validated data, tracked experiments, versioned data and models, an API, a dashboard, monitoring, drift detection, and retraining that only replaces the model when the new one is proven not worse. The data is synthetic and reproducible, so it demonstrates engineering, not real-world accuracy."

### 2. Architecture and technology choices (1.5 min)
**Show:** the diagram in README §4 (or ARCHITECTURE.md).
**Say:** "Offline, a DVC pipeline generates, validates, splits and trains. MLflow records every run, and each model is saved as a version with metadata. Online, FastAPI serves the current version, every prediction goes to SQLite, and monitoring computes drift and live accuracy. Retraining trains a candidate, compares it against the current model on the same data, and promotes or rejects it. Streamlit is just a client of the API. I chose scikit-learn because it's reliable and explainable, FastAPI for automatic validation and docs, and SQLite because it needs no server."

### 3. Reproducible pipeline with DVC (1 min)
**Run:**
```powershell
python -m dvc dag
python -m dvc repro
python -m dvc metrics show
```
**Expected:** the 4-stage graph; *"Data and pipelines are up to date"* (nothing changed, nothing reruns); the metrics table (F1 ≈ 0.690, recall 0.852, ROC-AUC ≈ 0.935).
**Say:** "dvc.lock stores the hash of every input and output. Change a parameter and only the affected stages rerun. Same seed, same data, same model."

### 4. Experiment tracking in MLflow (1 min)
**Click:** MLflow tab → **Experiments** → **predictive-maintenance** → select both `initial-train-*` runs → **Compare**.
**Expected:** Logistic Regression F1 ≈ 0.49 (recall 0.92) vs Random Forest F1 ≈ 0.69 (recall 0.85); the RF run has the `model/` artifacts and the `selected=True` tag.
**Say:** "Logistic regression is my baseline. Random forest wins on F1, so it's selected. Each run records hyperparameters, metrics and the dataset fingerprint, so I can always trace a model back to its data."

### 5. Make predictions (1.5 min)
**Click:** Dashboard → **Overview** (metrics, confusion matrix, versions) → **Predict** → sample *Worn tool, high torque* → **Predict failure risk**.
**Expected:** HIGH risk, probability ≈ 89% (exact value depends on the model version). Then choose *Healthy machine* → LOW, ≈ 1%.
**Say:** "Recall matters most: a missed failure is a breakdown, while a false alarm is just an inspection. That's why the threshold is 0.35 instead of 0.5, and why I don't report accuracy alone. 'Never fails' would already be 84% accurate."

**Then show the API itself:** http://localhost:8000/docs → `POST /predict` → **Try it out** → set `"torque": -5` → **Execute** → **422** error naming the field.
**Say:** "Invalid inputs are rejected, never guessed. The dashboard has no model of its own. It calls this API."

### 6. Monitoring and drift (1.5 min)
**Click:** Dashboard sidebar → enter `demo-key` → **Retraining** tab → set *Sensor drift* to **1.5** → **Simulate**.
**Expected:** "Logged 300 SIMULATED readings with SIMULATED outcomes...".
**Click:** **Monitoring** tab.
**Expected:** 300+ predictions logged; drift **significant** (air temperature, tool wear, ... with PSI > 0.25); live performance metrics; recent predictions table; sensor histogram.
**Say:** "Each prediction is logged in SQLite. PSI compares recent inputs with the training data feature by feature. Drift says the inputs changed, not that the model is wrong, so it only raises an advisory. Live performance needs real outcomes, which here are clearly marked as simulated."

### 7. Controlled retraining: promote or reject (1.5 min)
**Click:** **Retraining** tab → **Retrain now** (takes 10–60 s; the first MLflow use is the slow part).
**Expected:** either **PROMOTED: all acceptance checks passed** or **REJECTED: candidate F1 … not >= current …**. The JSON below shows `current_metrics` and `candidate_metrics` on the same evaluation rows. *Both outcomes demonstrate the system working.*
**Say:** "A lock prevents two retrainings at once. The candidate is trained on old plus new data, and then both models are scored on exactly the same evaluation set. It's promoted only if F1 is at least the current F1 and recall doesn't drop more than 2 points. Otherwise it's rejected and the current model keeps serving. Training finishing is not a reason to deploy."

**Show rollback:** click **Roll back to previous model** → "Now serving v…". The Overview tab shows the version change.
**Say:** "Old versions are never deleted, and the pointer file is swapped atomically, so rollback is instant and safe."

Optional terminal alternative for steps 6–7: `$env:PM_ADMIN_API_KEY="demo-key"; python scripts/run_demo.py`

### 8. Quality: tests, CI, Docker (1 min)
**Run:** `python -m pytest -q` → `52 passed`.
**Show:** GitHub → **Actions** → latest run (lint, tests, mini pipeline, Docker build + health check).
**Show:** `docker-compose.yml`.
**Say:** "Fifty-two tests cover validation, training, the API, drift maths, locking, promotion, rejection and rollback, all in temporary folders. CI reruns them on every push and also builds the Docker image and checks the container becomes healthy. Compose starts an init job that trains the first model, then the API, dashboard and MLflow, with health checks, volumes and a non-root user."

### 9. Limitations and conclusion (0.5 min)
**Say:** "Limitations: synthetic data, a random rather than time-based split, univariate drift, a local registry and a simple API key. With real data, I'd add time-based validation, a remote MLflow and DVC remote, real authentication and canary deployment. The main lesson is that the lifecycle (validation, traceability, monitoring and guarded updates) is what makes a model trustworthy in production."

---

## Backup paths

| Problem during the demo | Fallback |
|---|---|
| Docker not installed / not working | Everything above already runs without Docker. Just show `docker-compose.yml` and the CI Docker job. |
| MLflow UI won't open | `python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); print(mlflow.search_runs(experiment_names=['predictive-maintenance'])[['tags.mlflow.runName','metrics.f1','metrics.recall']])"` |
| Dashboard broken | use Swagger UI at http://localhost:8000/docs, or `python scripts/run_demo.py` |
| API port 8000 busy | start with `--port 8001`, and before starting the dashboard set `$env:API_URL="http://localhost:8001"` |
| Retraining says 409 "already in progress" | a crashed run left the lock: `Remove-Item artifacts\.retrain.lock` |
| No internet | nothing in the demo needs it, except showing GitHub Actions |
