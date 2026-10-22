# Project Guide: understanding every part, from zero

This guide assumes you know basic Python and nothing about machine learning operations. Each term is explained before it is used. Read it top to bottom once, then use the component sections as a reference.

---

## Part 1: The big picture

### The real-world problem
A factory machine slowly wears out. Its cutting tool gets blunt, it runs hotter, and it needs more force. Eventually it breaks. A surprise breakdown stops the production line, needs urgent (expensive) repairs and can injure people. If we could say *"machine M-007 will probably fail within the next day"*, the maintenance team could fix it during a planned stop instead.

### The objective
1. Build a model that reads one set of sensor values and outputs a **probability of failure in the next 24 operating hours**.
2. Build the **system around the model** so it can be trusted over time: checked data, recorded experiments, versioned files, an API other programs can call, monitoring, safe updates and an undo button.

Part 2 is what this guide focuses on. In industry the model is often the smallest piece. Keeping it correct, traceable and safely updatable is the real work, and that discipline is called **MLOps**.

### Key words you need first

| Word | Simple meaning |
|---|---|
| **Feature** | an input column the model uses, e.g. `torque` |
| **Target / label** | the answer we want to predict: `failure_within_horizon` (1 = fails within 24 h, 0 = doesn't) |
| **Classification** | predicting a category (here: fail / not fail) |
| **Training** | showing the model many examples with known answers so it learns patterns |
| **Inference / prediction** | using the trained model on new data where the answer is unknown |
| **Probability** | a number from 0 to 1. 0.8 means "80% likely" |
| **Threshold** | the cut-off that turns a probability into a yes/no decision (here 0.35) |
| **Pipeline** | a fixed sequence of steps applied in the same order every time |
| **Artifact** | a file produced by the pipeline (dataset, model, report) |
| **Version** | a saved, named snapshot you can go back to |

### How data moves through the system

```
generate → validate → split → [train LR, train RF] → pick best → save version → promote
                                                                       │
          sensor reading ──► API /predict ──► model ──► probability ──► SQLite log
                                                                       │
                     outcomes recorded later ──► performance  ◄── drift check
                                                     │
                                          retrain candidate → compare → promote or reject
```

---

## Part 2: The components

Each section answers six questions: **what is it, why is it needed, which file, how to run it, expected output, and a common failure with its fix.** Commands assume you are in the project folder with the virtual environment active (see README §7).

### 2.1 Synthetic data generation
- **What:** code that invents realistic sensor readings: 8,000 rows across 50 machines, each with `machine_type` (L/M/H quality), air and process temperature (kelvin), rotational speed (rpm), torque (Nm), tool wear (minutes), operating hours and number of past maintenance events.
- **How labels are made:** a hidden *risk score* goes up when the tool is worn past 170 minutes, when mechanical power is above 8.5 kW or below 3.8 kW, when the gap between process and air temperature is under 9 K (poor cooling), with age, with maintenance history, and for low-quality machines. A *sigmoid* function turns the score into a probability, and a random draw with that probability decides the label. About 15.6% of rows end up as failures.
- **Why:** we need data without buying a dataset, and it must be **reproducible**: the same `seed` always gives exactly the same rows, so anyone can rerun the project and get the same numbers.
- **File:** `src/data/generate_data.py` (`generate_dataset()`, plus `simulate_new_observations()` for "new" data with optional drift).
- **Run:** `python -m src.data.generate_data`
- **Expected output:** `Wrote 8000 rows to ...data\raw\sensors.csv (seed=42)` and `Failure rate: 15.65%`.
- **Common failure:** `ModuleNotFoundError: No module named 'src'`. You ran it from the wrong folder. `cd` into the project root and use `python -m ...`.

**What "prediction horizon" means:** the label answers *"does this machine fail within the next 24 operating hours after this reading?"* The 24 h is the horizon. A longer horizon gives earlier warnings but fuzzier predictions.

**Leakage prevention:** `machine_id` and `timestamp` are **not** used as features. If they were, the model could memorise "machine 7 fails a lot" instead of learning sensor patterns, and it would be useless for a new machine. The label itself is obviously never an input either.

### 2.2 Data validation
- **What:** a gatekeeper that inspects a dataset and lists every problem: missing columns, non-numeric values, missing values, infinite values, out-of-range numbers (e.g. torque must be 0–150 Nm), unknown machine types, unparseable timestamps, duplicate readings, and targets that aren't 0 or 1.
- **Why:** "garbage in, garbage out". A model trained on broken data quietly learns nonsense. The rule here is *reject, never silently fix*.
- **File:** `src/data/validate_data.py` (for datasets) and `app/schemas.py` (for API requests, using **Pydantic**, a library that checks the type and range of every field in a request).
- **Run:** `python -m src.data.validate_data`
- **Expected output:** `Dataset valid: 8000 rows`, and `reports/data_validation.json` containing `"valid": true`.
- **Common failure:** exit code 1 with lines like `Validation error: column 'torque' has 3 values outside [0.0, 150.0]`. Fix the data source. Don't loosen the rule unless the rule itself is wrong.

### 2.3 Feature engineering and preprocessing
- **What:** three new columns computed from the raw ones:
  - `temp_difference` = process − air temperature (cooling quality)
  - `power_kw` = torque × speed × 2π / 60 / 1000 (mechanical power)
  - `wear_x_torque` = tool wear × torque / 1000 (a worn tool under heavy load)

  Then **scaling** (numbers rescaled to mean 0 and spread 1, so no feature dominates just because its units are big) and **one-hot encoding** (`machine_type` L/M/H becomes three 0/1 columns, because models need numbers, not letters).
- **Why:** good features make patterns easier to learn. Putting them **inside a scikit-learn `Pipeline`** means training and prediction run the *exact same* code, which prevents the classic "it worked in my notebook but not in production" bug.
- **Important rule:** the scaler learns its averages **only from the training data**. If it also saw the test data, the test would no longer be a fair exam. This is called data leakage.
- **File:** `src/features/build_features.py`
- **Run:** `python -m src.features.build_features` (validates the data, then splits it 80/20 into `data/processed/train.csv` and `test.csv`)
- **Expected output:** `Train rows: 6400, test rows: 1600`
- **Limitation:** the split is random but *stratified* (both halves keep the same failure rate). Real sensor data arrives over time, so a stricter test would train on the past and test on the future.

### 2.4 Training and model selection
- **What:** two models are trained.
  - **Logistic Regression**: draws a straight-line boundary through the feature space. Simple, fast and easy to explain. This is the *baseline*.
  - **Random Forest**: hundreds of decision trees vote. It handles curved, "if-this-and-that" relationships. This is the *candidate*.

  Both get `class_weight="balanced"`, which tells them failures are rare and important. The one with the higher **F1** on the test set is kept.
- **Why two:** a baseline proves the complex model is actually worth its complexity.
- **File:** `src/models/train.py`, run by `scripts/train_model.py`
- **Run:** `python scripts/train_model.py --promote`
- **Expected output:** `random_forest: f1=0.690 recall=0.852 precision=0.580 roc_auc=0.935` and `Promoted v2026... as the served model`.
- **Common failure:** exit code 2, `Missing ...train.csv`. Run `python scripts/setup.py` first.

### 2.5 Evaluation: why accuracy is not enough
Imagine 100 readings, 16 of which are real failures. A model that always says "no failure" is 84% **accurate** and completely useless. So we use:

| Metric | Question it answers | Our model |
|---|---|---|
| **Precision** | Of the machines we flagged, how many really failed? | 0.580 |
| **Recall** | Of the machines that really failed, how many did we catch? | 0.852 |
| **F1** | One number balancing precision and recall | 0.690 |
| **ROC-AUC** | How well does the probability rank failures above non-failures, at any threshold? (1 = perfect, 0.5 = coin flip) | 0.935 |

The **confusion matrix** counts the four outcomes on the 1,600 test rows:

|  | predicted: OK | predicted: FAIL |
|---|---|---|
| **actually OK** | 1,196 (true negative) | 154 (false positive: unnecessary inspection) |
| **actually FAIL** | 37 (false negative: **missed breakdown**) | 213 (true positive) |

In maintenance a **false negative** costs far more than a false positive. That's why the decision threshold is 0.35 instead of 0.5: we accept more false alarms to miss fewer breakdowns. The threshold is configurable in `params.yaml`.

**File:** `src/models/evaluate.py`. Results are saved in `reports/metrics.json` and `reports/training_report.json`.

### 2.6 Model versioning (the registry)
- **What:** every trained model gets a folder `artifacts/models/<version>/` containing `model.joblib` (the fitted pipeline) and `metadata.json` (version, training time, dataset fingerprint, feature names, seed, metrics, threshold, model type, status). The file `artifacts/current.json` names the version the API serves, plus the previous one and a promotion history.
- **Dataset fingerprint:** a hash (short code) computed from the training data's content. If even one value changes, the fingerprint changes, so you can always tell exactly which data a model learned from.
- **Why:** you must always be able to answer *"which model made this prediction, trained on what?"*, and you must be able to go back.
- **Safety detail:** a new version is written to a temporary folder and then renamed in one step, and the pointer file is swapped the same way (`os.replace`). A reader never sees a half-written model.
- **File:** `src/models/registry.py`
- **Common failure:** the API says `no promoted model yet`. Run `python scripts/setup.py`.

### 2.7 MLflow (experiment tracking)
- **What:** a tool that records each training run's parameters, metrics, tags and files in a database and shows them in a web UI.
- **Why:** after 20 experiments you won't remember which settings gave the best recall. MLflow does.
- **File:** `_log_to_mlflow()` in `src/models/train.py`. Data goes to `mlruns/mlflow.db`.
- **Run:** `python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000`, then open http://localhost:5000.
- **Expected:** experiment **predictive-maintenance** with runs named `initial-train-random_forest`, `initial-train-logistic_regression`, and later `retrain-...` runs. Click a run to see its metrics and the `model/` files.
- **Common failure:** the UI is empty because it was started from a different folder (it then creates a new, empty database). Start it from the project root with exactly the command above.

### 2.8 DVC (data and pipeline versioning)
- **What:** DVC does for data what Git does for code. `dvc.yaml` lists four stages (generate → validate → prepare → train), each with its command, inputs (*deps*), settings (*params*) and outputs (*outs*). `dvc.lock` records the hash of everything after a run.
- **Why:** reproducibility. Anyone can run `dvc repro` and rebuild the exact same data and model. DVC also skips stages whose inputs haven't changed.
- **Run:** `python -m dvc repro`, `python -m dvc status`, `python -m dvc dag`, `python -m dvc metrics show`
- **Expected:** on a second `dvc repro` → *"Data and pipelines are up to date."*
- **Note:** no remote storage is configured. Data versions live in `.dvc/cache` on this computer only (see README §11 for adding a remote).
- **Common failure:** `'python' is not recognized` inside `dvc repro`. The virtual environment isn't active, so activate it first.

### 2.9 FastAPI (the prediction service)
- **What:** a web server. Other programs send sensor readings as **JSON** (a text format for data) over **HTTP** (the protocol web browsers use) and get a prediction back. FastAPI also generates interactive documentation automatically (**Swagger UI**).
- **Why:** a model sitting in a Python file can't be used by a factory's software. An API makes it a service anyone can call.
- **Files:** `app/main.py` (routes), `app/schemas.py` (request rules), `app/predictor.py` (model loading and prediction).
- **Run:** `python -m uvicorn app.main:app --port 8000` (**Uvicorn** is the program that runs the FastAPI app), then open http://localhost:8000/docs.
- **Expected:** `/health` returns `{"status":"ok","model_loaded":true,...}`.
- **Design points:**
  - The model is loaded once and cached. When `current.json` changes (promotion or rollback), the next request automatically loads the new version.
  - Training **never** happens inside a prediction request.
  - Admin routes need the `X-API-Key` header to match the `PM_ADMIN_API_KEY` environment variable. No secret is written in the code.
- **Common failure:** `[Errno 10048] ... address already in use`. Another program is using port 8000. Stop it or use `--port 8001` (and set `API_URL` for the dashboard).

### 2.10 Streamlit dashboard
- **What:** a web page built in pure Python with four tabs. *Overview* shows model metrics and versions. *Predict* is a form with three demo samples. *Monitoring* shows counts, drift, live performance, recent predictions and sensor histograms. *Retraining* has simulate / retrain / rollback buttons plus status and history.
- **Why:** humans (and professors) understand a screen faster than JSON.
- **Important:** the dashboard has **no model of its own**. Every number comes from the API, so what you see is what the service really does.
- **File:** `app/dashboard.py`
- **Run:** `python -m streamlit run app/dashboard.py`, then open http://localhost:8501.
- **Common failure:** a red box saying *"Cannot reach the API at http://localhost:8000"*. Start the API first.

### 2.11 Monitoring (SQLite logs, drift, performance)
- **What:** every prediction is saved in `logs/monitoring.db` (SQLite is a database stored in a single file, with no server needed) with its inputs, probability, risk, model version and latency. Later, the real outcome can be attached through `POST /outcomes`.
- **Drift:** **PSI** (Population Stability Index) compares, for each feature, how the recent inputs are spread across value ranges ("bins") versus the training data:
  `PSI = Σ (current% − reference%) × ln(current% / reference%)`.
  0 means identical. Below 0.1 is stable, 0.1–0.25 is moderate, 0.25 or more is significant.
- **Why drift matters:** a model is only reliable on data that looks like what it learned from. If machines start running hotter, predictions may become unreliable. But drift is a *warning*, not proof of failure. Only real outcomes can show the model is actually wrong.
- **Performance:** with at least 30 recorded outcomes, `/monitoring/performance` computes live precision, recall and F1.
- **File:** `app/monitoring.py`
- **Expected:** after simulating drift 1.5, `/monitoring/drift` reports `"status": "significant"` with air temperature and tool wear listed.
- **Common failure:** `"status": "insufficient_data"`. Fewer than 50 predictions have been logged. Make more predictions or use *Simulate*.

### 2.12 Controlled retraining, promotion and rollback
- **What:** a procedure that creates a new model when there is reason to, and lets it replace the old one **only if it is genuinely not worse**.
- **Triggers:** a manual button or API call, or automatically (if the scheduler is enabled) when 200 or more new labelled outcomes exist since the last run, or when live F1 drops 0.10 or more below the training F1. Drift alone only raises an advisory.
- **Steps:**
  1. Take the **lock** (`artifacts/.retrain.lock`, created with an "only if it doesn't exist" flag). A second request gets HTTP 409. A lock older than 30 minutes is treated as left over from a crash and removed.
  2. Validate the new labelled data. If it's invalid, stop with `failed_validation`.
  3. Train the candidate on *old train + 70% of new data*.
  4. Score the candidate **and** the current model on the **same** evaluation data (*old test + the other 30% of new data*). Comparing on different data would be meaningless.
  5. Apply the rules: candidate F1 ≥ 0.50, candidate F1 ≥ current F1, candidate recall not more than 0.02 below current.
  6. **Promote** (swap the pointer) or **reject** (mark the candidate `rejected`; the current model keeps serving).
  7. Save the report, release the lock, and do so even if something crashed.
- **Rollback:** `POST /admin/rollback` points `current.json` back at the previous version. Because old versions are never deleted, this is instant.
- **Files:** `src/pipelines/retrain_pipeline.py` (the procedure), `app/retraining.py` (triggers, status, background thread, scheduler).
- **Run:** dashboard → Retraining tab → *Retrain now*, or `python scripts/run_demo.py`.
- **Expected:** `PROMOTED: all acceptance checks passed` or `REJECTED: candidate F1 0.727 not >= current 0.732 + 0.0`. Both outcomes are correct behaviour.
- **Common failure:** HTTP 409 *"a retraining run is already in progress"* when nothing is running (left over from a crash). Wait 30 minutes or delete `artifacts/.retrain.lock`.

### 2.13 Docker
- **What:** Docker packs the app, Python and all libraries into an **image**. A running image is a **container**. Docker Compose starts several containers together: `init` (prepares data and the model once), `api`, `dashboard` and `mlflow`.
- **Why:** "works on my machine" problems disappear, because everyone runs the identical environment.
- **Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- **Run:** `docker compose up --build`
- **Details:** the containers run as a normal (non-root) user, health checks restart unhealthy services, data survives in named volumes, the dashboard reaches the API at `http://api:8000` (Compose's internal network), and secrets come from `.env`, never from the image.
- **Status:** not run on the build machine (Docker isn't installed there). The CI pipeline builds and health-checks the image on GitHub.

### 2.14 GitHub Actions (CI)
- **What:** **Continuous Integration**. Every time code is pushed, GitHub starts a fresh Linux machine, installs the project and runs the checks.
- **Why:** it catches broken code automatically, before anyone relies on it.
- **File:** `.github/workflows/ci.yml`
- **What it runs:** lint (`ruff`), all tests, a small real training pipeline with output-file checks, and a Docker build that trains inside the container and waits for a healthy `/health`.
- **Where to look:** the **Actions** tab of the GitHub repository. A green tick means everything passed.

### 2.15 Tests
- **What:** 52 small programs that check the code behaves correctly. They run in temporary folders, so they never touch your real data.
- **File:** `tests/` (one file per area)
- **Run:** `python -m pytest -q`
- **Expected:** `52 passed`

---

## Part 3: Training vs inference

| | Training | Inference (prediction) |
|---|---|---|
| When | offline, occasionally | online, any time a request arrives |
| Input | thousands of rows **with** answers | one or a few rows **without** answers |
| Output | a model file + metrics | a probability + risk level |
| Speed | seconds to minutes | milliseconds |
| Where | `scripts/train_model.py`, retraining pipeline | `POST /predict` |
| Learns? | yes, fits scaler and trees | no, only applies what was learned |

---

## Part 4: One reading, traced from request to response

A technician's software sends this for machine M-007:
```json
{"machine_id": "M-007", "machine_type": "L", "air_temperature": 302.5, "process_temperature": 310.9,
 "rotational_speed": 1380, "torque": 58.4, "tool_wear": 215, "operating_hours": 15200,
 "maintenance_history_count": 5}
```
1. **Uvicorn** receives the HTTP request and hands it to **FastAPI** (`app/main.py`, `predict()`).
2. **Pydantic** (`SensorReading` in `app/schemas.py`) checks every field: type L/M/H, torque between 0 and 150, no extra fields, no NaN or infinity. Anything wrong → 422 error, and the model is never called.
3. `predictor.get_model()` reads `artifacts/current.json` → version `v…-6ed7`, already cached in memory.
4. The pipeline runs:
   - feature step: temp_difference = 310.9 − 302.5 = **8.4 K** (below the 9 K "good cooling" level); power = 58.4 × 1380 × 2π / 60 / 1000 = **8.44 kW** (near the overload zone); wear × torque = 215 × 58.4 / 1000 = **12.6**
   - scaling and one-hot encoding with the averages learned during training
   - the Random Forest's trees vote → probability **0.8872**
5. 0.8872 ≥ threshold 0.35 → `failure_predicted: true`, risk **high**.
6. `monitoring.log_predictions()` stores the reading, probability, version and latency in SQLite with a new `prediction_id`.
7. The JSON response returns to the caller. The dashboard shows it as a red **HIGH** badge.
8. Later, someone reports `{"prediction_id": "...", "actual_failure": 1}` to `/outcomes`. That row now counts toward live performance and toward the 200-label retraining trigger.

---

## Part 5: Glossary

| Term | Meaning |
|---|---|
| API | a defined way for programs to talk to each other, here over HTTP |
| Artifact | a file produced by a pipeline (data, model, report) |
| Baseline | a simple model used as a reference point |
| CI/CD | automatically testing (CI) and delivering (CD) code on every change |
| Class imbalance | one answer (no failure) is far more common than the other |
| Confusion matrix | table of true/false positives/negatives |
| Container / image | packaged, runnable copy of an app with its environment |
| Data drift | the input data's distribution changing over time |
| Data leakage | information the model wouldn't have in real life sneaking into training |
| Endpoint | one URL of an API, e.g. `/predict` |
| F1-score | harmonic mean of precision and recall |
| Fingerprint / hash | short code computed from content. It changes if the content changes |
| Hyperparameter | a setting chosen before training (e.g. number of trees) |
| JSON | a text format for structured data |
| Lock | a marker that stops two processes doing the same job at once |
| One-hot encoding | turning a category into several 0/1 columns |
| Pipeline | fixed sequence of processing steps |
| Promotion | making a model version the one that serves predictions |
| PSI | Population Stability Index, a number measuring distribution change |
| Recall | share of real failures the model caught |
| Reproducibility | getting the same result when rerunning with the same inputs |
| Rollback | returning to the previous model version |
| Seed | the starting number for a random generator. Same seed, same "random" numbers |
| Stratified split | train/test split that keeps the same class ratio in both parts |
| Threshold | probability cut-off for a yes/no decision |
| Virtual environment | a project-private Python installation |
