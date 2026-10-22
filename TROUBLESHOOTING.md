# Troubleshooting

Each fix is marked **(seen)** if it happened while building this project and was resolved as shown, or **(general)** for standard diagnosis steps that weren't hit here.

## Python and installation

**`python -m pip` → "No module named pip", or `python --version` is not 3.11** (seen)
The `python` on PATH belongs to another tool or another version.
```powershell
py -0p                         # list installed Pythons
py -3.11 -m venv .venv         # build the venv from 3.11 explicitly
.\.venv\Scripts\python.exe --version
```
On the build machine, 3.11 came from `uv` (`uv python install 3.11`, then `uv venv --python 3.11`).

**"running scripts is disabled on this system" when activating** (general)
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and call `.\.venv\Scripts\python.exe` directly.

**`pip install` fails or is very slow** (general)
`python -m pip install --upgrade pip`, then retry. Check you're on Python 3.11 (pins are tested there). Behind a proxy, set `$env:HTTPS_PROXY`. If one package fails to build, read the first error line: it usually names a missing compiler or a wrong Python version.

**`ModuleNotFoundError: No module named 'src'` or `'app'`** (general)
Run commands from the project root and use the module form: `python -m src.data.generate_data`, `python -m uvicorn app.main:app`.

## Data, training and models

**`scripts/train_model.py` exits with code 2: "Missing ...train.csv"** (general)
Run `python scripts/setup.py` (or `python -m dvc repro`) first.

**Validation fails (exit code 1)** (general)
Read `reports/data_validation.json`. It lists each problem. Fix the data. Loosen a rule in `NUMERIC_RANGES` only if the rule itself is physically wrong.

**API: `503 no trained model is available`** (general)
`python scripts/setup.py`. Check `artifacts/current.json` exists and points to a folder in `artifacts/models/`.

**Model quality looks poor after changing the generator** (seen)
The first version of the generator made labels so noisy that F1 was only 0.445. The risk score was sharpened (see BUILD_HISTORY). If you edit `generate_data.py`, rerun `python -m dvc repro` and check `python -m dvc metrics show`.

## Running the services

**Port already in use (`[Errno 10048]`)** (general)
```powershell
netstat -ano | findstr :8000          # find the PID
Stop-Process -Id <PID>                 # or use another port:
python -m uvicorn app.main:app --port 8001
$env:API_URL = "http://localhost:8001"; python -m streamlit run app/dashboard.py
```

**Dashboard shows "Cannot reach the API"** (general)
Start the API first and wait for `Application startup complete` (5–10 s). Check http://localhost:8000/health. Under Docker, the dashboard uses `API_URL=http://api:8000`.

**Admin buttons say 401 or 503** (general)
503 means the API was started without `PM_ADMIN_API_KEY`: restart it with `$env:PM_ADMIN_API_KEY = "demo-key"`. 401 means the key typed in the dashboard sidebar doesn't match it.

**Requests return 422** (general)
The response body lists each invalid field (`loc`) and the reason (`msg`). Allowed ranges are shown in the Swagger schema at `/docs`.

**Dashboard message didn't appear after clicking** (seen)
The action takes a few seconds. Messages show *below* the button after it finishes. The first retraining is slower because MLflow loads.

## MLflow

**First training/retraining takes ~1 minute** (seen)
MLflow imports and creates its database tables on first use. Later runs are fast. To skip tracking: `$env:PM_DISABLE_MLFLOW = "1"`.

**MLflow UI shows no runs** (general)
It was started with a different backend. From the project root: `python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000`.

**"database is locked"** (general)
Two processes wrote at the same moment. Retry. Training continues even if logging fails (warning in the log).

## DVC

**`dvc` not recognized** (general)
Use `python -m dvc ...` with the venv active, or install the dev requirements (`pip install -r requirements-dev.txt`).

**`dvc repro` → "not a git repository" / "not a DVC repository"** (general)
Run it from the project root of a `git clone` (both `.git` and `.dvc` must exist). In a fresh folder: `git init`, then `python -m dvc init`.

**`dvc status` says `artifacts/models` changed after retraining** (expected)
Retraining adds versions to a *persistent* output. Run `python -m dvc commit train` if you want `dvc.lock` to record the new state.

**`dvc pull` / `dvc push` fail: "no remote"** (expected)
No remote storage is configured. See README §11 to add one.

## Retraining

**`409 a retraining run is already in progress`, but nothing is running** (general)
A crashed run left `artifacts/.retrain.lock`. It expires after 30 minutes (`lock_timeout_seconds`), or delete it: `Remove-Item artifacts\.retrain.lock`.

**Retraining status `failed_validation` with "timestamp" or "duplicate" errors** (seen, fixed)
Early versions failed on mixed timestamp formats and on batch predictions sharing a timestamp. Fixed by ISO-8601 parsing and by keying logged predictions on their unique id. If it reappears, read `decision_reasons` in `reports/retraining/<run>.json`.

**Candidate always rejected** (expected behaviour)
The candidate must match or beat the current F1 without losing more than 2 recall points. With little or noisy new data that's normal. Rules are in `params.yaml → retraining`.

**`409 no previous model version to roll back to`** (expected)
Only one model has ever been promoted.

## Docker

**`docker` not found** (seen)
Docker Desktop isn't installed. Everything runs without it (README §8). Install Docker Desktop (it requires WSL 2 on Windows) to use Compose.

**Build fails during `pip install`** (general)
Check internet access and free disk space. Retry with `docker compose build --no-cache`.

**`api` never starts / "dependency failed to start: init"** (general)
`docker compose logs init`. The init job trains the first model, and any error there blocks the API by design.

**Permission denied writing to `/app/...`** (general)
Volumes created by an older root-run image can be owned by root. Reset with `docker compose down -v` (this deletes the stored data and models) and run `docker compose up --build`.

## GitHub Actions

**Lint step fails** (general)
Run `python -m ruff check .` locally, fix the issues (`--fix` handles many automatically), and push.

**Tests pass locally but fail in CI** (general)
CI runs on Linux with a clean checkout. Look for Windows-only paths, files that exist only on your machine (data and models are git-ignored), or unpinned dependencies.

**Docker job fails health check** (general)
Expand "Train inside the container", where `docker logs pm` is printed at the end.
