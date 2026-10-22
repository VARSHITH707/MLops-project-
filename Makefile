# Optional shortcuts (needs `make`; on Windows use the PowerShell commands in README.md).
PY ?= .venv/Scripts/python.exe

.PHONY: setup train test lint api dashboard mlflow demo check repro docker

setup:      ## generate data and train the first model
	$(PY) scripts/setup.py
train:
	$(PY) scripts/train_model.py --promote
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
api:
	$(PY) -m uvicorn app.main:app --port 8000
dashboard:
	$(PY) -m streamlit run app/dashboard.py
mlflow:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
demo:
	$(PY) scripts/run_demo.py
check:
	$(PY) scripts/check_system.py --api http://localhost:8000
repro:
	$(PY) -m dvc repro
docker:
	docker compose up --build
