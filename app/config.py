"""Runtime settings: file locations, parameters and secrets read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    home: Path
    params: dict
    admin_api_key: str | None
    mlflow_tracking_uri: str
    mlflow_enabled: bool
    scheduler_enabled: bool

    @property
    def raw_data(self) -> Path:
        return self.home / "data" / "raw" / "sensors.csv"

    @property
    def processed_dir(self) -> Path:
        return self.home / "data" / "processed"

    @property
    def train_csv(self) -> Path:
        return self.processed_dir / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.processed_dir / "test.csv"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.home / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def db_path(self) -> Path:
        return self.logs_dir / "monitoring.db"


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    """Read settings fresh each call so tests (and Docker) can redirect paths with env vars."""
    home = Path(os.getenv("PM_HOME", PROJECT_ROOT)).resolve()
    params_file = Path(os.getenv("PM_PARAMS_FILE", home / "params.yaml"))
    if not params_file.exists():
        params_file = PROJECT_ROOT / "params.yaml"
    with open(params_file, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)

    default_uri = f"sqlite:///{(home / 'mlruns' / 'mlflow.db').as_posix()}"
    return Settings(
        home=home,
        params=params,
        admin_api_key=os.getenv("PM_ADMIN_API_KEY") or None,
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", default_uri),
        mlflow_enabled=not _flag("PM_DISABLE_MLFLOW", False),
        scheduler_enabled=_flag("PM_SCHEDULER_ENABLED", False),
    )
