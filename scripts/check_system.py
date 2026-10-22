"""Pre-demo health check: python version, packages, data, model, API, Docker.

    python scripts/check_system.py
    python scripts/check_system.py --api http://localhost:8000
Exit code 0 = everything needed for a local demo is ready.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from src.models import registry  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn", "pydantic", "fastapi", "uvicorn", "streamlit", "mlflow", "dvc", "pytest"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", help="also check a running API, e.g. http://localhost:8000")
    args = parser.parse_args(argv)
    settings = get_settings()
    failures = 0

    def report(ok: bool, label: str, detail: str = "", required: bool = True) -> None:
        nonlocal failures
        mark = "OK  " if ok else ("FAIL" if required else "WARN")
        print(f"[{mark}] {label}{' - ' + detail if detail else ''}")
        failures += int(required and not ok)

    report(sys.version_info[:2] == (3, 11), "Python 3.11", sys.version.split()[0])
    for pkg in PACKAGES:
        try:
            report(True, pkg, importlib.metadata.version(pkg))
        except importlib.metadata.PackageNotFoundError:
            report(False, pkg, "not installed", required=pkg not in {"dvc", "pytest"})

    report(settings.raw_data.exists(), "raw data", str(settings.raw_data))
    report(settings.train_csv.exists() and settings.test_csv.exists(), "train/test split", str(settings.processed_dir))
    version = registry.current_version(settings.artifacts_dir)
    report(version is not None, "served model", version or "none - run `python scripts/setup.py`")
    report(bool(settings.admin_api_key), "PM_ADMIN_API_KEY set", "admin endpoints need it", required=False)
    report(shutil.which("docker") is not None, "docker", "needed only for container deployment", required=False)

    if args.api:
        import requests

        try:
            health = requests.get(f"{args.api.rstrip('/')}/health", timeout=10).json()
            report(health.get("model_loaded", False), "API health", f"{args.api} -> {health.get('status')}")
        except requests.RequestException as exc:
            report(False, "API health", f"{args.api} unreachable ({type(exc).__name__})")

    print("\nREADY" if failures == 0 else f"\n{failures} required check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
