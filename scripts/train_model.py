"""Train on data/processed, save the model version, write reports.

The first model is promoted automatically (nothing to compare against). After that,
use the guarded retraining pipeline, or pass --promote to force it deliberately.
Exit codes: 0 success, 1 invalid data / training error, 2 missing input files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.config import get_settings  # noqa: E402
from src.data.validate_data import DataValidationError  # noqa: E402
from src.models import registry  # noqa: E402
from src.models.train import train_and_register, write_training_report  # noqa: E402

log = logging.getLogger("train_model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="promote even if a model is already served")
    args = parser.parse_args(argv)
    settings = get_settings()

    for path in (settings.train_csv, settings.test_csv):
        if not path.exists():
            log.error("Missing %s - run `python scripts/setup.py` or `dvc repro` first", path)
            return 2
    try:
        result = train_and_register(
            pd.read_csv(settings.train_csv), pd.read_csv(settings.test_csv), settings, run_name="initial-train"
        )
    except (DataValidationError, ValueError) as exc:
        log.error("Training failed: %s", exc)
        return 1

    report = write_training_report(result, settings.reports_dir)
    version = result["model_version"]
    if args.promote or registry.current_version(settings.artifacts_dir) is None:
        registry.promote(version, settings.artifacts_dir, reason="initial training")
        registry.update_metadata(version, settings.artifacts_dir, status="promoted")
        log.info("Promoted %s as the served model", version)
    else:
        log.info("Saved %s as a candidate; current model left unchanged", version)
    m = result["metrics"]
    log.info("Selected %s: precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f",
             result["model_type"], m["precision"], m["recall"], m["f1"], m["roc_auc"])
    log.info("Report: %s", report)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
