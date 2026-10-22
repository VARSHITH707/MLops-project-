"""Explicit data validation. Invalid data is rejected with readable messages, never patched."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generate_data import MACHINE_TYPES, TARGET

log = logging.getLogger(__name__)

# Physically plausible limits (wider than what the generator produces).
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "air_temperature": (250.0, 350.0),  # kelvin
    "process_temperature": (250.0, 400.0),  # kelvin
    "rotational_speed": (0.0, 5000.0),  # rpm
    "torque": (0.0, 150.0),  # newton-metres
    "tool_wear": (0.0, 400.0),  # minutes of tool use
    "operating_hours": (0.0, 200000.0),
    "maintenance_history_count": (0.0, 1000.0),
}
ID_COLUMNS = ["machine_id", "timestamp"]
FEATURE_COLUMNS = ["machine_type", *NUMERIC_RANGES]


class DataValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_dataframe(
    df: pd.DataFrame, require_target: bool = True, check_duplicates: bool = True
) -> None:
    """Raise DataValidationError listing every problem found.

    ``check_duplicates=False`` is used for logged predictions, which are keyed by a unique
    prediction id instead (one batch request shares a timestamp)."""
    errors: list[str] = []
    if df.empty:
        raise DataValidationError(["dataset is empty"])

    required = ID_COLUMNS + FEATURE_COLUMNS + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError([f"missing required columns: {missing}"])

    nulls = df[required].isna().sum()
    for col, count in nulls[nulls > 0].items():
        errors.append(f"column '{col}' has {count} missing values")

    for col, (low, high) in NUMERIC_RANGES.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"column '{col}' must be numeric, got {df[col].dtype}")
            continue
        values = df[col].to_numpy(dtype=float)
        if np.isinf(values).any():
            errors.append(f"column '{col}' contains infinite values")
        bad = df[col].notna() & ~np.isinf(values) & ((values < low) | (values > high))
        if bad.any():
            errors.append(f"column '{col}' has {int(bad.sum())} values outside [{low}, {high}]")

    unknown = sorted(set(df["machine_type"].dropna().astype(str)) - set(MACHINE_TYPES))
    if unknown:
        errors.append(f"unknown machine_type values: {unknown}; allowed: {list(MACHINE_TYPES)}")

    if pd.to_datetime(df["timestamp"], errors="coerce", format="ISO8601", utc=True).isna().any():
        errors.append("column 'timestamp' has unparseable values")

    dupes = int(df.duplicated(subset=ID_COLUMNS).sum()) if check_duplicates else 0
    if dupes:
        errors.append(f"{dupes} duplicate (machine_id, timestamp) observations")

    if require_target:
        bad_target = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad_target:
            errors.append(f"target '{TARGET}' must be 0 or 1, found {sorted(bad_target)}")

    if errors:
        raise DataValidationError(errors)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validate the raw sensor dataset")
    parser.add_argument("--input", type=Path, default=settings.raw_data)
    parser.add_argument("--report", type=Path, default=settings.reports_dir / "data_validation.json")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    try:
        validate_dataframe(df)
        report = {"valid": True, "rows": len(df), "errors": []}
    except DataValidationError as exc:
        report = {"valid": False, "rows": len(df), "errors": exc.errors}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["valid"]:
        for err in report["errors"]:
            log.error("Validation error: %s", err)
        return 1
    log.info("Dataset valid: %d rows, report at %s", len(df), args.report)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
