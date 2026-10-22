"""Feature engineering and preprocessing, shared by training and inference.

The engineered features are computed inside the scikit-learn Pipeline, so the API gets
exactly the same transformations as training. Scalers/encoders are fitted on the
training split only (``Pipeline.fit``), then reused unchanged at prediction time.

Not used as features (to prevent leakage / memorisation): machine_id, timestamp and
the target itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.generate_data import MACHINE_TYPES, TARGET
from src.data.validate_data import FEATURE_COLUMNS, NUMERIC_RANGES, validate_dataframe

log = logging.getLogger(__name__)

ENGINEERED = ["temp_difference", "power_kw", "wear_x_torque"]
NUMERIC_FEATURES = [*NUMERIC_RANGES, *ENGINEERED]
CATEGORICAL_FEATURES = ["machine_type"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLUMNS].copy()
    out["temp_difference"] = out["process_temperature"] - out["air_temperature"]
    out["power_kw"] = out["torque"] * out["rotational_speed"] * 2 * np.pi / 60 / 1000
    out["wear_x_torque"] = out["tool_wear"] * out["torque"] / 1000
    return out


def build_preprocessor() -> list[tuple[str, object]]:
    """Pipeline steps placed before the classifier."""
    columns = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(categories=[list(MACHINE_TYPES)], handle_unknown="error"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return [("features", FunctionTransformer(add_engineered_features)), ("preprocess", columns)]


def split_dataset(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split. A time-based split would be stricter for real sensor streams."""
    stratify = df[TARGET] if df[TARGET].nunique() > 1 and df[TARGET].value_counts().min() >= 2 else None
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.config import get_settings

    settings = get_settings()
    cfg = settings.params["data"]
    parser = argparse.ArgumentParser(description="Validate and split raw data into train/test")
    parser.add_argument("--input", type=Path, default=settings.raw_data)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    validate_dataframe(df)
    train, test = split_dataset(df, cfg["test_size"], cfg["seed"])
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(settings.train_csv, index=False)
    test.to_csv(settings.test_csv, index=False)
    log.info("Train rows: %d, test rows: %d", len(train), len(test))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
