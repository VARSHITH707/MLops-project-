import numpy as np
import pytest

from src.data.generate_data import TARGET, generate_dataset
from src.data.validate_data import DataValidationError, validate_dataframe
from src.features.build_features import ENGINEERED, add_engineered_features, split_dataset


@pytest.fixture
def df():
    return generate_dataset(300, seed=0)


def errors_for(frame, **kwargs):
    with pytest.raises(DataValidationError) as exc:
        validate_dataframe(frame, **kwargs)
    return " | ".join(exc.value.errors)


def test_valid_data_passes(df):
    validate_dataframe(df)


def test_empty_dataset_rejected(df):
    assert "empty" in errors_for(df.iloc[0:0])


def test_missing_column_rejected(df):
    assert "missing required columns" in errors_for(df.drop(columns=["torque"]))


def test_missing_values_rejected(df):
    df.loc[0, "torque"] = np.nan
    assert "missing values" in errors_for(df)


def test_out_of_range_rejected(df):
    df.loc[0, "rotational_speed"] = -5
    assert "outside" in errors_for(df)


def test_infinite_rejected(df):
    df.loc[0, "air_temperature"] = np.inf
    assert "infinite" in errors_for(df)


def test_wrong_type_rejected(df):
    df["torque"] = "high"
    assert "must be numeric" in errors_for(df)


def test_unknown_category_rejected(df):
    df.loc[0, "machine_type"] = "Z"
    assert "unknown machine_type" in errors_for(df)


def test_invalid_target_rejected(df):
    df.loc[0, TARGET] = 2
    assert "must be 0 or 1" in errors_for(df)


def test_duplicates_rejected(df):
    assert "duplicate" in errors_for(df.iloc[[0, 0, 1]])


def test_target_optional_for_inference(df):
    validate_dataframe(df.drop(columns=[TARGET]), require_target=False)


def test_engineered_features(df):
    out = add_engineered_features(df)
    assert set(ENGINEERED) <= set(out.columns)
    assert "machine_id" not in out and "timestamp" not in out and TARGET not in out
    np.testing.assert_allclose(out["temp_difference"], df["process_temperature"] - df["air_temperature"])


def test_split_is_reproducible_and_stratified(df):
    a_train, a_test = split_dataset(df, 0.2, seed=1)
    b_train, b_test = split_dataset(df, 0.2, seed=1)
    assert a_test.equals(b_test) and len(a_train) + len(a_test) == len(df)
    assert abs(a_test[TARGET].mean() - df[TARGET].mean()) < 0.05
