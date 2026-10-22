import pandas as pd

from src.data.generate_data import TARGET, generate_dataset, simulate_new_observations
from src.data.validate_data import FEATURE_COLUMNS, NUMERIC_RANGES, validate_dataframe


def test_same_seed_gives_identical_data():
    pd.testing.assert_frame_equal(generate_dataset(500, seed=1), generate_dataset(500, seed=1))


def test_different_seed_gives_different_data():
    assert not generate_dataset(500, seed=1).equals(generate_dataset(500, seed=2))


def test_size_columns_and_ranges():
    df = generate_dataset(1000, seed=3)
    assert len(df) == 1000
    assert set(FEATURE_COLUMNS + [TARGET, "machine_id", "timestamp"]) <= set(df.columns)
    for col, (low, high) in NUMERIC_RANGES.items():
        assert df[col].between(low, high).all(), col
    validate_dataframe(df)


def test_target_is_binary_and_not_trivial():
    rate = generate_dataset(5000, seed=42)[TARGET].mean()
    assert 0.05 < rate < 0.35


def test_drift_shifts_sensors():
    base = simulate_new_observations(1000, seed=5)
    drifted = simulate_new_observations(1000, seed=5, drift=2.0)
    assert drifted["air_temperature"].mean() > base["air_temperature"].mean() + 3
    assert drifted["tool_wear"].mean() > base["tool_wear"].mean()
