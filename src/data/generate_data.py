"""Reproducible synthetic industrial sensor data.

Each row is one sensor reading. The target ``failure_within_horizon`` is 1 when the
simulated machine fails within the next ``horizon_hours`` operating hours.

How labels are simulated: a hidden "risk score" rises when the tool is worn, when
mechanical power is outside a safe band, when heat cannot escape (small gap between
process and air temperature), with machine age, and for the low-quality machine type.
The score is turned into a probability with a sigmoid and a coin is flipped with that
probability. The coin flip (plus sensor noise) means the classes overlap, so the task
is learnable but not trivially separable.

This is a simulation of an engineering pipeline, not a model of any real machine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MACHINE_TYPES = ("L", "M", "H")  # low / medium / high quality variants
TARGET = "failure_within_horizon"


def _failure_probability(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    power_w = df["torque"] * df["rotational_speed"] * 2 * np.pi / 60
    temp_gap = df["process_temperature"] - df["air_temperature"]
    type_effect = df["machine_type"].map({"L": 1.0, "M": 0.0, "H": -1.0}).to_numpy()

    score = (
        -8.0
        + 2.75 * np.clip(df["tool_wear"] - 170, 0, None) / 20  # worn tool
        + 3.5 * np.clip(power_w - 8500, 0, None) / 1000  # overload
        + 3.5 * np.clip(3800 - power_w, 0, None) / 1000  # stall / underpowered
        + 2.5 * np.clip(9.0 - temp_gap, 0, None)  # poor heat dissipation
        + 0.00015 * df["operating_hours"]  # ageing
        + 0.2 * df["maintenance_history_count"]  # history of problems
        + type_effect
        + rng.normal(0, 1.5, len(df))  # unobserved factors
    )
    return 1 / (1 + np.exp(-score.to_numpy()))


def generate_dataset(
    n_samples: int,
    seed: int,
    n_machines: int = 50,
    start_time: str = "2026-01-01T00:00:00",
    drift: float = 0.0,
) -> pd.DataFrame:
    """Return ``n_samples`` labelled readings. ``drift`` > 0 shifts sensors (hotter, more worn)."""
    if n_samples <= 0 or n_machines <= 0:
        raise ValueError("n_samples and n_machines must be positive")
    rng = np.random.default_rng(seed)

    machine_idx = rng.integers(0, n_machines, n_samples)
    machine_ids = np.array([f"M-{i:03d}" for i in range(n_machines)])
    type_of_machine = rng.choice(MACHINE_TYPES, n_machines, p=[0.5, 0.3, 0.2])

    air = np.clip(rng.normal(300 + 3 * drift, 2, n_samples), 290, 315)
    process = np.clip(air + rng.normal(10 - 1.0 * drift, 1.0, n_samples), air + 2, 330)
    speed = np.clip(rng.normal(1500 - 60 * drift, 170, n_samples), 1150, 2900)
    torque = np.clip(40 - (speed - 1500) * 0.035 + rng.normal(0, 7, n_samples), 3, 80)
    tool_wear = np.clip(rng.uniform(0, 240, n_samples) + 40 * drift, 0, 300)
    operating_hours = rng.uniform(0, 20000, n_samples)
    maintenance = rng.poisson(operating_hours / 4000)

    # Readings of one machine are an hour apart, in row order.
    order = pd.Series(machine_idx).groupby(machine_idx).cumcount().to_numpy()
    timestamps = pd.Timestamp(start_time) + pd.to_timedelta(order, unit="h")

    df = pd.DataFrame(
        {
            "machine_id": machine_ids[machine_idx],
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%S"),
            "machine_type": type_of_machine[machine_idx],
            "air_temperature": air.round(2),
            "process_temperature": process.round(2),
            "rotational_speed": speed.round(0),
            "torque": torque.round(2),
            "tool_wear": tool_wear.round(0),
            "operating_hours": operating_hours.round(1),
            "maintenance_history_count": maintenance.astype(int),
        }
    )
    df[TARGET] = (rng.random(n_samples) < _failure_probability(df, rng)).astype(int)
    return df


def simulate_new_observations(n: int, seed: int, drift: float = 0.0) -> pd.DataFrame:
    """New simulated readings (with SIMULATED outcomes) for monitoring/retraining demos."""
    return generate_dataset(n, seed=seed, start_time="2026-06-01T00:00:00", drift=drift)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.config import get_settings

    settings = get_settings()
    cfg = settings.params["data"]
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data")
    parser.add_argument("--n-samples", type=int, default=cfg["n_samples"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--output", type=Path, default=settings.raw_data)
    args = parser.parse_args(argv)

    df = generate_dataset(args.n_samples, args.seed, cfg["n_machines"], cfg["start_time"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Wrote %d rows to %s (seed=%d)", len(df), args.output, args.seed)
    log.info("Failure rate: %.2f%%", 100 * df[TARGET].mean())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
