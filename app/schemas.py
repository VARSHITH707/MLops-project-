"""Request/response models. Pydantic rejects invalid payloads with a 422 error."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.data.validate_data import NUMERIC_RANGES


def _range(name: str, description: str):
    low, high = NUMERIC_RANGES[name]
    return Field(..., ge=low, le=high, allow_inf_nan=False, description=f"{description} [{low}-{high}]")


class SensorReading(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "machine_id": "M-007",
                "machine_type": "L",
                "air_temperature": 302.5,
                "process_temperature": 310.9,
                "rotational_speed": 1380,
                "torque": 58.4,
                "tool_wear": 215,
                "operating_hours": 15200,
                "maintenance_history_count": 5,
            }
        },
    )

    machine_id: str | None = Field(None, max_length=64, description="Optional, used only for logging")
    machine_type: Literal["L", "M", "H"] = Field(..., description="Machine quality variant")
    air_temperature: float = _range("air_temperature", "Ambient air temperature, kelvin")
    process_temperature: float = _range("process_temperature", "Process temperature, kelvin")
    rotational_speed: float = _range("rotational_speed", "Spindle speed, rpm")
    torque: float = _range("torque", "Torque, newton-metres")
    tool_wear: float = _range("tool_wear", "Tool wear, minutes")
    operating_hours: float = _range("operating_hours", "Total operating hours")
    maintenance_history_count: int = Field(..., ge=0, le=1000, description="Past maintenance events")


class BatchRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    prediction_id: str
    failure_predicted: bool
    failure_probability: float
    risk_category: Literal["low", "medium", "high"]
    threshold: float
    model_version: str
    timestamp: str
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class OutcomeRequest(BaseModel):
    prediction_id: str = Field(..., min_length=1, max_length=64)
    actual_failure: Literal[0, 1]
    source: Literal["observed", "simulated"] = "observed"


class RetrainRequest(BaseModel):
    reason: str = Field("manual trigger", max_length=200)
    wait: bool = Field(False, description="Block until retraining finishes (useful for demos/tests)")


class SimulateRequest(BaseModel):
    n: int = Field(300, ge=1, le=5000)
    seed: int = Field(7, ge=0)
    drift: float = Field(0.0, ge=0.0, le=3.0, description="0 = like training data, 1+ = shifted sensors")
    record_outcomes: bool = True
