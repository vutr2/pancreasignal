"""Request / response models of the API (Phase 6). Pydantic checks every request before the model
sees it; anything invalid gets HTTP 422 with a message saying what is wrong.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pancreasignal.api.units import DATASET_UNITS, to_dataset_unit

# request field -> analyte name used by the model and api/units.py
ANALYTES = {"lyve1": "LYVE1", "reg1b": "REG1B", "tff1": "TFF1", "creatinine": "creatinine"}


class Measurement(BaseModel):
    """One lab value WITH its unit, e.g. {"value": 8.8, "unit": "mmol/L"}."""
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0, allow_inf_nan=False)
    unit: str = Field(min_length=1, examples=["ng/mL"])


class PredictRequest(BaseModel):
    """One urine sample + age + sex. Each lab value carries its own unit."""
    # extra="forbid": a typo such as "lyve_1" must fail loudly, not be silently ignored
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "lyve1": {"value": 3.0, "unit": "ng/mL"},
        "reg1b": {"value": 100.0, "unit": "ng/mL"},
        "tff1": {"value": 500.0, "unit": "ng/mL"},
        "creatinine": {"value": 7.1, "unit": "mmol/L"},
        "age": 66,
        "sex": "M",
    }]})

    lyve1: Measurement
    reg1b: Measurement
    tff1: Measurement
    creatinine: Measurement
    age: float = Field(gt=0, lt=120, allow_inf_nan=False)
    sex: Literal["M", "F"]

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, value):
        """Accept 'M', 'm', 'male', 'Male' (and the same for F); anything else -> 422."""
        if isinstance(value, str):
            return {"m": "M", "male": "M", "f": "F", "female": "F"}.get(value.strip().lower(), value)
        return value

    @model_validator(mode="after")
    def check_units(self):
        """Reject unknown units here, so the error is a 422 that names the allowed units."""
        self.in_dataset_units()
        return self

    def in_dataset_units(self) -> dict[str, float]:
        """{'LYVE1': ng/mL, 'REG1B': ng/mL, 'TFF1': ng/mL, 'creatinine': mg/mL}."""
        return {analyte: to_dataset_unit(analyte, getattr(self, field).value, getattr(self, field).unit)
                for field, analyte in ANALYTES.items()}


class Reason(BaseModel):
    feature: str
    direction: Literal["raises", "lowers"]
    contribution_log_odds: float
    training_percentile: float | None
    text: str


class OutOfRange(BaseModel):
    feature: str
    value: float
    training_min: float
    training_max: float


class PredictResponse(BaseModel):
    probability: float = Field(description="Model output for the training population (case-control); "
                                           "NOT a real-world risk")
    reasons: list[Reason] = Field(description="Top 3 features by size of their effect on this prediction")
    out_of_distribution: bool
    out_of_range: list[OutOfRange]
    inputs_in_dataset_units: dict[str, float] = Field(
        description=f"What the model saw after unit conversion: {DATASET_UNITS}")
    warnings: list[str]
    model: dict
