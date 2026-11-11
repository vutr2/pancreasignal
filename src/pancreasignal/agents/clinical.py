"""The agent's input: one case = urine panel (as the API takes it) + optional clinical context (Phase 9).
Tests: tests/test_agent.py

The urine model does not use the clinical context. The CRITIC does: e.g. jaundice makes a high
plasma CA19-9 unreliable, a CA19-9 of ~0 may mean the person cannot make it (Lewis negative).
Phase 7 (reading lab-report images) was skipped, so cases arrive as structured JSON with units.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pancreasignal.api.schemas import Measurement, PredictRequest
from pancreasignal.api.units import normalize_unit

BILIRUBIN_UMOL_PER_MG_DL = 17.1   # total bilirubin: 1 mg/dL = 17.1 µmol/L
BILIRUBIN_UNITS = {"mg/dl": 1.0, "umol/l": 1 / BILIRUBIN_UMOL_PER_MG_DL}   # -> mg/dL
CA199_UNITS = {"u/ml": 1.0, "ku/l": 1.0}                                  # 1 kU/L = 1 U/mL


def _convert(value: float, unit: str, table: dict[str, float], name: str) -> float:
    """Value in the reference unit of `table`; unknown units are rejected, never guessed."""
    key = normalize_unit(unit)
    if key not in table:
        raise ValueError(f"Unknown unit {unit!r} for {name}. Allowed: {sorted(table)}")
    return value * table[key]


class ClinicalContext(BaseModel):
    """Optional facts the critic uses. Anything left out is simply not checked."""
    model_config = ConfigDict(extra="forbid")

    plasma_ca19_9: Measurement | None = None    # U/mL (or kU/L)
    total_bilirubin: Measurement | None = None  # mg/dL or µmol/L
    jaundice: bool | None = None
    new_onset_diabetes: bool | None = None

    @model_validator(mode="after")
    def check_units(self):
        self.normalised()          # unknown units -> 422-style validation error here
        return self

    def normalised(self) -> dict:
        """{'ca19_9_u_ml': float | None, 'bilirubin_mg_dl': float | None, 'jaundice': ..., 'new_onset_diabetes': ...}"""
        return {
            "ca19_9_u_ml": None if self.plasma_ca19_9 is None else
            _convert(self.plasma_ca19_9.value, self.plasma_ca19_9.unit, CA199_UNITS, "plasma CA19-9"),
            "bilirubin_mg_dl": None if self.total_bilirubin is None else
            _convert(self.total_bilirubin.value, self.total_bilirubin.unit, BILIRUBIN_UNITS, "total bilirubin"),
            "jaundice": self.jaundice,
            "new_onset_diabetes": self.new_onset_diabetes,
        }


class CaseInput(BaseModel):
    """One case for the agent: an id, the urine panel request, and the clinical context."""
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")  # safe as a file name
    urine: PredictRequest
    clinical: ClinicalContext = ClinicalContext()
