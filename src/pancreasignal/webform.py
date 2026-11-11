"""The demo's input form <-> the agent's case JSON. Shared by the Gradio demo (demo.py) and the browser
demo (browser_demo.py), so both build EXACTLY the same case from the same fields. Pure Python (no
Gradio, no scikit-learn). Tests: tests/test_demo.py

Form fields, in this order (case_from_form's arguments):
    case_id, lyve1, lyve1_unit, reg1b, reg1b_unit, tff1, tff1_unit, creatinine, creatinine_unit, age, sex,
    ca19_9, ca19_9_unit, bilirubin, bilirubin_unit, jaundice, new_onset_diabetes
"""
from __future__ import annotations

from pancreasignal.api.units import CREATININE_UNITS, PROTEIN_UNITS, normalize_unit

PROTEIN_UNIT_CHOICES = list(PROTEIN_UNITS)
CREATININE_UNIT_CHOICES = list(CREATININE_UNITS)
CA199_UNIT_CHOICES = ["U/mL", "kU/L"]
BILIRUBIN_UNIT_CHOICES = ["mg/dL", "µmol/L"]
URINE_ANALYTES = ("lyve1", "reg1b", "tff1", "creatinine")


def _unit_choice(unit: str, choices: list[str]) -> str:
    """The dropdown spelling of `unit` ('ug/mL' -> 'µg/mL'); a unit not in the list is kept as typed."""
    return next((c for c in choices if normalize_unit(c) == normalize_unit(unit)), unit)


def form_from_case(case: dict) -> list:
    """Form values, in case_from_form's argument order, for one case of evals/agent_cases.yaml."""
    urine, clinical = case.get("urine") or {}, case.get("clinical") or {}
    values: list = [case["case_id"]]
    # Step 1: urine panel: value + unit per analyte (missing -> empty value, default unit)
    for analyte in URINE_ANALYTES:
        choices = CREATININE_UNIT_CHOICES if analyte == "creatinine" else PROTEIN_UNIT_CHOICES
        m = urine.get(analyte)
        values += [m["value"], _unit_choice(m["unit"], choices)] if m else [None, choices[0]]
    values += [urine.get("age"), urine.get("sex")]
    # Step 2: clinical context
    for key, choices in (("plasma_ca19_9", CA199_UNIT_CHOICES), ("total_bilirubin", BILIRUBIN_UNIT_CHOICES)):
        m = clinical.get(key)
        values += [m["value"], _unit_choice(m["unit"], choices)] if m else [None, choices[0]]
    return [*values, bool(clinical.get("jaundice")), bool(clinical.get("new_onset_diabetes"))]


def case_from_form(case_id, lyve1, lyve1_unit, reg1b, reg1b_unit, tff1, tff1_unit, creatinine, creatinine_unit,
                   age, sex, ca19_9, ca19_9_unit, bilirubin, bilirubin_unit, jaundice, new_onset_diabetes) -> dict:
    """The agent's case JSON. Empty fields are LEFT OUT, so the agent's own intake validation reports
    them (no creatinine -> 'urine.creatinine: Field required'); nothing is filled in by the web page."""
    def m(value, unit):
        return None if value is None else {"value": float(value), "unit": unit}

    urine = {"lyve1": m(lyve1, lyve1_unit), "reg1b": m(reg1b, reg1b_unit), "tff1": m(tff1, tff1_unit),
             "creatinine": m(creatinine, creatinine_unit), "age": age, "sex": sex or None}
    clinical = {"plasma_ca19_9": m(ca19_9, ca19_9_unit), "total_bilirubin": m(bilirubin, bilirubin_unit),
                # an unticked box means "not stated", not "no": the critic only acts on what is known
                "jaundice": True if jaundice else None,
                "new_onset_diabetes": True if new_onset_diabetes else None}
    return {"case_id": (case_id or "").strip() or "my_case",
            "urine": {k: v for k, v in urine.items() if v is not None},
            "clinical": {k: v for k, v in clinical.items() if v is not None}}


