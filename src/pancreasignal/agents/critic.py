"""The critic (Phase 9): fixed rules that decide whether the risk estimate can be shown, and what the
reader must know about it. Tests: tests/test_agent.py

Rules, not an LLM, on purpose: every decision is explainable, testable and cannot be hallucinated.
Each finding has a severity:
  block : the estimate must NOT be shown (abstain), e.g. a probable unit error
  warn  : shown, with a clear caveat, e.g. CA19-9 measured during jaundice
  info  : context for the reader, e.g. new-onset diabetes is a known high-risk group
and may carry a question for the guideline RAG, so the report can cite evidence for the caveat.

Thresholds marked HEURISTIC are common practice, not taken from the corpus; they are flags for a
human reviewer, not clinical rules.
"""
from __future__ import annotations

CA199_ULN_U_ML = 37.0        # usual upper limit of normal for plasma CA19-9 (also used in the corpus)
CA199_VERY_LOW_U_ML = 2.0    # HEURISTIC: ~0 may mean the person cannot make CA19-9 (Lewis negative)
BILIRUBIN_RAISED_MG_DL = 2.0  # HEURISTIC: total bilirubin clearly above normal (~1.2 mg/dL)
CREATININE_DILUTE_MG_ML = 0.2   # HEURISTIC: 20 mg/dL, the usual "dilute urine" cut-off
CREATININE_CONCENTRATED_MG_ML = 3.0  # HEURISTIC: 300 mg/dL, very concentrated urine

Q_JAUNDICE = "Is CA19-9 reliable in a patient with obstructive jaundice?"
Q_LEWIS = "Why do some pancreatic cancer patients have undetectable CA19-9?"
Q_DIABETES = "Should people with new-onset diabetes be investigated for pancreatic cancer?"


def _finding(code: str, severity: str, en: str, vi: str, question: str | None = None) -> dict:
    return {"code": code, "severity": severity, "message_en": en, "message_vi": vi, "rag_question": question}


def review_case(intake_errors: list[str], risk: dict | None, clinical: dict | None,
                risk_error: str | None = None) -> dict:
    """All findings + the decision ('proceed' or 'abstain'). `clinical` = ClinicalContext.normalised();
    `risk_error` = why the risk model could not answer (e.g. the API was down)."""
    findings: list[dict] = []
    clinical = clinical or {}

    # Step 1: invalid input -> nothing else can be trusted
    if intake_errors:
        findings.append(_finding(
            "INTAKE_INVALID", "block",
            "The case could not be read: " + "; ".join(intake_errors),
            "Không đọc được dữ liệu ca bệnh: " + "; ".join(intake_errors)))
        return {"decision": "abstain", "findings": findings}
    if risk_error:
        findings.append(_finding(
            "RISK_UNAVAILABLE", "block",
            f"The risk model did not answer: {risk_error}",
            f"Model nguy cơ không trả lời được: {risk_error}"))

    # Step 2: the risk model's own warnings (a value 10x outside training is most likely a unit error)
    if risk is not None:
        unit_warnings = [w for w in risk["warnings"] if "check the unit" in w]
        if unit_warnings:
            findings.append(_finding(
                "UNIT_SUSPECT", "block",
                "A value is far outside the training data, most likely a unit error: " + " ".join(unit_warnings),
                "Có giá trị nằm rất xa dữ liệu huấn luyện, nhiều khả năng nhập sai đơn vị: " + " ".join(unit_warnings)))
        elif risk["out_of_distribution"]:
            names = ", ".join(sorted({o["feature"] for o in risk["out_of_range"]}))
            findings.append(_finding(
                "OUT_OF_DISTRIBUTION", "warn",
                f"Outside the training data ({names}): the estimate is an extrapolation.",
                f"Nằm ngoài dữ liệu huấn luyện ({names}): kết quả là ngoại suy, độ tin cậy thấp."))

        # Step 3: urine concentration (every protein is divided by creatinine)
        creatinine = risk["inputs_in_dataset_units"]["creatinine"]
        if creatinine < CREATININE_DILUTE_MG_ML:
            findings.append(_finding(
                "DILUTE_URINE", "warn",
                f"Very dilute urine (creatinine {creatinine:.2f} mg/mL): creatinine normalisation is less reliable; repeat on a morning sample.",
                f"Nước tiểu rất loãng (creatinine {creatinine:.2f} mg/mL): chuẩn hoá theo creatinine kém tin cậy; nên lấy lại mẫu nước tiểu buổi sáng."))
        elif creatinine > CREATININE_CONCENTRATED_MG_ML:
            findings.append(_finding(
                "CONCENTRATED_URINE", "warn",
                f"Very concentrated urine (creatinine {creatinine:.2f} mg/mL): creatinine normalisation is less reliable.",
                f"Nước tiểu rất đặc (creatinine {creatinine:.2f} mg/mL): chuẩn hoá theo creatinine kém tin cậy."))

    # Step 4: cholestasis makes plasma CA19-9 unreliable (false positives)
    ca199, bilirubin = clinical.get("ca19_9_u_ml"), clinical.get("bilirubin_mg_dl")
    cholestasis = bool(clinical.get("jaundice")) or (bilirubin is not None and bilirubin > BILIRUBIN_RAISED_MG_DL)
    if cholestasis and ca199 is not None and ca199 > CA199_ULN_U_ML:
        findings.append(_finding(
            "CHOLESTASIS_CA199", "warn",
            f"CA19-9 is raised ({ca199:g} U/mL) during jaundice/cholestasis: it can be falsely high; re-measure after biliary drainage.",
            f"CA19-9 tăng ({ca199:g} U/mL) trong lúc vàng da/ứ mật: có thể dương tính giả; nên đo lại sau khi dẫn lưu mật.",
            Q_JAUNDICE))
    elif cholestasis:
        findings.append(_finding(
            "CHOLESTASIS", "info",
            "Jaundice/cholestasis: a plasma CA19-9 measured now would be unreliable.",
            "Đang vàng da/ứ mật: nếu đo CA19-9 huyết tương lúc này thì kết quả kém tin cậy.",
            Q_JAUNDICE))

    # Step 5: CA19-9 ~0 cannot rule anything out (the person may be unable to make it)
    if ca199 is not None and ca199 <= CA199_VERY_LOW_U_ML:
        findings.append(_finding(
            "CA199_VERY_LOW", "info",
            f"CA19-9 is almost undetectable ({ca199:g} U/mL): the person may be Lewis antigen negative, so a normal CA19-9 does not rule out cancer.",
            f"CA19-9 gần như không đo được ({ca199:g} U/mL): có thể người này âm tính kháng nguyên Lewis, nên CA19-9 bình thường không loại trừ được ung thư.",
            Q_LEWIS))

    # Step 6: risk-group context
    if clinical.get("new_onset_diabetes"):
        findings.append(_finding(
            "NEW_ONSET_DIABETES", "info",
            "New-onset diabetes is a recognised high-risk group for pancreatic cancer.",
            "Đái tháo đường mới khởi phát là nhóm nguy cơ cao đã được ghi nhận đối với ung thư tuỵ.",
            Q_DIABETES))

    # Step 7: one 'block' is enough to withhold the estimate
    decision = "abstain" if any(f["severity"] == "block" for f in findings) else "proceed"
    return {"decision": decision, "findings": findings}
