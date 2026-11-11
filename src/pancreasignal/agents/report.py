"""Bilingual (English / Vietnamese) case report (Phase 9). Tests: tests/test_agent.py

Built from TEMPLATES, not written by an LLM: every sentence comes either from fixed text, from the
risk model's numbers, from the critic's rules, or verbatim from the cited guideline evidence
(which stays in English, the language of the sources, with its citations). Nothing new is invented
at this step, and the Vietnamese version is built from the same fields, not machine-translated.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

SEVERITY = {"block": ("BLOCK", "CHẶN"), "warn": ("WARNING", "CẢNH BÁO"), "info": ("NOTE", "LƯU Ý")}
STATUS = {"approve": ("APPROVED by reviewer", "ĐÃ DUYỆT bởi người duyệt"),
          "reject": ("REJECTED by reviewer: not for release", "BỊ TỪ CHỐI bởi người duyệt: không phát hành")}
DISCLAIMER = ("Research prototype. Not a medical device. Not for clinical use.",
              "Nguyên mẫu nghiên cứu. Không phải thiết bị y tế. Không dùng cho chẩn đoán hay điều trị.")
FEATURE_VI = {"age": "Tuổi", "sex": "Giới tính"}


def _reason_vi(r: dict) -> str:
    """The Vietnamese reason, rebuilt from the reason's fields (not translated from the English text)."""
    tang = "làm tăng" if r["direction"] == "raises" else "làm giảm"
    if r["feature"] == "sex":
        return f"Giới tính {tang} nguy cơ một chút."
    if r["feature"] == "age":
        return f"Tuổi (bách phân vị {r['training_percentile']:.0f} của bệnh nhân huấn luyện) {tang} nguy cơ."
    return f"{r['feature']} (chia theo creatinine) ở bách phân vị {r['training_percentile']:.0f} của mẫu huấn luyện; {tang} nguy cơ."


def _evidence_lines(evidence: list[dict]) -> list[str]:
    """Guideline evidence, verbatim with citations (same block in both languages)."""
    lines = []
    for e in evidence:
        lines.append(f"- **Q:** {e['question']}")
        if e.get("error"):
            lines.append(f"  - _(evidence unavailable / không lấy được bằng chứng: {e['error']})_")
            continue
        lines.append(f"  - **A:** {e['answer']}")
        for c in e.get("citations", []):
            lines.append(f"    - [{c['chunk_id']}] {c['title']} ({c['journal']}, {c['year']}), {c['section'][:90]}")
    return lines or ["- (none / không có)"]


def build_report(state: dict) -> dict:
    """{'markdown': str, 'summary': dict} for a finished run (any decision, including rejected)."""
    case_id = state.get("case_id", "?")
    critic, risk = state["critic"], state.get("risk")
    review = state.get("review") or {}
    status_en, status_vi = STATUS.get(review.get("decision"), ("NOT REVIEWED", "CHƯA DUYỆT"))
    show_risk = critic["decision"] == "proceed" and risk is not None

    # Step 1: the risk block, or why it is withheld (both languages)
    if show_risk:
        perf = risk["model"].get("performance_reference") or {}
        auc = f"{perf['auroc']:.3f}" if perf.get("auroc") is not None else "n/a"
        risk_en = [(f"**Model output: {risk['probability']:.0%}** (urine panel, task {risk['model']['task']}; "
                   f"validation AUROC across hospitals {auc})."),
                   *[f"- {w}" for w in risk["warnings"]],
                   "", "What drove it (top 3):", *[f"- {r['text']}" for r in risk["reasons"]]]
        risk_vi = [(f"**Kết quả của model: {risk['probability']:.0%}** (bộ marker nước tiểu, bài toán {risk['model']['task']}; "
                   f"AUROC khi kiểm định trên bệnh viện khác: {auc})."),
                   "- Đây là xác suất từ dữ liệu bệnh-chứng (34% PDAC), KHÔNG phải nguy cơ thực tế trong sàng lọc.",
                   "- Xác suất chưa được hiệu chỉnh cho bệnh viện mới.",
                   "", "Yếu tố ảnh hưởng nhiều nhất (top 3):", *[f"- {_reason_vi(r)}" for r in risk["reasons"]]]
    else:
        blocks = [f for f in critic["findings"] if f["severity"] == "block"]
        risk_en = ["**Risk estimate withheld.** " + " ".join(f["message_en"] for f in blocks)]
        risk_vi = ["**Không đưa ra ước tính nguy cơ.** " + " ".join(f["message_vi"] for f in blocks)]

    # Step 2: critic findings in each language
    checks_en = [f"- [{SEVERITY[f['severity']][0]}] {f['message_en']}" for f in critic["findings"]] or ["- None."]
    checks_vi = [f"- [{SEVERITY[f['severity']][1]}] {f['message_vi']}" for f in critic["findings"]] or ["- Không có."]
    evidence = _evidence_lines(state.get("evidence", []))
    reviewer = f"{review.get('reviewer', '?')}: {review.get('note') or '-'}"

    # Step 3: assemble
    md = [f"# PancreaSignal case report: {case_id}", "",
          f"_{DISCLAIMER[0]}_  ", f"_{DISCLAIMER[1]}_", "",
          f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} | Status: **{status_en}** | Trạng thái: **{status_vi}**", "",
          "## English", "", "### Risk estimate", "", *risk_en, "", "### Checks", "", *checks_en, "",
          "### Guideline evidence", "", *evidence, "", f"### Reviewer\n\n{status_en}. {reviewer}", "",
          "## Tiếng Việt", "", "### Ước tính nguy cơ", "", *risk_vi, "", "### Kiểm tra", "", *checks_vi, "",
          "### Bằng chứng từ y văn (trích nguyên văn tiếng Anh, kèm nguồn)", "", *evidence, "",
          f"### Người duyệt\n\n{status_vi}. {reviewer}", ""]
    summary = {"case_id": case_id, "decision": critic["decision"], "review": review.get("decision"),
               "probability": risk["probability"] if show_risk else None,
               "findings": [f["code"] for f in critic["findings"]],
               "evidence_questions": [e["question"] for e in state.get("evidence", [])]}
    return {"markdown": "\n".join(md), "summary": summary}


def save_report(report: dict, out_dir: str | Path, run_id: str) -> dict:
    """Write <run_id>.md (for people) and <run_id>.json (for tests and audits)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out / f"{run_id}.md", out / f"{run_id}.json"
    md_path.write_text(report["markdown"])
    json_path.write_text(json.dumps(report["summary"], indent=2, ensure_ascii=False) + "\n")
    return {"markdown": str(md_path), "json": str(json_path)}
