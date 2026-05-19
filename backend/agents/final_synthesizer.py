"""Stage 7 — Final Synthesis Agent.

Merges all resolved sub-answers into a single FinalAnswer and also populates
final_response (API-compatible QueryResponse shape) for the existing endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from agents.llm import llm_completion, llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This information is for educational purposes only and does not constitute legal advice. "
    "Always consult qualified regulatory counsel before making compliance decisions. "
    "Regulations change — verify against the current eCFR before relying on any information provided."
)

_RULING_SYSTEM = """\
You are an FDA regulatory compliance expert.  Based on the provided sub-answers, issue a direct
ruling for the original user question.

Return ONLY valid JSON:
{
  "ruling": "YES" | "NO" | "CONDITIONAL",
  "ruling_summary": "1-2 sentence plain-language summary",
  "requirements": [
    {"item": "specific requirement", "citation": "[§101.54]"}
  ],
  "compliance_checklist": [
    "Step 1: ...",
    "Step 2: ..."
  ]
}"""


def _compute_confidence_level(sub_answers: list[dict]) -> tuple[str, str]:
    """Derive confidence_level and explanation from sub-answer confidences."""
    if not sub_answers:
        return "LOW", "No sub-answers available."
    confidences = [sa.get("confidence", 0.0) for sa in sub_answers]
    avg = sum(confidences) / len(confidences)
    low_sqs = [sa["sub_question_text"] for sa in sub_answers if sa.get("confidence", 1.0) < 0.5]

    if avg >= 0.8:
        level = "HIGH"
        explanation = f"Average confidence {avg:.0%} across {len(sub_answers)} sub-question(s)."
    elif avg >= 0.5:
        level = "MEDIUM"
        explanation = f"Average confidence {avg:.0%}."
    else:
        level = "LOW"
        explanation = f"Average confidence {avg:.0%}."

    if low_sqs:
        explanation += f" Low-confidence sub-questions: {'; '.join(low_sqs[:2])}."

    return level, explanation


def _build_all_citations(sub_answers: list[dict]) -> list[str]:
    """Collect and deduplicate all CFR section citations across sub-answers."""
    seen: set[str] = set()
    result: list[str] = []
    for sa in sub_answers:
        for cit in sa.get("citations", []):
            sec = cit.get("cfr_section", "")
            if sec and sec not in seen:
                seen.add(sec)
                result.append(sec)
    return result


def _build_regulation_excerpts(sub_answers: list[dict]) -> list[dict]:
    """Deduplicated list of regulation excerpts used across sub-answers."""
    seen: set[str] = set()
    excerpts: list[dict] = []
    for sa in sub_answers:
        for chunk in sa.get("chunks_used", []):
            cit = chunk.get("cfr_citation", "")
            if cit and cit not in seen:
                seen.add(cit)
                excerpts.append({"cfr_section": cit, "text": chunk.get("text", "")[:400]})
    return excerpts


def _aggregate_caveats(sub_answers: list[dict]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for sa in sub_answers:
        for caveat in sa.get("caveats", []):
            if caveat and caveat not in seen:
                seen.add(caveat)
                result.append(caveat)
    return result


def final_synthesizer_node(state: ComplianceState) -> dict:
    """Assemble the FinalAnswer from resolved sub-answers and generate final_response."""
    query = state.get("query", "")
    resolved_answers = state.get("resolved_answers", state.get("sub_answers", []))
    unresolved_conflicts = state.get("unresolved_conflicts", [])

    if not resolved_answers:
        final_answer = {
            "ruling": "NO",
            "ruling_summary": "Insufficient information to answer the question.",
            "requirements": [],
            "confidence_level": "LOW",
            "confidence_explanation": "No sub-answers available.",
            "regulation_excerpts": [],
            "unresolved_conflicts": unresolved_conflicts,
            "low_confidence_sub_questions": [],
            "caveats": [],
            "compliance_checklist": [],
            "all_citations": [],
        }
        final_response = _build_query_response(final_answer, unresolved_conflicts)
        return {"final_answer": final_answer, "final_response": final_response}

    # ── Ruling determination ────────────────────────────────────────────────────
    sub_answers_summary = "\n\n".join(
        f"Sub-question: {sa['sub_question_text']}\nAnswer: {sa.get('answer', '')[:500]}"
        for sa in resolved_answers
    )
    messages = [
        {"role": "system", "content": _RULING_SYSTEM},
        {"role": "user", "content": f"Original question: {query}\n\n{sub_answers_summary}"},
    ]
    try:
        raw = llm_completion_json(messages, max_tokens=1024, temperature=0.1)
        ruling_result = parse_llm_json(raw, messages)
    except Exception as exc:
        logger.warning("Ruling LLM call failed: %s", exc)
        ruling_result = {
            "ruling": "CONDITIONAL",
            "ruling_summary": "See sub-answers for details.",
            "requirements": [],
            "compliance_checklist": [],
        }

    ruling = ruling_result.get("ruling", "CONDITIONAL")
    ruling_summary = ruling_result.get("ruling_summary", "")
    requirements = ruling_result.get("requirements", [])
    compliance_checklist = ruling_result.get("compliance_checklist", [])

    # ── Assemble FinalAnswer ────────────────────────────────────────────────────
    confidence_level, confidence_explanation = _compute_confidence_level(resolved_answers)
    all_citations = _build_all_citations(resolved_answers)
    regulation_excerpts = _build_regulation_excerpts(resolved_answers)
    caveats = _aggregate_caveats(resolved_answers)
    low_conf_sqs = [sa["sub_question_text"] for sa in resolved_answers if sa.get("confidence", 1.0) < 0.5]

    final_answer = {
        "ruling": ruling,
        "ruling_summary": ruling_summary,
        "requirements": requirements,
        "confidence_level": confidence_level,
        "confidence_explanation": confidence_explanation,
        "regulation_excerpts": regulation_excerpts,
        "unresolved_conflicts": unresolved_conflicts,
        "low_confidence_sub_questions": low_conf_sqs,
        "caveats": caveats,
        "compliance_checklist": compliance_checklist,
        "all_citations": all_citations,
    }

    final_response = _build_query_response(final_answer, unresolved_conflicts)

    logger.info(
        "Final synthesis: ruling=%s, confidence=%s, citations=%d",
        ruling, confidence_level, len(all_citations),
    )
    return {"final_answer": final_answer, "final_response": final_response}


def _build_query_response(final_answer: dict, unresolved_conflicts: list[dict]) -> dict:
    """
    Map FinalAnswer to the QueryResponse shape expected by the API endpoints.
    Preserves all existing frontend-compatible fields.
    """
    ruling = final_answer.get("ruling", "CONDITIONAL")
    ruling_summary = final_answer.get("ruling_summary", "")
    requirements = final_answer.get("requirements", [])

    # Build answer text: ruling_summary + requirements list
    answer_lines = [f"**Ruling: {ruling}**\n\n{ruling_summary}"]
    if requirements:
        answer_lines.append("\n**Requirements:**")
        for req in requirements:
            item = req.get("item", "")
            citation = req.get("citation", "")
            answer_lines.append(f"- {item} {citation}".strip())
    if final_answer.get("compliance_checklist"):
        answer_lines.append("\n**Compliance Checklist:**")
        for step in final_answer["compliance_checklist"]:
            answer_lines.append(f"- {step}")

    answer = "\n".join(answer_lines)

    # Build citations list
    citations = []
    for cit_str in final_answer.get("all_citations", []):
        citations.append({
            "section": cit_str,
            "title": "",
            "text_snippet": "",
            "source_chunk_ids": [],
        })

    confidence_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}
    confidence_score = confidence_map.get(final_answer.get("confidence_level", "LOW"), 0.4)

    conflicts_detected = bool(unresolved_conflicts)
    conflict_details = [
        {
            "sections": c.get("conflicting_sections", []),
            "description": c.get("description", ""),
        }
        for c in unresolved_conflicts
    ]

    retrieved_sections = final_answer.get("all_citations", [])

    confidence_level = final_answer.get("confidence_level", "LOW")
    verification_passed = confidence_level in ("HIGH", "MEDIUM")

    return {
        "answer": answer,
        "citations": citations,
        "confidence_score": confidence_score,
        "conflicts_detected": conflicts_detected,
        "conflict_details": conflict_details,
        "disclaimer": DISCLAIMER,
        "retrieved_sections": retrieved_sections,
        "verification_passed": verification_passed,
    }
