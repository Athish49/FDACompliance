"""Stage 6 — Consistency & Conflict Detection Agent.

Cross-examines claims across all sub-answers, identifies contradictions,
attempts automatic resolution by section specificity (recency resolution is
not implemented — effective_date is not present in the payload schema).
"""
from __future__ import annotations

import logging

from agents.llm import llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

_CONFLICT_CHECK_SYSTEM = """\
You are an FDA regulatory expert.  You will be given two regulatory claims from different
sub-answers.  Determine whether they contradict each other.

Return ONLY valid JSON:
{"contradicts": true, "explanation": "brief reason"}
or
{"contradicts": false, "explanation": ""}"""

_SPECIFICITY_SYSTEM = """\
Given two conflicting CFR sections, which is more specific (lower-level, narrower scope)?
Return ONLY valid JSON:
{"winner": "101.54", "reason": "More specific than Part 101 general rule"}"""


def _sections_from_sub_answer(sub_answer: dict) -> list[str]:
    """Extract unique CFR section strings from a sub-answer."""
    sections = set()
    for cit in sub_answer.get("citations", []):
        sec = cit.get("cfr_section", "")
        if sec:
            sections.add(sec)
    for chunk in sub_answer.get("chunks_used", []):
        cit = chunk.get("cfr_citation", "")
        if cit:
            sections.add(cit)
    return list(sections)


def _check_contradiction(claim_a: str, claim_b: str) -> tuple[bool, str]:
    """Ask LLM whether two claims contradict. Returns (contradicts, explanation)."""
    messages = [
        {"role": "system", "content": _CONFLICT_CHECK_SYSTEM},
        {"role": "user", "content": f"Claim A: {claim_a}\nClaim B: {claim_b}"},
    ]
    try:
        raw = llm_completion_json(messages, max_tokens=256, temperature=0.1)
        result = parse_llm_json(raw, messages)
        return bool(result.get("contradicts", False)), result.get("explanation", "")
    except Exception as exc:
        logger.debug("Contradiction check failed: %s", exc)
        return False, ""


def _resolve_by_specificity(sec_a: str, sec_b: str) -> tuple[str | None, str]:
    """Attempt to resolve conflict by section specificity. Returns (winner, reason)."""
    # Heuristic: longer section number string → more specific (e.g. "101.54" vs "101")
    def specificity(s: str) -> int:
        parts = s.replace("21 CFR ", "").replace("§", "").strip().split(".")
        return len(parts) * 10 + len(s)

    if not sec_a and not sec_b:
        return None, ""
    if not sec_b:
        return sec_a, "Only one section available"
    if not sec_a:
        return sec_b, "Only one section available"

    if specificity(sec_a) > specificity(sec_b):
        return sec_a, f"{sec_a} is more specific than {sec_b}"
    if specificity(sec_b) > specificity(sec_a):
        return sec_b, f"{sec_b} is more specific than {sec_a}"

    # Equal specificity — ask LLM
    messages = [
        {"role": "system", "content": _SPECIFICITY_SYSTEM},
        {"role": "user", "content": f"Section A: {sec_a}\nSection B: {sec_b}"},
    ]
    try:
        raw = llm_completion_json(messages, max_tokens=128, temperature=0.1)
        result = parse_llm_json(raw, messages)
        winner = result.get("winner")
        reason = result.get("reason", "")
        if winner in (sec_a, sec_b):
            return winner, reason
    except Exception:
        pass
    return None, ""


def consistency_detector_node(state: ComplianceState) -> dict:
    """Detect and attempt to resolve conflicts across sub-answers."""
    sub_answers = state.get("sub_answers", [])

    if len(sub_answers) <= 1:
        return {
            "resolved_answers": sub_answers,
            "unresolved_conflicts": [],
        }

    unresolved_conflicts: list[dict] = []
    resolved_answers = [sa.copy() for sa in sub_answers]

    # ── Step 1+2: Cross-examine claims across all sub-answer pairs ─────────────
    for i in range(len(sub_answers)):
        for j in range(i + 1, len(sub_answers)):
            sa_i = sub_answers[i]
            sa_j = sub_answers[j]

            claims_i = [cv["claim_text"] for cv in sa_i.get("claim_verification", []) if cv.get("is_supported")]
            claims_j = [cv["claim_text"] for cv in sa_j.get("claim_verification", []) if cv.get("is_supported")]

            if not claims_i or not claims_j:
                continue

            # Sample claim pairs to keep LLM calls manageable
            sample_i = claims_i[:3]
            sample_j = claims_j[:3]

            for ci in sample_i:
                for cj in sample_j:
                    contradicts, explanation = _check_contradiction(ci, cj)
                    if not contradicts:
                        continue

                    secs_i = _sections_from_sub_answer(sa_i)
                    secs_j = _sections_from_sub_answer(sa_j)

                    sec_a = secs_i[0] if secs_i else ""
                    sec_b = secs_j[0] if secs_j else ""

                    # ── Step 3: Resolution ────────────────────────────────────
                    winner, resolution_reason = _resolve_by_specificity(sec_a, sec_b)
                    resolved = winner is not None

                    if resolved:
                        loser_idx = i if winner == sec_b else j
                        resolved_answers[loser_idx].setdefault("flags", [])
                        resolved_answers[loser_idx]["flags"].append("CONFLICT_RESOLVED")
                        resolved_answers[loser_idx].setdefault("caveats", [])
                        resolved_answers[loser_idx]["caveats"].append(
                            f"Overridden by {winner}: {resolution_reason}"
                        )

                    conflict = {
                        "sub_question_ids": [sa_i["sub_question_id"], sa_j["sub_question_id"]],
                        "conflicting_sections": [sec_a, sec_b],
                        "description": explanation,
                        "resolved": resolved,
                        "resolution": resolution_reason if resolved else None,
                        "winning_section": winner,
                    }
                    unresolved_conflicts.append(conflict)

    truly_unresolved = [c for c in unresolved_conflicts if not c["resolved"]]

    logger.info(
        "Consistency: %d conflicts found, %d unresolved",
        len(unresolved_conflicts), len(truly_unresolved),
    )

    return {
        "resolved_answers": resolved_answers,
        "unresolved_conflicts": truly_unresolved,
    }
