"""Stage 7 — Final Synthesis Agent.

Merges all resolved sub-answers into a single FinalAnswer and also populates
final_response (API-compatible QueryResponse shape) for the existing endpoints.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from agents.llm import llm_completion, llm_completion_json, parse_llm_json
from agents.session_logger import close_session, get_session
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This information is for educational purposes only and does not constitute legal advice. "
    "Always consult qualified regulatory counsel before making compliance decisions. "
    "Regulations change — verify against the current eCFR before relying on any information provided."
)

_RULING_SYSTEM = """\
You are an FDA regulatory compliance expert. Based on the provided sub-answers, issue a direct
ruling for the original user question.

PRIORITY: The PRIMARY sub-answer (highest retrieval confidence) is marked with [PRIMARY].
Rule primarily from it. Use SUPPORTING sub-answers only if their content is consistent and
non-contradictory with the primary.

REASONING STEP — apply only when the question involves a numeric threshold:
If the PRIMARY sub-answer retrieved the relevant numeric values (e.g. DRV, percentage threshold,
serving size limit), derive the answer mathematically and state the calculation in reasoning_steps.
Example: "Protein DRV = 50g. Excellent source threshold = 20%% DV = 10g. 20g > 10g → YES."
Only compute from values explicitly present in the sub-answers. Do NOT invent numbers.
If no calculation is needed, return reasoning_steps as an empty list.

Return ONLY valid JSON:
{
  "ruling": "YES" | "NO" | "CONDITIONAL",
  "ruling_summary": "1-2 sentence plain-language summary",
  "reasoning_steps": [],
  "requirements": [
    {"item": "specific requirement", "citation": "[§101.54]"}
  ],
  "compliance_checklist": [
    "Step 1: ...",
    "Step 2: ..."
  ]
}"""


_CONF_WEAK_THRESHOLD = 0.3  # Sub-answers below this are noise — excluded from confidence scoring


def _compute_confidence_level(sub_answers: list[dict]) -> tuple[str, str]:
    """Derive confidence_level and explanation using weighted confidence scoring.

    Sub-answers with confidence < 0.3 are excluded from the average — they add
    noise without contributing signal. If the one reliable sub-answer scores 0.6,
    the overall level is MEDIUM, not LOW (which a simple average would produce when
    the other two sub-answers scored 0.2 each).
    """
    if not sub_answers:
        return "LOW", "No sub-answers available."

    strong = [sa for sa in sub_answers if sa.get("confidence", 0.0) >= _CONF_WEAK_THRESHOLD]
    weak   = [sa for sa in sub_answers if sa.get("confidence", 0.0) <  _CONF_WEAK_THRESHOLD]

    # Score against strong sub-answers only; fall back to full pool if all are weak
    pool = strong if strong else sub_answers
    confidences = [sa.get("confidence", 0.0) for sa in pool]
    avg = sum(confidences) / len(confidences)

    if avg >= 0.8:
        level = "HIGH"
        explanation = f"Confidence {avg:.0%} across {len(pool)} reliable sub-question(s)."
    elif avg >= 0.5:
        level = "MEDIUM"
        explanation = f"Confidence {avg:.0%} across {len(pool)} reliable sub-question(s)."
    else:
        level = "LOW"
        explanation = f"Confidence {avg:.0%}."

    if weak and strong:
        explanation += (
            f" {len(weak)} of {len(sub_answers)} sub-question(s) had weak retrieval"
            f" (confidence < {_CONF_WEAK_THRESHOLD:.0%}) and were excluded from scoring."
        )

    low_sqs = [sa["sub_question_text"] for sa in sub_answers if sa.get("confidence", 1.0) < 0.5]
    if low_sqs:
        explanation += f" Low-confidence sub-questions: {'; '.join(low_sqs[:2])}."

    return level, explanation


def _select_primary_sub_answer(sub_answers: list[dict]) -> dict | None:
    """Return the sub-answer with the highest confidence — the LLM should rule from this."""
    if not sub_answers:
        return None
    return max(sub_answers, key=lambda sa: sa.get("confidence", 0.0))


def _build_partial_info_note(all_answers: list[dict], resolved_answers: list[dict]) -> str:
    """Return a human-readable note when the ruling is based on partial evidence, else ''."""
    total = len(all_answers)
    answered = len(resolved_answers)
    reliable = sum(1 for sa in resolved_answers if sa.get("confidence", 0.0) >= _CONF_WEAK_THRESHOLD)

    parts: list[str] = []
    if answered < total:
        parts.append(
            f"{total - answered} of {total} sub-question(s) found no supporting evidence"
        )
    if reliable < answered:
        parts.append(
            f"{answered - reliable} of {answered} answered sub-question(s) had low retrieval"
            f" confidence (< {_CONF_WEAK_THRESHOLD:.0%})"
        )
    if not parts:
        return ""
    return "; ".join(parts) + ". Ruling is based on partial information."


def _build_ruling_user_content(
    query: str,
    resolved_answers: list[dict],
    primary: dict | None,
    partial_info: str,
) -> str:
    """Build the user message for the ruling LLM, labelling the primary sub-answer."""
    lines = [f"Original question: {query}"]
    for sa in resolved_answers:
        is_primary = primary is not None and sa is primary
        label = "[PRIMARY — rule primarily from this]" if is_primary else "[SUPPORTING]"
        conf = sa.get("confidence", 0.0)
        lines.append(
            f"{label} Sub-question (confidence={conf:.2f}): {sa['sub_question_text']}\n"
            f"Answer: {sa.get('answer', '')[:500]}"
        )
    if partial_info:
        lines.append(f"Context note: {partial_info}")
    return "\n\n".join(lines)


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


_ROLE_PATTERNS = [
    ("threshold_requirement",  re.compile(r"\b\d+\s*(?:g|mg|mcg|%|percent|DV|iu|IU|oz)\b", re.I)),
    ("calculation_method",     re.compile(r"\b(?:calculat|formula|comput|divide|multiply|determin)\w*\b", re.I)),
    ("exemption",              re.compile(r"\b(?:exempt|except|does not apply|not required|exclusion)\b", re.I)),
    ("definition",             re.compile(r"\b(?:means|is defined as|shall mean|refers to)\b", re.I)),
    ("serving_size_context",   re.compile(r"\b(?:RACC|reference amount|serving size|per serving|edible portion)\b", re.I)),
]


def _classify_regulatory_role(text: str) -> str:
    """Rule-based classification of a regulation excerpt's role. No LLM involved."""
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(text):
            return role
    return "labeling_requirement"


def _build_regulation_excerpts(sub_answers: list[dict]) -> list[dict]:
    """Deduplicated list of regulation excerpts used across sub-answers."""
    seen: set[str] = set()
    excerpts: list[dict] = []
    for sa in sub_answers:
        for chunk in sa.get("chunks_used", []):
            cit = chunk.get("cfr_citation", "")
            if cit and cit not in seen:
                seen.add(cit)
                text = chunk.get("text", "")[:400]
                excerpts.append({
                    "cfr_section": cit,
                    "text": text,
                    "regulatory_role": _classify_regulatory_role(text),
                })
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


def _build_retrieval_quality(active_answers: list[dict]) -> str:
    """Classify the overall evidence retrieval quality.

    DIRECT_MATCH  — all sub-questions resolved with CORRECT CRAG verdict
    PARTIAL_MATCH — at least one CORRECT, some AMBIGUOUS or reformulations needed
    INFERRED      — no CORRECT verdicts; ruling is based on indirect evidence
    """
    if not active_answers:
        return "INFERRED"
    verdicts = [sa.get("crag_verdict", "") for sa in active_answers]
    reformulations = [len(sa.get("reformulation_log", [])) for sa in active_answers]
    all_correct = all(v == "CORRECT" for v in verdicts)
    any_correct = any(v == "CORRECT" for v in verdicts)
    any_reformulated = any(r > 0 for r in reformulations)
    if all_correct and not any_reformulated:
        return "DIRECT_MATCH"
    if any_correct:
        return "PARTIAL_MATCH"
    return "INFERRED"


def _build_regulatory_threshold(reasoning_steps: list[str]) -> dict | None:
    """Extract numeric threshold data from LLM reasoning steps without an extra LLM call."""
    if not reasoning_steps:
        return None
    joined = " ".join(reasoning_steps)
    num_match = re.search(r"(\d+(?:\.\d+)?)\s*(g|mg|mcg|%\s*DV|%|iu|oz|ml)\b", joined, re.I)
    if not num_match:
        return None
    value_str, unit = num_match.group(1), num_match.group(2).strip()
    try:
        value = float(value_str)
    except ValueError:
        return None
    return {
        "detected": True,
        "required_value": value,
        "required_unit": unit,
        "reasoning_chain": reasoning_steps,
    }


def _build_confidence_factors(
    active_answers: list[dict],
    compliance_checklist: list[str],
) -> dict:
    """Compute three user-facing confidence signals for the Analysis Report panel.

    CFR Match Strength   — best reranker score across all retrieved chunks; how
                           closely the regulation text matched the question.
    Regulation Coverage  — fraction of sub-questions answered directly from CFR
                           text (CRAG verdict CORRECT); 0 % means everything was
                           inferred or ambiguous.
    Compliance Readiness — depth of the compliance action plan produced; 3+ steps
                           = 100 %, scales linearly below that.
    """
    # CFR Match Strength: best cross-encoder score across all supporting chunks
    top_scores = []
    for sa in active_answers:
        for chunk in sa.get("chunks_used", []):
            score = chunk.get("reranker_score") or chunk.get("score")
            if score is not None:
                top_scores.append(float(score))
    cfr_match_strength = round(max(top_scores), 3) if top_scores else 0.0

    # Regulation Coverage: how many sub-questions found a direct CFR answer
    total_sqs = len(active_answers)
    correct_sqs = sum(1 for sa in active_answers if sa.get("crag_verdict") == "CORRECT")
    regulation_coverage = round(correct_sqs / total_sqs, 3) if total_sqs else 0.0

    # Compliance Readiness: smooth scale — 3+ checklist steps = fully ready
    compliance_readiness = round(min(len(compliance_checklist) / 3, 1.0), 3)

    return {
        "cfr_match_strength": cfr_match_strength,
        "regulation_coverage": regulation_coverage,
        "compliance_readiness": compliance_readiness,
    }


def _build_compliance_risk_flags(
    active_answers: list[dict],
    unresolved_conflicts: list[dict],
) -> list[dict]:
    """Derive risk flags from pipeline outputs — no LLM required."""
    flags: list[dict] = []

    total_claims = sum(len(sa.get("claim_verification", [])) for sa in active_answers)
    unverified = sum(len(sa.get("unverified_claims", [])) for sa in active_answers)
    if total_claims > 0 and unverified / total_claims > 0.25:
        flags.append({
            "type": "UNVERIFIED_CLAIMS",
            "severity": "warning",
            "description": (
                f"{unverified} of {total_claims} regulatory claim(s) could not be "
                "directly grounded in retrieved CFR text. Manual verification recommended."
            ),
        })

    ambiguous_count = sum(1 for sa in active_answers if sa.get("crag_verdict") == "AMBIGUOUS")
    if ambiguous_count:
        flags.append({
            "type": "AMBIGUOUS_REGULATORY_COVERAGE",
            "severity": "warning",
            "description": (
                f"{ambiguous_count} sub-question(s) returned ambiguous retrieval results. "
                "The CFR may contain overlapping or conditional requirements."
            ),
        })

    max_retries = max((len(sa.get("reformulation_log", [])) for sa in active_answers), default=0)
    if max_retries >= 2:
        flags.append({
            "type": "RETRIEVAL_DIFFICULTY",
            "severity": "info",
            "description": (
                f"Up to {max_retries} query reformulation(s) were needed to locate relevant "
                "CFR sections, which may indicate a novel or narrowly-defined question."
            ),
        })

    low_conf_count = sum(
        1 for sa in active_answers if sa.get("confidence", 1.0) < _CONF_WEAK_THRESHOLD
    )
    if low_conf_count:
        flags.append({
            "type": "LIMITED_CFR_COVERAGE",
            "severity": "warning",
            "description": (
                f"{low_conf_count} sub-question(s) had insufficient supporting evidence "
                "in the CFR database. The regulation may not explicitly address this scenario."
            ),
        })

    if unresolved_conflicts:
        flags.append({
            "type": "REGULATORY_CONFLICT",
            "severity": "warning",
            "description": (
                f"{len(unresolved_conflicts)} unresolved conflict(s) detected between "
                "CFR sections. Consult regulatory counsel to resolve."
            ),
        })

    return flags


def _build_sub_question_breakdown(active_answers: list[dict]) -> list[dict]:
    """Per-sub-question evidence summary for the UI breakdown panel."""
    breakdown = []
    for sa in active_answers:
        cited = list({c.get("cfr_citation", "") for c in sa.get("chunks_used", []) if c.get("cfr_citation")})
        claim_vf = sa.get("claim_verification", [])
        total_claims = len(claim_vf)
        verified_claims = sum(1 for cv in claim_vf if cv.get("is_supported"))
        breakdown.append({
            "topic": sa.get("sub_question_text", ""),
            "evidence_verdict": sa.get("crag_verdict", ""),
            "confidence": round(sa.get("confidence", 0.0), 3),
            "cited_sections": cited,
            "search_attempts": len(sa.get("reformulation_log", [])) + 1,
            "claims_verified": verified_claims,
            "claims_total": total_claims,
        })
    return breakdown


def _build_evidence_analysis(
    analyzed_query: dict,
    active_answers: list[dict],
    all_answers: list[dict],
    unresolved_conflicts: list[dict],
    reasoning_steps: list[str],
    compliance_checklist: list[str],
    pipeline_started_at: float | None,
) -> dict:
    """Assemble the full evidence_analysis composite field."""
    pipeline_duration_ms: int | None = None
    if pipeline_started_at:
        pipeline_duration_ms = int((time.time() - pipeline_started_at) * 1000)

    entities = analyzed_query.get("entities", {})
    query_classification = {
        "intent": analyzed_query.get("intent_type", "compliance_check"),
        "fda_domain": entities.get("fda_domain"),
        "product_type": entities.get("product_type"),
        "claim_type": entities.get("claim_type"),
        "is_multi_part": analyzed_query.get("is_multi_part", False),
    }

    total_claims = sum(len(sa.get("claim_verification", [])) for sa in active_answers)
    grounded = sum(
        1 for sa in active_answers
        for cv in sa.get("claim_verification", [])
        if cv.get("is_supported")
    )
    claim_verification_summary = {
        "total_extracted": total_claims,
        "grounded_in_cfr": grounded,
        "grounding_rate": round(grounded / total_claims, 3) if total_claims else 0.0,
    }

    unique_sections = len({
        c.get("cfr_citation", "")
        for sa in active_answers
        for c in sa.get("chunks_used", [])
        if c.get("cfr_citation")
    })
    cross_refs_followed = sum(len(sa.get("cross_refs_resolved", [])) for sa in active_answers)
    regulation_reach = {
        "unique_sections": unique_sections,
        "cross_references_followed": cross_refs_followed,
    }

    return {
        "retrieval_quality": _build_retrieval_quality(active_answers),
        "query_classification": query_classification,
        "pipeline_duration_ms": pipeline_duration_ms,
        "claim_verification": claim_verification_summary,
        "regulation_reach": regulation_reach,
        "sub_question_breakdown": _build_sub_question_breakdown(active_answers),
        "confidence_factors": _build_confidence_factors(active_answers, compliance_checklist),
        "regulatory_threshold": _build_regulatory_threshold(reasoning_steps),
        "compliance_steps_summary": {"total_steps": len(compliance_checklist)},
        "compliance_risk_flags": _build_compliance_risk_flags(active_answers, unresolved_conflicts),
    }


def final_synthesizer_node(state: ComplianceState) -> dict:
    """Assemble the FinalAnswer from resolved sub-answers and generate final_response."""
    query = state.get("query", "")
    all_answers = state.get("resolved_answers", state.get("sub_answers", []))
    unresolved_conflicts = state.get("unresolved_conflicts", [])
    domain_mismatches = state.get("domain_mismatches", [])
    analyzed_query = state.get("analyzed_query", {})
    pipeline_started_at = state.get("pipeline_started_at")

    # Exclude skipped sub-answers from synthesis — they have no evidence and
    # would drag down confidence averaging with noise 0.0 scores.
    resolved_answers = [sa for sa in all_answers if not sa.get("skipped")]
    skipped_count = len(all_answers) - len(resolved_answers)
    if skipped_count:
        logger.info(
            "[final_synthesizer] %d skipped sub-answer(s) excluded from synthesis",
            skipped_count,
        )

    if not resolved_answers:
        final_answer = {
            "ruling": "NO",
            "ruling_summary": "Insufficient information to answer the question.",
            "requirements": [],
            "confidence_level": "LOW",
            "confidence_explanation": "No sub-answers available.",
            "regulation_excerpts": [],
            "unresolved_conflicts": unresolved_conflicts,
            "domain_mismatches": domain_mismatches,
            "low_confidence_sub_questions": [],
            "caveats": [],
            "compliance_checklist": [],
            "all_citations": [],
        }
        final_response = _build_query_response(final_answer, unresolved_conflicts, domain_mismatches, {})
        return {"final_answer": final_answer, "final_response": final_response}

    # ── Ruling determination ────────────────────────────────────────────────────
    primary = _select_primary_sub_answer(resolved_answers)
    partial_info = _build_partial_info_note(all_answers, resolved_answers)

    if primary:
        logger.info(
            "[final_synthesizer] primary sub-answer: %s (confidence=%.2f)",
            primary.get("sub_question_id", "?"), primary.get("confidence", 0.0),
        )

    user_content = _build_ruling_user_content(query, resolved_answers, primary, partial_info)
    messages = [
        {"role": "system", "content": _RULING_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = llm_completion_json(messages, max_tokens=1200, temperature=0.1)
        ruling_result = parse_llm_json(raw, messages)
    except Exception as exc:
        logger.warning("Ruling LLM call failed: %s", exc)
        ruling_result = {
            "ruling": "CONDITIONAL",
            "ruling_summary": "See sub-answers for details.",
            "reasoning_steps": [],
            "requirements": [],
            "compliance_checklist": [],
        }

    ruling = ruling_result.get("ruling", "CONDITIONAL")
    ruling_summary = ruling_result.get("ruling_summary", "")
    reasoning_steps = ruling_result.get("reasoning_steps", [])
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
        "reasoning_steps": reasoning_steps,
        "partial_information_note": partial_info or None,
        "requirements": requirements,
        "confidence_level": confidence_level,
        "confidence_explanation": confidence_explanation,
        "regulation_excerpts": regulation_excerpts,
        "unresolved_conflicts": unresolved_conflicts,
        "domain_mismatches": domain_mismatches,
        "low_confidence_sub_questions": low_conf_sqs,
        "caveats": caveats,
        "compliance_checklist": compliance_checklist,
        "all_citations": all_citations,
    }

    evidence_analysis = _build_evidence_analysis(
        analyzed_query=analyzed_query,
        active_answers=resolved_answers,
        all_answers=all_answers,
        unresolved_conflicts=unresolved_conflicts,
        reasoning_steps=reasoning_steps,
        compliance_checklist=compliance_checklist,
        pipeline_started_at=pipeline_started_at,
    )

    final_response = _build_query_response(
        final_answer, unresolved_conflicts, domain_mismatches, evidence_analysis
    )

    logger.info(
        "[final_synthesizer] ruling=%s | confidence=%s | citations=%d | sub_questions=%d | duration_ms=%s",
        ruling, confidence_level, len(all_citations), len(resolved_answers),
        evidence_analysis.get("pipeline_duration_ms"),
    )

    session_id = state.get("session_id", "")
    if session_id:
        sl = get_session(session_id)
        if sl:
            sl.log_final_synthesis(
                ruling=ruling,
                confidence_level=confidence_level,
                citations_count=len(all_citations),
                low_confidence_sub_questions=low_conf_sqs,
            )
        close_session(session_id)

    return {"final_answer": final_answer, "final_response": final_response}


def _build_query_response(
    final_answer: dict,
    unresolved_conflicts: list[dict],
    domain_mismatches: list[dict] | None = None,
    evidence_analysis: dict | None = None,
) -> dict:
    """
    Map FinalAnswer to the QueryResponse shape expected by the API endpoints.
    Preserves all existing frontend-compatible fields.
    """
    ruling = final_answer.get("ruling", "CONDITIONAL")
    ruling_summary = final_answer.get("ruling_summary", "")
    requirements = final_answer.get("requirements", [])

    # Build answer text: ruling_summary + optional reasoning + requirements
    answer_lines = [f"**Ruling: {ruling}**\n\n{ruling_summary}"]
    reasoning_steps = final_answer.get("reasoning_steps", [])
    if reasoning_steps:
        answer_lines.append("\n**Reasoning:**")
        for step in reasoning_steps:
            answer_lines.append(f"- {step}")
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
    partial_note = final_answer.get("partial_information_note")
    if partial_note:
        answer_lines.append(f"\n*{partial_note}*")

    answer = "\n".join(answer_lines)

    # Build citations list, enriched with text snippets from regulation_excerpts
    excerpt_map = {
        e["cfr_section"]: e["text"]
        for e in final_answer.get("regulation_excerpts", [])
        if e.get("cfr_section")
    }
    citations = []
    for cit_str in final_answer.get("all_citations", []):
        citations.append({
            "section": cit_str,
            "title": "",
            "text_snippet": excerpt_map.get(cit_str, ""),
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

    structured_answer = {
        "ruling": final_answer.get("ruling", "CONDITIONAL"),
        "ruling_summary": final_answer.get("ruling_summary", ""),
        "requirements": final_answer.get("requirements", []),
        "compliance_checklist": final_answer.get("compliance_checklist", []),
        "confidence_explanation": final_answer.get("confidence_explanation", ""),
        "confidence_level": confidence_level,
        "regulation_excerpts": final_answer.get("regulation_excerpts", []),
        "caveats": final_answer.get("caveats", []),
        "reasoning_steps": final_answer.get("reasoning_steps", []),
        "low_confidence_sub_questions": final_answer.get("low_confidence_sub_questions", []),
        "partial_information_note": final_answer.get("partial_information_note"),
    }

    return {
        "answer": answer,
        "citations": citations,
        "confidence_score": confidence_score,
        "conflicts_detected": conflicts_detected,
        "conflict_details": conflict_details,
        "domain_mismatches": domain_mismatches or [],
        "disclaimer": DISCLAIMER,
        "retrieved_sections": retrieved_sections,
        "verification_passed": verification_passed,
        "structured_answer": structured_answer,
        "evidence_analysis": evidence_analysis or {},
    }
