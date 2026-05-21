"""Stage 6 — Consistency & Conflict Detection Agent.

Cross-examines claims across all sub-answers, identifies contradictions,
attempts automatic resolution by section specificity (recency resolution is
not implemented — effective_date is not present in the payload schema).
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from agents.llm import llm_completion_json, parse_llm_json
from agents.session_logger import get_session
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

# Sub-answers below this confidence have no reliable claims worth checking.
_CONF_THRESHOLD = 0.3

# 21 CFR Part number ranges → broad FDA domain.
# Used for domain coherence check — cross-domain citations are a retrieval
# failure signal, not a regulatory conflict.
_PART_RANGES: list[tuple[int, int, str]] = [
    (1,    99,   "administrative"),
    (100,  199,  "food"),
    (200,  499,  "drug"),
    (500,  599,  "animal"),
    (600,  699,  "biological"),
    (700,  799,  "cosmetic"),
    (800,  999,  "device"),
    (1000, 1099, "device"),
    (1100, 1199, "tobacco"),
    (1200, 1299, "administrative"),
]

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


def _part_number_from_section(section: str) -> int | None:
    """Parse the CFR Part number from a section string like '§101.9' or '21 CFR 101.54'."""
    cleaned = section.replace("21 CFR", "").replace("§", "").strip()
    m = re.match(r"(\d+)", cleaned)
    return int(m.group(1)) if m else None


def _domain_from_part(part: int) -> str | None:
    for lo, hi, domain in _PART_RANGES:
        if lo <= part <= hi:
            return domain
    return None


def _dominant_domain(sub_answer: dict) -> str | None:
    """Return the most-cited FDA domain for a sub-answer, or None if undetermined."""
    domains: list[str] = []
    for sec in _sections_from_sub_answer(sub_answer):
        part = _part_number_from_section(sec)
        if part is not None:
            d = _domain_from_part(part)
            if d:
                domains.append(d)
    if not domains:
        return None
    return Counter(domains).most_common(1)[0][0]


def _domains_compatible(d1: str | None, d2: str | None) -> bool:
    """Return True if the two domains can legitimately share regulatory claims."""
    if d1 is None or d2 is None:
        return True  # undetermined — don't block contradiction check
    if d1 == d2:
        return True
    # Administrative parts (1-99, 1200-1299) apply across all domains
    if "administrative" in (d1, d2):
        return True
    return False


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

    # ── Tier 1: Skipped sub-answers (Layer 6) — no evidence, pass through ──────
    active = [sa for sa in sub_answers if not sa.get("skipped")]
    skipped = [sa for sa in sub_answers if sa.get("skipped")]

    if skipped:
        logger.info(
            "[consistency] %d sub-answer(s) skipped (no evidence) — excluded from conflict check",
            len(skipped),
        )

    # ── Tier 2: Low-confidence sub-answers — no reliable claims worth checking ──
    eligible = [sa for sa in active if sa.get("confidence", 1.0) >= _CONF_THRESHOLD]
    low_conf_excluded = [sa for sa in active if sa.get("confidence", 1.0) < _CONF_THRESHOLD]

    if low_conf_excluded:
        for sa in low_conf_excluded:
            sa.setdefault("flags", [])
            sa["flags"].append("LOW_CONF_EXCLUDED_FROM_CONFLICT_CHECK")
        logger.info(
            "[consistency] %d sub-answer(s) below confidence threshold (%.2f) — excluded from conflict check",
            len(low_conf_excluded), _CONF_THRESHOLD,
        )

    if len(eligible) <= 1:
        return {
            "resolved_answers": eligible + low_conf_excluded + skipped,
            "unresolved_conflicts": [],
            "domain_mismatches": [],
        }

    unresolved_conflicts: list[dict] = []
    domain_mismatches: list[dict] = []
    resolved_answers = [sa.copy() for sa in eligible]

    # ── Step 1: Domain coherence pre-check ────────────────────────────────────
    # Cross-domain citation pairs (food vs. drug, device vs. biological, etc.)
    # indicate a retrieval failure — the LLM pulled chunks from the wrong
    # regulatory domain. Flag them as retrieval_domain_mismatch and skip the
    # contradiction LLM calls for that pair to avoid false conflict reports.
    domain_mismatch_pairs: set[tuple[int, int]] = set()

    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            sa_i = eligible[i]
            sa_j = eligible[j]
            d_i = _dominant_domain(sa_i)
            d_j = _dominant_domain(sa_j)

            if not _domains_compatible(d_i, d_j):
                domain_mismatch_pairs.add((i, j))
                mismatch = {
                    "sub_question_ids": [sa_i["sub_question_id"], sa_j["sub_question_id"]],
                    "domain_a": d_i,
                    "domain_b": d_j,
                    "description": (
                        f"Sub-answer {sa_i['sub_question_id']} cites {d_i} regulations "
                        f"while {sa_j['sub_question_id']} cites {d_j} regulations — "
                        "likely a retrieval domain error, not a regulatory conflict."
                    ),
                }
                domain_mismatches.append(mismatch)
                # Flag the sub-answers themselves so the synthesizer can note it
                for idx, sa in [(i, sa_i), (j, sa_j)]:
                    resolved_answers[idx].setdefault("flags", [])
                    if "RETRIEVAL_DOMAIN_MISMATCH" not in resolved_answers[idx]["flags"]:
                        resolved_answers[idx]["flags"].append("RETRIEVAL_DOMAIN_MISMATCH")

    if domain_mismatches:
        logger.warning(
            "[consistency] %d domain mismatch pair(s) detected — skipping contradiction check for those pairs",
            len(domain_mismatches),
        )

    # ── Step 2+3: Cross-examine claims for compatible pairs ───────────────────
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            if (i, j) in domain_mismatch_pairs:
                continue  # retrieval failure — contradiction check not meaningful

            sa_i = eligible[i]
            sa_j = eligible[j]

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

                    # ── Step 4: Resolution ────────────────────────────────────
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
        "[consistency] %d conflict(s) found, %d unresolved, %d domain mismatch(es)",
        len(unresolved_conflicts), len(truly_unresolved), len(domain_mismatches),
    )

    session_id = state.get("session_id", "")
    if session_id:
        sl = get_session(session_id)
        if sl:
            sl.log_consistency(len(unresolved_conflicts), len(truly_unresolved))

    return {
        "resolved_answers": resolved_answers + low_conf_excluded + skipped,
        "unresolved_conflicts": truly_unresolved,
        "domain_mismatches": domain_mismatches,
    }
