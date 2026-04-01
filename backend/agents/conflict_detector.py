"""Conflict detector — cross-section conflict detection + final response assembly."""

from __future__ import annotations

import logging

from agents.llm import llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

CONFLICT_SYSTEM = """\
You are an FDA regulatory analyst checking for conflicts between CFR sections. Examine the retrieved regulatory text for:

1. General rules vs. specific exceptions that may apply
2. Conditional applicability (e.g., rules that only apply to certain product types)
3. Contradictory requirements between different sections or parts
4. Superseded or overridden provisions

Return ONLY valid JSON:
{
  "conflicts_detected": false,
  "conflict_flags": []
}

Or if conflicts exist:
{
  "conflicts_detected": true,
  "conflict_flags": [
    {"sections": ["21 CFR 101.13", "21 CFR 101.62"], "description": "explanation of conflict or exception"}
  ]
}

Only flag genuine conflicts or important exceptions — do not flag sections that simply cover different topics."""

DISCLAIMER = (
    "This information is for educational and informational purposes only. "
    "It does not constitute legal or regulatory advice. Always consult the "
    "official FDA regulations at ecfr.gov and qualified regulatory counsel "
    "for compliance decisions."
)


def conflict_detector_node(state: ComplianceState) -> ComplianceState:
    """Detect conflicts between retrieved sections and assemble the final response."""
    # ── Conflict detection ────────────────────────────────────────────────
    source_parts: list[str] = []
    for chunk in state.get("retrieved_chunks", [])[:8]:
        citation = chunk.get("cfr_citation", "")
        text = chunk.get("text", "")
        source_parts.append(f"[{citation}] {text[:500]}")

    conflicts_detected = False
    conflict_flags: list[dict] = []

    if len(source_parts) >= 2:
        messages = [
            {"role": "system", "content": CONFLICT_SYSTEM},
            {"role": "user", "content": "\n---\n".join(source_parts)},
        ]
        try:
            raw = llm_completion_json(messages, max_tokens=512)
            parsed = parse_llm_json(raw, messages)
            conflicts_detected = parsed.get("conflicts_detected", False)
            conflict_flags = parsed.get("conflict_flags", [])
        except (ValueError, RuntimeError) as exc:
            logger.warning("Conflict detection failed: %s", exc)

    if conflicts_detected:
        logger.info("Conflict detector: %d conflicts flagged", len(conflict_flags))
    else:
        logger.info("Conflict detector: no conflicts")

    # ── Assemble final response ───────────────────────────────────────────
    retrieved_sections: list[str] = []
    seen_sections: set[str] = set()
    for chunk in state.get("retrieved_chunks", []):
        cit = chunk.get("cfr_citation")
        if cit and cit not in seen_sections:
            seen_sections.add(cit)
            retrieved_sections.append(cit)

    final_response = {
        "answer": state.get("draft_answer", ""),
        "citations": state.get("citations", []),
        "confidence_score": state.get("confidence_score", 0.0),
        "conflicts_detected": conflicts_detected,
        "conflict_details": conflict_flags,
        "disclaimer": DISCLAIMER,
        "retrieved_sections": retrieved_sections,
        "verification_passed": state.get("verification_passed", False),
    }

    return {
        "conflicts_detected": conflicts_detected,
        "conflict_flags": conflict_flags,
        "final_response": final_response,
    }
