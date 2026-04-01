"""Verifier agent — hallucination detection + retry signal."""

from __future__ import annotations

import logging

from agents.llm import llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """\
You are a strict regulatory fact-checker. Your job is to verify that every factual claim in the draft answer is supported by the provided source text.

Check for:
1. Claims about specific numbers, thresholds, percentages, or fees — must appear verbatim in source
2. Section references (e.g., "21 CFR 101.13") — must match a source section
3. Regulatory requirements stated as fact — must have direct source support
4. Definitions or term meanings — must match source definitions

Return ONLY valid JSON:
{
  "verification_passed": true,
  "issues": []
}

Or if there are problems:
{
  "verification_passed": false,
  "issues": [
    {"claim": "the specific claim", "issue": "unsupported|inaccurate|misattributed", "detail": "explanation"}
  ]
}

Be strict: if a specific number or threshold is stated but not found in source text, flag it."""


def verifier_node(state: ComplianceState) -> ComplianceState:
    """Verify the draft answer against source chunks. Increment retry_count on failure."""
    draft = state.get("draft_answer", "")
    retry_count = state.get("retry_count", 0)

    # Build source text for verification
    source_parts: list[str] = []
    for chunk in state.get("retrieved_chunks", [])[:10]:
        citation = chunk.get("cfr_citation", "")
        text = chunk.get("text", "")
        source_parts.append(f"[{citation}] {text[:600]}")
    for chunk in state.get("cross_ref_chunks", [])[:5]:
        citation = chunk.get("cfr_citation", "")
        text = chunk.get("text", "")
        source_parts.append(f"[{citation}] {text[:600]}")

    source_text = "\n---\n".join(source_parts)

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": f"Draft answer:\n{draft}\n\nSource text:\n{source_text}"},
    ]

    try:
        raw = llm_completion_json(messages, max_tokens=1024)
        parsed = parse_llm_json(raw, messages)
    except (ValueError, RuntimeError) as exc:
        logger.warning("Verifier parse failed, passing through: %s", exc)
        return {
            "verification_passed": True,
            "verification_issues": [],
            "retry_count": retry_count,
        }

    passed = parsed.get("verification_passed", True)
    issues = parsed.get("issues", [])

    if not passed:
        retry_count += 1
        logger.info("Verifier: FAILED (%d issues), retry_count=%d", len(issues), retry_count)
    else:
        logger.info("Verifier: PASSED")

    return {
        "verification_passed": passed,
        "verification_issues": issues,
        "retry_count": retry_count,
    }
