"""
Violation Classifier — LangGraph node.

For each claim mapping, uses an LLM to classify whether the claim complies
with the retrieved CFR sections. Also performs a single pass over the full
document to detect missing required labeling elements (gap analysis).

Writes violations to state.
"""

from __future__ import annotations

import json
import logging

from agents.llm import llm_completion_json, parse_llm_json

logger = logging.getLogger(__name__)

_CLASSIFY_SYSTEM = """\
You are an FDA regulatory compliance expert. You will be given a label claim and the \
most relevant CFR regulatory sections. Determine whether the claim complies with the regulations.

Output a JSON object with:
  - compliant: true | false
  - violation_type: one of [missing, incorrect, prohibited, insufficient_disclosure, compliant]
  - severity: one of [critical, high, medium, low, none]
  - explanation: concise explanation (1-3 sentences) citing specific CFR sections
  - cfr_citation: the primary CFR citation (e.g. "21 CFR 101.9") or null

Rules for severity:
  - critical: false health/disease claim, prohibited ingredient, safety risk
  - high: missing mandatory labeling element, grossly misleading claim
  - medium: incorrect format, insufficient disclosure, ambiguous claim
  - low: minor technical deviation that is unlikely to mislead consumers
  - none: claim is compliant

If the regulatory sections provided are not relevant to the claim, say compliant = true \
with violation_type = compliant and a note that no applicable regulation was found.
"""

_GAP_SYSTEM = """\
You are an FDA regulatory compliance expert specializing in food labeling requirements \
under 21 CFR Part 101. Given the full text of a product label, identify any REQUIRED \
labeling elements that are MISSING or INCOMPLETE.

Required elements to check (non-exhaustive):
  - Statement of identity (common/usual name)
  - Net quantity of contents
  - Nutrition Facts panel (21 CFR 101.9)
  - Ingredient list (21 CFR 101.4)
  - Allergen declaration (FALCPA / 21 CFR 101.4(d))
  - Name and place of business of manufacturer/packer/distributor (21 CFR 101.5)
  - Country of origin (if applicable)

Output a JSON object with key "missing_elements": a list of objects, each with:
  - element: name of the required element
  - severity: one of [critical, high, medium, low]
  - explanation: why it is required and what is missing
  - cfr_citation: relevant CFR citation or null

Return {"missing_elements": []} if all required elements appear to be present.
"""


def _classify_single_claim(mapping: dict) -> dict | None:
    """
    Run LLM classification for one claim mapping.
    Returns a violation dict, or None if the claim is fully compliant.
    """
    claim_text = mapping.get("claim_text", "")
    claim_type = mapping.get("claim_type", "")
    top_chunks = mapping.get("top_chunks", [])

    # Build context string from top CFR chunks
    context_parts = []
    for chunk in top_chunks[:3]:  # limit to 3 chunks to stay within token budget
        citation = chunk.get("cfr_citation") or ""
        text = chunk.get("text") or ""
        if text:
            context_parts.append(f"[{citation}]\n{text[:600]}")
    context = "\n\n".join(context_parts) if context_parts else "No relevant CFR sections found."

    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"CLAIM TYPE: {claim_type}\n"
                f"CLAIM TEXT: {claim_text}\n\n"
                f"RELEVANT CFR SECTIONS:\n{context}"
            ),
        },
    ]

    try:
        raw = llm_completion_json(messages, max_tokens=512)
        parsed = parse_llm_json(raw, messages)
    except Exception as exc:
        logger.warning("LLM classification failed for claim '%s': %s", claim_text[:60], exc)
        return None

    if parsed.get("compliant") or parsed.get("violation_type") == "compliant":
        return None  # no violation

    return {
        "claim_text": claim_text,
        "claim_type": claim_type,
        "location_hint": mapping.get("location_hint", ""),
        "cfr_citation": parsed.get("cfr_citation"),
        "violation_type": parsed.get("violation_type", "incorrect"),
        "severity": parsed.get("severity", "medium"),
        "explanation": parsed.get("explanation", ""),
        "relevant_cfr_text": top_chunks[0].get("text", "")[:400] if top_chunks else "",
    }


def _check_missing_elements(document_text: str) -> list[dict]:
    """Single LLM call to detect missing required labeling elements (gap analysis)."""
    prompt_text = document_text[:8_000]
    messages = [
        {"role": "system", "content": _GAP_SYSTEM},
        {
            "role": "user",
            "content": f"Analyze this product label for missing required elements:\n\n{prompt_text}",
        },
    ]

    try:
        raw = llm_completion_json(messages, max_tokens=1024)
        parsed = parse_llm_json(raw, messages)
        missing = parsed.get("missing_elements", [])
        return missing if isinstance(missing, list) else []
    except Exception as exc:
        logger.warning("Gap analysis LLM call failed: %s", exc)
        return []


def violation_classifier_node(state: dict) -> dict:
    """
    LangGraph node: classify violations for each claim mapping + gap analysis.
    """
    if state.get("error"):
        return {}

    claim_mappings = state.get("claim_mappings", [])
    document_text = state.get("document_text", "")
    violations = []

    # Per-claim compliance classification
    for mapping in claim_mappings:
        result = _classify_single_claim(mapping)
        if result is not None:
            violations.append(result)

    # Gap analysis — missing required elements
    if document_text:
        missing_elements = _check_missing_elements(document_text)
        for elem in missing_elements:
            violations.append(
                {
                    "claim_text": f"[Missing element] {elem.get('element', 'Unknown')}",
                    "claim_type": "missing_required_element",
                    "location_hint": "entire document",
                    "cfr_citation": elem.get("cfr_citation"),
                    "violation_type": "missing",
                    "severity": elem.get("severity", "high"),
                    "explanation": elem.get("explanation", ""),
                    "relevant_cfr_text": "",
                }
            )

    logger.info(
        "Violation classification complete: %d violation(s) identified", len(violations)
    )
    return {"violations": violations}
