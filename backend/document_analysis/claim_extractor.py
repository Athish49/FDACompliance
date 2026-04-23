"""
Claim Extractor — LangGraph node.

Reads document_text from state, uses an LLM to extract structured label claims
that may be subject to FDA/CFR regulation, and writes extracted_claims to state.
"""

from __future__ import annotations

import json
import logging

from agents.llm import llm_completion_json, parse_llm_json

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an FDA regulatory expert. Your task is to extract all label claims and assertions \
from a food/drug product label document that could be subject to FDA/CFR regulation.

For each claim, output:
  - claim_text: the exact or near-exact text of the claim
  - claim_type: one of [ingredient, nutrition_fact, health_claim, structure_function_claim,
      disease_claim, net_weight, allergen, serving_size, storage_instruction,
      manufacturer_info, expiration, other]
  - location_hint: brief description of where in the document this appears (e.g. "front panel", "ingredient list")

Return a JSON object with key "claims" containing a list of these objects.
Extract ALL regulatory-relevant claims, not just the ones that seem problematic.
If the document has no extractable label claims, return {"claims": []}.
"""


def claim_extractor_node(state: dict) -> dict:
    """
    LangGraph node: extract label claims from document_text.
    On LLM failure, sets error and returns empty claims so the graph can short-circuit.
    """
    doc_text = state.get("document_text", "")
    if not doc_text:
        return {"error": "document_text is empty", "extracted_claims": []}

    # Truncate context sent to LLM if very long (LLM has context limits)
    prompt_text = doc_text[:12_000]

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Extract all regulatory-relevant label claims from the following document.\n\n"
                f"DOCUMENT:\n{prompt_text}"
            ),
        },
    ]

    try:
        raw = llm_completion_json(messages, max_tokens=2048)
        parsed = parse_llm_json(raw, messages)
        claims = parsed.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        logger.info("Extracted %d claims from document", len(claims))
        return {"extracted_claims": claims}
    except Exception as exc:
        logger.exception("claim_extractor_node failed: %s", exc)
        return {"error": f"Claim extraction failed: {exc}", "extracted_claims": []}
