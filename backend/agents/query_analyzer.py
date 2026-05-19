"""Stage 1 — Query Analysis Agent.

Parses the raw question into a structured AnalyzedQuery.  Detects ambiguity
before decomposition begins so the pipeline can request clarification early.
"""
from __future__ import annotations

import logging

from agents.llm import llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an FDA regulatory compliance expert. Analyze the user's compliance question and return ONLY valid JSON:

{
  "intent_type": "compliance_check" | "definition" | "procedure" | "penalty",
  "entities": {
    "product_type": "<product or null>",
    "claim_type": "<label claim type or null>",
    "fda_domain": "<FDA domain (food/drug/device/cosmetic) or null>",
    "explicit_refs": ["101.54", "21 CFR 101.9"]
  },
  "is_multi_part": false,
  "needs_clarification": false,
  "clarification_question": null
}

Guidelines:
- intent_type "compliance_check": asking whether something is allowed/required
- intent_type "definition": asking what a regulatory term means
- intent_type "procedure": asking how to do something
- intent_type "penalty": asking about enforcement, fines, violations
- explicit_refs: ONLY if the user typed a CFR section/part number (e.g. "21 CFR 101.54")
- needs_clarification: true ONLY when the question is genuinely too vague (missing product type,
  claim context, and regulatory domain — all three). Be conservative; most questions can proceed.
- clarification_question: specific targeted question when needs_clarification is true, else null"""


def query_analyzer_node(state: ComplianceState) -> dict:
    """Classify intent, extract entities, detect ambiguity."""
    query = state["query"]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": query},
    ]
    raw = llm_completion_json(messages, max_tokens=512, temperature=0.1)
    try:
        result = parse_llm_json(raw, messages)
    except ValueError as exc:
        logger.warning("Query analyzer parse failed: %s", exc)
        result = {}

    result.setdefault("intent_type", "compliance_check")
    result.setdefault("entities", {})
    result["entities"].setdefault("product_type", None)
    result["entities"].setdefault("claim_type", None)
    result["entities"].setdefault("fda_domain", None)
    result["entities"].setdefault("explicit_refs", [])
    result.setdefault("is_multi_part", False)
    result.setdefault("needs_clarification", False)
    result.setdefault("clarification_question", None)

    logger.info(
        "Query analysis: intent=%s, multi_part=%s, clarification=%s",
        result["intent_type"], result["is_multi_part"], result["needs_clarification"],
    )

    return {
        "analyzed_query": result,
        "needs_clarification": result["needs_clarification"],
        "clarification_question": result["clarification_question"],
    }
