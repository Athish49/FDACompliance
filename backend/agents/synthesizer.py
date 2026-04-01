"""Synthesizer agent — generate grounded answer with inline CFR citations."""

from __future__ import annotations

import logging

from agents.llm import llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM = """\
You are an FDA regulatory compliance expert. Using ONLY the provided CFR source text, answer the user's question with precise, grounded citations.

Rules:
1. Every factual claim MUST be supported by the source text below
2. Use inline citations like [21 CFR 101.13] referencing the specific section
3. If the source text does not contain enough information, say so explicitly
4. Do NOT invent or hallucinate any regulatory requirements
5. Use plain language but maintain regulatory accuracy

Return ONLY valid JSON:
{
  "answer": "Your detailed answer with [21 CFR X.Y] inline citations...",
  "citations": [
    {"section": "21 CFR 101.13", "title": "Nutrient content claims", "text_snippet": "relevant quote from source"}
  ],
  "confidence_score": 0.85
}

Confidence score guidelines:
- 0.9-1.0: Direct, specific answer fully supported by source text
- 0.7-0.89: Good answer but some aspects inferred or partially covered
- 0.5-0.69: Partial answer; significant gaps in source coverage
- Below 0.5: Mostly uncertain; limited source support"""


def _build_context(state: ComplianceState) -> str:
    """Assemble context block from definitions, primary chunks, and cross-refs."""
    parts: list[str] = []

    # Definitions
    definitions = state.get("definitions_resolved", {})
    if definitions:
        parts.append("=== REGULATORY DEFINITIONS ===")
        for term, defn in definitions.items():
            parts.append(f"• {term}: {defn[:500]}")
        parts.append("")

    # Primary retrieved chunks (top 8)
    retrieved = state.get("retrieved_chunks", [])[:8]
    if retrieved:
        parts.append("=== PRIMARY SOURCE SECTIONS ===")
        for i, chunk in enumerate(retrieved, 1):
            citation = chunk.get("cfr_citation", "Unknown section")
            preamble = chunk.get("section_preamble", "")
            text = chunk.get("text", "")
            overflow_text = ""
            for oc in chunk.get("overflow_chunks", []):
                overflow_text += " " + oc.get("text", "")
            full_text = f"{preamble} {text}{overflow_text}".strip()
            parts.append(f"[{i}] {citation}\n{full_text[:800]}")
        parts.append("")

    # Cross-reference chunks (top 4)
    xref = state.get("cross_ref_chunks", [])[:4]
    if xref:
        parts.append("=== CROSS-REFERENCED SECTIONS ===")
        for i, chunk in enumerate(xref, 1):
            citation = chunk.get("cfr_citation", "Unknown section")
            text = chunk.get("text", "")
            parts.append(f"[X{i}] {citation}\n{text[:600]}")

    return "\n".join(parts)


def synthesizer_node(state: ComplianceState) -> ComplianceState:
    """Generate a grounded answer with citations from retrieved context."""
    query = state["query"]
    context = _build_context(state)

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {"role": "user", "content": f"Question: {query}\n\n{context}"},
    ]

    try:
        raw = llm_completion_json(messages, max_tokens=2048)
        parsed = parse_llm_json(raw, messages)
    except (ValueError, RuntimeError) as exc:
        logger.error("Synthesizer failed: %s", exc)
        return {
            "draft_answer": f"I was unable to generate an answer due to a processing error: {exc}",
            "citations": [],
            "confidence_score": 0.0,
        }

    answer = parsed.get("answer", "No answer generated.")
    citations = parsed.get("citations", [])
    confidence = float(parsed.get("confidence_score", 0.5))

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    logger.info("Synthesizer: %d citations, confidence=%.2f", len(citations), confidence)

    return {
        "draft_answer": answer,
        "citations": citations,
        "confidence_score": confidence,
    }
