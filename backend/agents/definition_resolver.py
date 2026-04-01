"""Definition resolver — identify regulated terms (LLM) + look up definitions (Qdrant)."""

from __future__ import annotations

import logging

from agents.llm import llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

TERM_EXTRACTION_SYSTEM = """\
You are an FDA regulatory expert. Given retrieved CFR text chunks and a user question, identify regulated terms that need formal definitions to properly answer the question.

Only list terms that:
1. Have specific regulatory definitions in 21 CFR (e.g., "nutrient content claim", "dietary supplement", "misbranded")
2. Are important for understanding the answer

Return ONLY valid JSON:
{"terms": ["term1", "term2"]}

If no terms need formal definitions, return: {"terms": []}"""


def _get_retriever():
    from agents.retriever_node import _get_retriever
    return _get_retriever()


def definition_resolver_node(state: ComplianceState) -> ComplianceState:
    """Identify regulated terms via LLM, then look up definitions in Qdrant."""
    from retrieval.retriever import SearchFilters

    query = state["query"]
    retrieved = state.get("retrieved_chunks", [])

    # ── Phase 1: Identify terms needing definitions ───────────────────────
    context_snippets = "\n---\n".join(
        c.get("text", "")[:300] for c in retrieved[:8]
    )

    messages = [
        {"role": "system", "content": TERM_EXTRACTION_SYSTEM},
        {"role": "user", "content": f"Question: {query}\n\nRetrieved CFR text:\n{context_snippets}"},
    ]

    try:
        raw = llm_completion_json(messages, max_tokens=256)
        parsed = parse_llm_json(raw, messages)
        terms = parsed.get("terms", [])
    except (ValueError, RuntimeError) as exc:
        logger.warning("Term extraction failed: %s", exc)
        terms = []

    logger.info("Definition resolver: %d terms identified", len(terms))

    # ── Phase 2: Look up each term in Qdrant ──────────────────────────────
    retriever = _get_retriever()
    definitions: dict[str, str] = {}
    definition_chunks: list[dict] = []

    for term in terms[:10]:  # cap to avoid excessive lookups
        def_filter = SearchFilters(chunk_type="definition")
        results = retriever.search(
            query=f"definition of {term}",
            top_k=3,
            use_reranker=False,
            filters=def_filter,
        )

        # Try exact match on the 'defines' field first
        matched = None
        for r in results:
            if r.defines and r.defines.lower() == term.lower():
                matched = r
                break

        # Fallback: best scoring result
        if not matched and results:
            matched = results[0]

        if matched:
            definitions[term] = matched.text
            definition_chunks.append({
                "chunk_id": matched.chunk_id,
                "term": term,
                "text": matched.text,
                "cfr_citation": matched.cfr_citation,
            })
            logger.debug("Resolved definition: %s → %s", term, matched.cfr_citation)
        else:
            logger.debug("No definition found for: %s", term)

    logger.info("Definition resolver: %d/%d terms resolved", len(definitions), len(terms))

    return {
        "definitions_resolved": definitions,
        "definition_chunks": definition_chunks,
    }
