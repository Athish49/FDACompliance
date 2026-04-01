"""Retriever agent — Qdrant hybrid search + cross-reference expansion (no LLM)."""

from __future__ import annotations

import logging
from typing import Optional

from agents.state import ComplianceState

logger = logging.getLogger(__name__)

# Lazy-loaded retriever singleton
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from retrieval.retriever import CFRRetriever
        _retriever = CFRRetriever()
    return _retriever


def _chunk_to_dict(result) -> dict:
    """Convert a SearchResult to a plain dict for state storage."""
    return {
        "chunk_id": result.chunk_id,
        "score": result.score,
        "reranker_score": result.reranker_score,
        "text": result.text,
        "cfr_citation": result.cfr_citation,
        "chunk_type": result.chunk_type,
        "section_preamble": result.section_preamble,
        "hierarchy": result.hierarchy,
        "defines": result.defines,
        "overflow_chunks": result.overflow_chunks,
        "metadata": result.metadata,
    }


def retriever_node(state: ComplianceState) -> ComplianceState:
    """Search Qdrant for each sub-question, deduplicate, expand cross-references."""
    from retrieval.retriever import SearchFilters

    retriever = _get_retriever()
    query = state["query"]
    sub_questions = state.get("sub_questions", [query])
    raw_filters = state.get("search_filters", {})

    # Build filters from planner output
    filters = SearchFilters(
        part_number=raw_filters.get("part_number"),
        section_number=raw_filters.get("section_number"),
        chapter_number=raw_filters.get("chapter_number"),
    )

    # ── Search for each sub-question ──────────────────────────────────────
    seen_ids: set[str] = set()
    retrieved: list[dict] = []

    for sq in sub_questions:
        results = retriever.search(query=sq, top_k=10, use_reranker=True, filters=filters)
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                retrieved.append(_chunk_to_dict(r))

    logger.info("Retriever: %d unique chunks from %d sub-questions",
                len(retrieved), len(sub_questions))

    # ── Cross-reference expansion (replaces Neo4j) ────────────────────────
    xref_sections: set[str] = set()
    for chunk in retrieved:
        xrefs = chunk.get("metadata", {}).get("cross_references_internal", [])
        for xref in xrefs:
            xref_sections.add(xref)

    cross_ref_chunks: list[dict] = []
    xref_seen: set[str] = set()

    for section_num in xref_sections:
        xref_filter = SearchFilters(section_number=section_num)
        results = retriever.search(
            query=query, top_k=3, use_reranker=False, filters=xref_filter,
        )
        for r in results:
            if r.chunk_id not in seen_ids and r.chunk_id not in xref_seen:
                xref_seen.add(r.chunk_id)
                cross_ref_chunks.append(_chunk_to_dict(r))

    logger.info("Retriever: %d cross-ref chunks from %d referenced sections",
                len(cross_ref_chunks), len(xref_sections))

    return {
        "retrieved_chunks": retrieved,
        "cross_ref_chunks": cross_ref_chunks,
    }
