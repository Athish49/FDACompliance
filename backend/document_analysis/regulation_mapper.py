"""
Regulation Mapper — LangGraph node.

For each extracted claim, searches the CFR vector store to find the most
relevant regulatory sections, then writes claim_mappings to state.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Number of CFR chunks to retrieve per claim
_TOP_K = 5

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from retrieval.retriever import CFRRetriever
        from config import get_settings

        s = get_settings()
        from retrieval.retriever import RetrieverConfig

        _retriever = CFRRetriever(
            RetrieverConfig(
                qdrant_url=s.qdrant_url,
                qdrant_api_key=s.qdrant_api_key,
                collection_name=s.qdrant_collection,
            )
        )
    return _retriever


def _serialize_chunk(result) -> dict:
    """Convert a SearchResult into a plain dict for JSON-serialisable state."""
    return {
        "chunk_id": result.chunk_id,
        "score": result.score,
        "reranker_score": result.reranker_score,
        "text": result.text,
        "cfr_citation": result.cfr_citation,
        "chunk_type": result.chunk_type,
        "section_preamble": result.section_preamble,
        "defines": result.defines,
        "metadata": result.metadata,
    }


def regulation_mapper_node(state: dict) -> dict:
    """
    LangGraph node: for each extracted claim, retrieve relevant CFR chunks.

    Skips retrieval and passes through if an upstream error is already set,
    or if there are no claims to map.
    """
    if state.get("error"):
        return {}  # let graph route to report_builder

    claims = state.get("extracted_claims", [])
    if not claims:
        return {"claim_mappings": []}

    retriever = _get_retriever()
    mappings = []

    for claim in claims:
        claim_text = claim.get("claim_text", "")
        claim_type = claim.get("claim_type", "")
        if not claim_text:
            continue

        # Build a retrieval query combining claim type for specificity
        query = f"{claim_type}: {claim_text}" if claim_type else claim_text

        try:
            results = retriever.search(
                query=query,
                top_k=_TOP_K,
                use_reranker=True,
            )
            top_chunks = [_serialize_chunk(r) for r in results]
            # Use reranker score of top result as mapping confidence
            mapping_confidence = (
                results[0].reranker_score
                if results and results[0].reranker_score is not None
                else (results[0].score if results else 0.0)
            )
        except Exception as exc:
            logger.warning("Retrieval failed for claim '%s': %s", claim_text[:60], exc)
            top_chunks = []
            mapping_confidence = 0.0

        mappings.append(
            {
                "claim_text": claim_text,
                "claim_type": claim_type,
                "location_hint": claim.get("location_hint", ""),
                "top_chunks": top_chunks,
                "mapping_confidence": mapping_confidence,
            }
        )

    logger.info("Regulation mapping complete: %d claim(s) mapped", len(mappings))
    return {"claim_mappings": mappings}
