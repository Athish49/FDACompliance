"""
CFR Hybrid Retriever — BGE-M3 Dense + Sparse + Reranker
=========================================================
Performs hybrid search over the Qdrant "cfr_chunks" collection using
BGE-M3 dense and sparse vectors, fuses results with Reciprocal Rank
Fusion (RRF), and optionally reranks with bge-reranker-v2-m3.

Usage:
    retriever = CFRRetriever()
    results = retriever.search("food labeling requirements for sodium")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrieverConfig:
    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "cfr_chunks"

    # BGE-M3 encoder
    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True

    # Search parameters
    dense_top_k: int = 60
    sparse_top_k: int = 60
    rrf_k: int = 60
    final_top_k: int = 20

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 10

    # Overflow expansion
    expand_overflow: bool = True
    max_overflow_hops: int = 5


@dataclass
class SearchFilters:
    """Optional filters narrowing the Qdrant search scope."""
    part_number: Optional[str] = None
    chapter_number: Optional[str] = None
    subpart_letter: Optional[str] = None
    section_number: Optional[str] = None
    chunk_type: Optional[str] = None
    source_file: Optional[str] = None


@dataclass
class SearchResult:
    """A single result returned from hybrid search."""
    chunk_id: str
    score: float
    reranker_score: Optional[float]
    text: str
    cfr_citation: Optional[str]
    chunk_type: Optional[str]
    section_preamble: Optional[str]
    hierarchy: dict
    defines: Optional[str]
    overflow_chunks: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

class CFRRetriever:
    """Hybrid dense+sparse retriever with cross-encoder reranking."""

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
        self._model = None
        self._reranker = None
        self._client = None

    # ── Lazy loaders ──────────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
            )
        return self._client

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            logger.info("Loading BGE-M3 for retrieval: %s", self.config.model_name)
            self._model = BGEM3FlagModel(self.config.model_name, use_fp16=self.config.use_fp16)
        return self._model

    @property
    def reranker(self):
        if self._reranker is None:
            from FlagEmbedding import FlagReranker
            logger.info("Loading reranker: %s", self.config.reranker_model)
            self._reranker = FlagReranker(
                self.config.reranker_model,
                use_fp16=self.config.use_fp16,
            )
        return self._reranker

    # ── Query encoding ────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> tuple[list[float], dict]:
        """Encode query with BGE-M3 → (dense_vec, sparse_dict)."""
        output = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vec = output["dense_vecs"][0].tolist()
        weights = output["lexical_weights"][0]
        sparse_dict = {
            "indices": [int(i) for i in sorted(weights.keys())],
            "values": [float(weights[i]) for i in sorted(weights.keys())],
        }
        return dense_vec, sparse_dict

    # ── Qdrant filter builder ─────────────────────────────────────────────

    @staticmethod
    def _build_qdrant_filter(filters: Optional[SearchFilters]):
        """Convert SearchFilters to a Qdrant Filter object (or None)."""
        if filters is None:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        field_map = {
            "part_number": filters.part_number,
            "chapter_number": filters.chapter_number,
            "subpart_letter": filters.subpart_letter,
            "section_number": filters.section_number,
            "chunk_type": filters.chunk_type,
            "source_file": filters.source_file,
        }
        for field_name, value in field_map.items():
            if value is not None:
                conditions.append(
                    FieldCondition(key=field_name, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None

    # ── Search methods ────────────────────────────────────────────────────

    def _search_dense(
        self, dense_vec: list[float], qdrant_filter, top_k: int
    ) -> list[tuple[str, float, dict]]:
        """Dense vector search. Returns [(point_id, score, payload), ...]."""
        from qdrant_client.models import NamedVector

        results = self.client.query_points(
            collection_name=self.config.collection_name,
            query=dense_vec,
            using="dense",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            (point.id, point.score, point.payload)
            for point in results.points
        ]

    def _search_sparse(
        self, sparse_dict: dict, qdrant_filter, top_k: int
    ) -> list[tuple[str, float, dict]]:
        """Sparse vector search. Returns [(point_id, score, payload), ...]."""
        from qdrant_client.models import SparseVector

        results = self.client.query_points(
            collection_name=self.config.collection_name,
            query=SparseVector(
                indices=sparse_dict["indices"],
                values=sparse_dict["values"],
            ),
            using="sparse",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            (point.id, point.score, point.payload)
            for point in results.points
        ]

    # ── RRF fusion ────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        dense_results: list[tuple[str, float, dict]],
        sparse_results: list[tuple[str, float, dict]],
        k: int = 60,
        top_k: int = 20,
    ) -> list[tuple[str, float, dict]]:
        """
        Reciprocal Rank Fusion over two result lists.
        score(d) = sum(1 / (k + rank + 1)) across lists where d appears.
        """
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, (pid, _score, payload) in enumerate(dense_results):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            payloads[pid] = payload

        for rank, (pid, _score, payload) in enumerate(sparse_results):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            payloads[pid] = payload

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(pid, score, payloads[pid]) for pid, score in ranked]

    # ── Reranker ──────────────────────────────────────────────────────────

    def _rerank(
        self, query: str, candidates: list[tuple[str, float, dict]], top_k: int
    ) -> list[tuple[str, float, float, dict]]:
        """
        Cross-encoder reranking.
        Returns [(point_id, rrf_score, reranker_score, payload), ...] sorted by reranker_score.
        """
        if not candidates:
            return []

        pairs = [[query, cand[2].get("text", "")] for cand in candidates]
        reranker_scores = self.reranker.compute_score(pairs, normalize=True)

        # compute_score returns a single float for single pair, list otherwise
        if isinstance(reranker_scores, float):
            reranker_scores = [reranker_scores]

        scored = [
            (cand[0], cand[1], float(rs), cand[2])
            for cand, rs in zip(candidates, reranker_scores)
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    # ── Overflow expansion ────────────────────────────────────────────────

    def _expand_overflow(self, payload: dict) -> list[dict]:
        """Follow overflow_sequence.next_chunk_id chain (max N hops)."""
        overflow_chunks = []
        seq = payload.get("overflow_sequence")
        if not seq or not isinstance(seq, dict):
            return overflow_chunks

        next_id = seq.get("next_chunk_id")
        for _ in range(self.config.max_overflow_hops):
            if not next_id:
                break
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            results = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="chunk_id", match=MatchValue(value=next_id))]
                ),
                limit=1,
                with_payload=True,
            )
            points, _ = results
            if not points:
                break
            p = points[0].payload
            overflow_chunks.append({
                "chunk_id": p.get("chunk_id"),
                "text": p.get("text"),
            })
            next_seq = p.get("overflow_sequence")
            next_id = next_seq.get("next_chunk_id") if isinstance(next_seq, dict) else None

        return overflow_chunks

    # ── Main search orchestrator ──────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        filters: Optional[SearchFilters] = None,
    ) -> list[SearchResult]:
        """
        Full hybrid search pipeline:
        1. Encode query (dense + sparse)
        2. Parallel dense and sparse search
        3. RRF fusion
        4. Cross-encoder reranking (optional)
        5. Overflow expansion (optional)
        """
        final_top_k = top_k or self.config.final_top_k
        qdrant_filter = self._build_qdrant_filter(filters)

        # 1. Encode
        dense_vec, sparse_dict = self._encode_query(query)

        # 2. Search both indexes
        dense_results = self._search_dense(
            dense_vec, qdrant_filter, self.config.dense_top_k
        )
        sparse_results = self._search_sparse(
            sparse_dict, qdrant_filter, self.config.sparse_top_k
        )

        # 3. RRF fusion
        rrf_top_k = final_top_k if not use_reranker else self.config.final_top_k
        fused = self._rrf_fuse(
            dense_results, sparse_results,
            k=self.config.rrf_k,
            top_k=rrf_top_k,
        )

        # 4. Rerank
        if use_reranker:
            reranked = self._rerank(
                query, fused, top_k=min(final_top_k, self.config.reranker_top_k)
            )
        else:
            reranked = [(pid, score, None, payload) for pid, score, payload in fused]

        # 5. Build results with overflow expansion
        results = []
        for pid, rrf_score, reranker_score, payload in reranked:
            overflow = []
            if self.config.expand_overflow:
                overflow = self._expand_overflow(payload)

            results.append(SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                score=rrf_score,
                reranker_score=reranker_score,
                text=payload.get("text", ""),
                cfr_citation=payload.get("cfr_citation"),
                chunk_type=payload.get("chunk_type"),
                section_preamble=payload.get("section_preamble"),
                hierarchy=payload.get("hierarchy", {}),
                defines=payload.get("defines"),
                overflow_chunks=overflow,
                metadata={
                    "part_number": payload.get("part_number"),
                    "chapter_number": payload.get("chapter_number"),
                    "section_number": payload.get("section_number"),
                    "source_file": payload.get("source_file"),
                    "cross_references_internal": payload.get("cross_references_internal", []),
                    "paragraph_labels": payload.get("paragraph_labels", []),
                    "metrics": payload.get("metrics", []),
                },
            ))

        return results

    # ── Single chunk fetch ────────────────────────────────────────────────

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        """Fetch a single chunk by its chunk_id field. Returns payload or None."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results = self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
            ),
            limit=1,
            with_payload=True,
        )
        points, _ = results
        if not points:
            return None
        return points[0].payload
