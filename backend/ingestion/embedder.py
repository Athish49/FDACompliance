"""
CFR Chunk Embedder → Qdrant (BGE-M3 Hybrid Dense + Sparse)
============================================================
Reads the chunks JSON produced by extractor.py, generates dense (1024-d)
and sparse (lexical-weight) embeddings using BAAI/bge-m3, and upserts them
into a single Qdrant collection with named vectors.

Each Qdrant point stores:
  - id       : SHA-256 of chunk_id (UUID-compatible hex)
  - vectors  : {"dense": [...], "sparse": SparseVector(...)}
  - payload  : flattened metadata + full text for filtered retrieval

Dependencies:
  FlagEmbedding>=1.2.10
  qdrant-client>=1.9.0
  tqdm>=4.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbedderConfig:
    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "cfr_chunks"

    # BGE-M3 model
    model_name: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    use_fp16: bool = True

    # Processing
    batch_size: int = 32

    # Text construction
    prepend_preamble: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk_id to a UUID-like hex string via SHA-256."""
    digest = hashlib.sha256(chunk_id.encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _build_embed_text(chunk: dict, config: EmbedderConfig) -> str:
    """
    Construct the text string that will be embedded for a chunk.
    Prepends hierarchy breadcrumb and section_preamble so each vector
    carries enough context to be self-sufficient.
    """
    parts: list[str] = []

    # Breadcrumb prefix (improves retrieval specificity)
    h = chunk.get("hierarchy", {})
    breadcrumb_parts = []
    for level in ("title", "chapter", "subchapter", "part", "subpart", "section"):
        node = h.get(level)
        if node and isinstance(node, dict):
            name = node.get("name") or node.get("number") or ""
            number = node.get("number") or ""
            label = f"{level.capitalize()} {number}: {name}".strip(": ")
            breadcrumb_parts.append(label)
    if breadcrumb_parts:
        parts.append(" > ".join(breadcrumb_parts))

    # Section preamble (governing condition for multi-paragraph sections)
    if config.prepend_preamble:
        preamble = chunk.get("section_preamble") or ""
        if preamble:
            parts.append(preamble)

    # Main chunk text
    parts.append(chunk.get("text", ""))

    return " ".join(parts)


def _load_chunks(chunks_path: str | Path) -> list[dict]:
    with open(chunks_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["chunks"]


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ─────────────────────────────────────────────────────────────────────────────
# BGE-M3 model loading & encoding
# ─────────────────────────────────────────────────────────────────────────────

_bgem3_model = None


def _load_bgem3_model(config: EmbedderConfig):
    """Lazy-load BGEM3FlagModel (cached across calls)."""
    global _bgem3_model
    if _bgem3_model is not None:
        return _bgem3_model
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise ImportError(
            "FlagEmbedding is required. Install with: pip install FlagEmbedding"
        ) from exc
    logger.info("Loading BGE-M3 model: %s (fp16=%s)", config.model_name, config.use_fp16)
    _bgem3_model = BGEM3FlagModel(config.model_name, use_fp16=config.use_fp16)
    return _bgem3_model


def _encode_batch(model, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
    """
    Encode a batch of texts with BGE-M3.

    Returns:
        (dense_vectors, sparse_vectors)
        - dense_vectors: list of 1024-d float lists
        - sparse_vectors: list of {"indices": [...], "values": [...]} dicts
    """
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vecs = output["dense_vecs"]
    lexical_weights = output["lexical_weights"]

    dense_list = [vec.tolist() for vec in dense_vecs]

    sparse_list = []
    for weights in lexical_weights:
        indices = sorted(weights.keys())
        values = [float(weights[idx]) for idx in indices]
        sparse_list.append({"indices": [int(i) for i in indices], "values": values})

    return dense_list, sparse_list


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant collection setup
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_collection(client, config: EmbedderConfig) -> None:
    """Create Qdrant collection with named dense + sparse vectors if it doesn't exist."""
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        SparseVectorParams,
        VectorParams,
    )

    existing = [c.name for c in client.get_collections().collections]
    if config.collection_name in existing:
        logger.info("Using existing Qdrant collection '%s'", config.collection_name)
        return

    logger.info("Creating Qdrant collection '%s'", config.collection_name)
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config={
            "dense": VectorParams(size=config.dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )

    # Create payload indexes for filtered search
    indexed_fields = {
        "chunk_id": PayloadSchemaType.KEYWORD,
        "part_number": PayloadSchemaType.KEYWORD,
        "chapter_number": PayloadSchemaType.KEYWORD,
        "chunk_type": PayloadSchemaType.KEYWORD,
        "cfr_citation": PayloadSchemaType.TEXT,
        "section_number": PayloadSchemaType.KEYWORD,
        "defines": PayloadSchemaType.KEYWORD,
        "is_overflow_chunk": PayloadSchemaType.BOOL,
        "source_file": PayloadSchemaType.KEYWORD,
    }
    for field_name, schema_type in indexed_fields.items():
        client.create_payload_index(
            collection_name=config.collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
    logger.info("Created %d payload indexes", len(indexed_fields))


# ─────────────────────────────────────────────────────────────────────────────
# Payload builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_payload(chunk: dict) -> dict:
    """Flatten chunk hierarchy into top-level fields for Qdrant filtering."""
    h = chunk.get("hierarchy", {})
    return {
        # Flattened metadata for filtered search
        "chunk_id": chunk["chunk_id"],
        "chunk_type": chunk.get("chunk_type"),
        "cfr_citation": chunk.get("cfr_citation"),
        "title_number": (h.get("title") or {}).get("number"),
        "chapter_number": (h.get("chapter") or {}).get("number"),
        "part_number": (h.get("part") or {}).get("number"),
        "subpart_letter": (h.get("subpart") or {}).get("letter"),
        "section_number": (h.get("section") or {}).get("number"),
        "source_file": chunk.get("source_file"),
        "defines": chunk.get("defines"),
        "is_overflow_chunk": chunk.get("is_overflow_chunk", False),
        # Full text and context for retrieval
        "text": chunk.get("text"),
        "section_preamble": chunk.get("section_preamble"),
        "cross_references_internal": chunk.get("cross_references_internal", []),
        "hierarchy": h,
        "paragraph_labels": chunk.get("paragraph_labels", []),
        "metrics": chunk.get("metrics", []),
        "overflow_sequence": chunk.get("overflow_sequence"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks_path: str | Path = "backend/data/chunks/cfr_chunks.json",
    config: Optional[EmbedderConfig] = None,
) -> dict:
    """
    Load chunks from *chunks_path*, embed with BGE-M3 (dense + sparse),
    and upsert into Qdrant.

    Returns a summary dict with keys: total_upserted, collection_name, duration_seconds.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    if config is None:
        config = EmbedderConfig()

    chunks = _load_chunks(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # Connect to Qdrant
    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    _ensure_collection(client, config)

    # Load BGE-M3
    model = _load_bgem3_model(config)

    start = time.time()
    total_upserted = 0
    batches = list(_batched(chunks, config.batch_size))

    for batch in tqdm(batches, desc="Embedding batches", unit="batch"):
        texts = [_build_embed_text(c, config) for c in batch]

        # Encode with BGE-M3 → dense + sparse
        dense_vecs, sparse_vecs = _encode_batch(model, texts)

        # Build Qdrant points with named vectors
        points = [
            PointStruct(
                id=_chunk_id_to_uuid(chunk["chunk_id"]),
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"],
                    ),
                },
                payload=_build_payload(chunk),
            )
            for chunk, dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs)
        ]

        client.upsert(collection_name=config.collection_name, points=points)
        total_upserted += len(points)

    duration = round(time.time() - start, 2)
    logger.info(
        "Embedding complete: %d points upserted to '%s' in %.1fs",
        total_upserted,
        config.collection_name,
        duration,
    )
    return {
        "total_upserted": total_upserted,
        "collection_name": config.collection_name,
        "duration_seconds": duration,
    }
