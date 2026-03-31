"""
CFR Chunk Embedder → Qdrant
============================
Reads the chunks JSON produced by extractor.py, generates dense embeddings
for each chunk, and upserts them into a Qdrant collection.

Each Qdrant point stores:
  - id       : SHA-256 of chunk_id (UUID-compatible hex)
  - vector   : embedding of the chunk text
  - payload  : full chunk dict (all metadata fields for filtered retrieval)

The payload intentionally mirrors the chunk schema so that downstream
retrieval can filter by cfr_citation, part number, effective_date, etc.

Dependencies (add to requirements.txt):
  qdrant-client>=1.9.0
  sentence-transformers>=3.0.0   # or openai>=1.0.0 for OpenAI embeddings
  tqdm>=4.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

    # Embedding model
    # Use "sentence-transformers" or "openai"
    embedding_backend: str = "sentence-transformers"
    # Sentence-Transformers model (good balance of quality / speed for regulatory text)
    st_model_name: str = "BAAI/bge-large-en-v1.5"
    # OpenAI model (alternative — set embedding_backend = "openai")
    openai_model_name: str = "text-embedding-3-large"
    openai_api_key: Optional[str] = None

    # Vector dimension — must match the chosen model:
    #   BAAI/bge-large-en-v1.5  → 1024
    #   text-embedding-3-large  → 3072
    vector_size: int = 1024

    # Processing
    batch_size: int = 64           # chunks per embedding batch
    upsert_batch_size: int = 128   # points per Qdrant upsert call

    # Text construction
    # When True, the section_preamble (if present) is prepended to chunk text
    # before embedding so each vector is self-contained.
    prepend_preamble: bool = True




# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk_id to a UUID-like hex string via SHA-256."""
    digest = hashlib.sha256(chunk_id.encode()).hexdigest()
    # Format as UUID: 8-4-4-4-12
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _build_embed_text(chunk: dict, config: EmbedderConfig) -> str:
    """
    Construct the text string that will be embedded for a chunk.
    Prepends hierarchy breadcrumb, section_preamble, and overlap tail
    so each vector carries enough context to be self-sufficient.
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

    # Section preamble (govering condition for multi-paragraph sections)
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
# Embedding backends
# ─────────────────────────────────────────────────────────────────────────────

def _load_st_model(model_name: str):
    """Lazy-load a SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc
    logger.info("Loading SentenceTransformer model: %s", model_name)
    return SentenceTransformer(model_name)


def _embed_st(model, texts: list[str], config: EmbedderConfig) -> list[list[float]]:
    """Embed a batch of texts using SentenceTransformer."""
    vectors = model.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity compatible
    )
    return [v.tolist() for v in vectors]


def _embed_openai(texts: list[str], config: EmbedderConfig) -> list[list[float]]:
    """Embed texts using OpenAI embeddings API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required. Install with: pip install openai") from exc

    client = OpenAI(api_key=config.openai_api_key)
    response = client.embeddings.create(input=texts, model=config.openai_model_name)
    return [item.embedding for item in response.data]


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant collection setup
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_collection(client, config: EmbedderConfig) -> None:
    """Create Qdrant collection if it does not already exist."""
    try:
        from qdrant_client.models import Distance, VectorParams
    except ImportError as exc:
        raise ImportError("qdrant-client is required. Install with: pip install qdrant-client") from exc

    existing = [c.name for c in client.get_collections().collections]
    if config.collection_name not in existing:
        logger.info("Creating Qdrant collection '%s'", config.collection_name)
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(size=config.vector_size, distance=Distance.COSINE),
        )
    else:
        logger.info("Using existing Qdrant collection '%s'", config.collection_name)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks_path: str | Path = "backend/data/chunks/cfr_chunks.json",
    config: Optional[EmbedderConfig] = None,
) -> dict:
    """
    Load chunks from *chunks_path*, embed them, and upsert into Qdrant.

    Returns a summary dict with keys: total_upserted, collection_name, duration_seconds.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
    except ImportError as exc:
        raise ImportError("qdrant-client is required. Install with: pip install qdrant-client") from exc

    if config is None:
        config = EmbedderConfig()

    chunks = _load_chunks(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # Connect to Qdrant
    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    _ensure_collection(client, config)

    # Load embedding model
    model = None
    if config.embedding_backend == "sentence-transformers":
        model = _load_st_model(config.st_model_name)

    start = time.time()
    total_upserted = 0

    for batch in _batched(chunks, config.batch_size):
        texts = [_build_embed_text(c, config) for c in batch]

        # Generate embeddings
        if config.embedding_backend == "sentence-transformers":
            vectors = _embed_st(model, texts, config)
        elif config.embedding_backend == "openai":
            vectors = _embed_openai(texts, config)
        else:
            raise ValueError(f"Unknown embedding_backend: {config.embedding_backend!r}")

        # Build Qdrant points — payload contains the full chunk dict
        points = [
            PointStruct(
                id=_chunk_id_to_uuid(chunk["chunk_id"]),
                vector=vector,
                payload={
                    # Flattened metadata fields for Qdrant filtered search
                    "chunk_id": chunk["chunk_id"],
                    "chunk_type": chunk.get("chunk_type"),
                    "cfr_citation": chunk.get("cfr_citation"),
                    "title_number": chunk.get("hierarchy", {}).get("title", {}).get("number"),
                    "chapter_number": chunk.get("hierarchy", {}).get("chapter", {}).get("number"),
                    "part_number": (chunk.get("hierarchy", {}).get("part") or {}).get("number"),
                    "subpart_letter": (chunk.get("hierarchy", {}).get("subpart") or {}).get("letter"),
                    "section_number": (chunk.get("hierarchy", {}).get("section") or {}).get("number"),
                    "source_file": chunk.get("source_file"),
                    # Full chunk for retrieval context
                    "text": chunk.get("text"),
                    "section_preamble": chunk.get("section_preamble"),
                    "defines": chunk.get("defines"),
                    "cross_references_internal": chunk.get("cross_references_internal", []),
                    "is_overflow_chunk": chunk.get("is_overflow_chunk", False),
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]

        # Upsert to Qdrant in sub-batches
        for upsert_batch in _batched(points, config.upsert_batch_size):
            client.upsert(collection_name=config.collection_name, points=upsert_batch)
            total_upserted += len(upsert_batch)

        logger.debug("Upserted %d / %d", total_upserted, len(chunks))

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
