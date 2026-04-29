"""
CFR Chunk Embedder → Qdrant (BGE-M3 Hybrid Dense + Sparse)
============================================================
Reads the chunks JSON produced by extractor.py, generates dense (1024-d)
and sparse (lexical-weight) embeddings using BAAI/bge-m3, and upserts them
into a single Qdrant collection with named vectors.

Reliability guarantees
-----------------------
* wait=True on every upsert — each batch is confirmed stored before the
  next batch is encoded.
* Per-batch retry with exponential backoff (default 3 retries, base 2 s).
  A single failed batch does not abort the run; failures are tracked.
  If the very first batch fails all retries, we abort early (dead server).
* Content-hash skip — the SHA-256 of the embed text is stored in the
  payload. On subsequent runs, chunks whose hash hasn't changed are not
  re-embedded or re-uploaded (saves minutes of model inference).
* Orphan cleanup — after all upserts, any Qdrant point whose chunk_id is
  no longer present in the source is deleted in batches of 500.
  This is opt-in (`cleanup_orphans=True`) and is only enabled by the
  pipeline when the extraction step also ran (fresh source data).

Deduplication strategy (industry standard for RAG corpora)
-----------------------------------------------------------
* Point IDs are SHA-256(chunk_id) — deterministic, so re-running always
  targets the same Qdrant point (no phantom duplicates).
* Upsert is idempotent: same ID → overwrite, new ID → insert.
* Content-hash comparison avoids unnecessary overwrites.
* Orphan cleanup removes stale points from previously-seen CFR revisions.

Each Qdrant point stores:
  - id            : SHA-256 of chunk_id (UUID-compatible hex)
  - vectors       : {"cfr-dense": [...], "cfr-sparse": SparseVector(...)}
  - payload       : flattened metadata + full text + content_hash

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
    collection_name: str = "FDAComplianceAI"

    # BGE-M3 model
    model_name: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    use_fp16: bool = True

    # Processing
    batch_size: int = 64

    # Retry
    max_retries: int = 3
    retry_backoff_base: float = 2.0   # seconds; actual wait = base^attempt

    # Text construction
    prepend_preamble: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — IDs and text
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a string chunk_id to a UUID-like hex string via SHA-256."""
    digest = hashlib.sha256(chunk_id.encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _content_hash(embed_text: str) -> str:
    """SHA-256 of the embed text — used to detect unchanged chunks."""
    return hashlib.sha256(embed_text.encode()).hexdigest()


def _build_embed_text(chunk: dict, config: EmbedderConfig) -> str:
    """
    Construct the text string that will be embedded for a chunk.
    Prepends hierarchy breadcrumb and section_preamble so each vector
    carries enough context to be self-sufficient.

    NOTE: The content_hash must be derived from this function's output,
    not from the raw chunk text, because the vector represents this text.
    """
    parts: list[str] = []

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

    if config.prepend_preamble:
        preamble = chunk.get("section_preamble") or ""
        if preamble:
            parts.append(preamble)

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
        import transformers.utils
        import transformers.utils.import_utils
        if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):
            transformers.utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
        if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
            transformers.utils.import_utils.is_torch_fx_available = lambda: False
    except ImportError:
        pass

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise ImportError(
            "FlagEmbedding is required. Install with: pip install FlagEmbedding"
        ) from exc

    import torch

    if torch.backends.mps.is_available():
        device = "mps"
        device_label = "Apple GPU (MPS)"
    elif torch.cuda.is_available():
        device = "cuda"
        device_label = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_label = "CPU"

    print(f"\n[EMBED] *** Inference device: {device_label} ***\n", flush=True)
    logger.info("Loading BGE-M3 model: %s  device=%s  fp16=%s", config.model_name, device, config.use_fp16)
    _bgem3_model = BGEM3FlagModel(config.model_name, use_fp16=config.use_fp16, device=device)
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

    dense_list = [vec.tolist() for vec in output["dense_vecs"]]

    sparse_list = []
    for weights in output["lexical_weights"]:
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
        HnswConfigDiff,
        PayloadSchemaType,
        SparseIndexParams,
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
            "cfr-dense": VectorParams(
                size=config.dense_dim,
                distance=Distance.COSINE,
                on_disk=False,
                hnsw_config=HnswConfigDiff(m=24, payload_m=24, ef_construct=256),
            ),
        },
        sparse_vectors_config={
            "cfr-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=True),
            ),
        },
    )

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

def _build_payload(chunk: dict, embed_text: str) -> dict:
    """Flatten chunk hierarchy into top-level fields for Qdrant filtering."""
    h = chunk.get("hierarchy", {})
    return {
        # Filterable metadata
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
        # Full text and context
        "text": chunk.get("text"),
        "section_preamble": chunk.get("section_preamble"),
        "cross_references_internal": chunk.get("cross_references_internal", []),
        "hierarchy": h,
        "paragraph_labels": chunk.get("paragraph_labels", []),
        "metrics": chunk.get("metrics", []),
        "overflow_sequence": chunk.get("overflow_sequence"),
        # Change-detection fingerprint (SHA-256 of the embedded text)
        "content_hash": _content_hash(embed_text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Upsert with retry
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_with_retry(
    client,
    collection_name: str,
    points: list,
    max_retries: int,
    backoff_base: float,
    batch_index: int,
) -> bool:
    """
    Upsert a batch of points into Qdrant with wait=True.

    Retries up to max_retries times with exponential backoff on any exception.
    Returns True on success, False if all retries were exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,   # block until the batch is indexed on the server
            )
            if attempt > 0:
                logger.info(
                    "[EMBED] Batch %d succeeded after %d retry(ies)",
                    batch_index, attempt,
                )
            return True
        except Exception as exc:
            if attempt < max_retries:
                wait = backoff_base ** attempt
                logger.warning(
                    "[EMBED] Batch %d upsert failed (attempt %d/%d): %s — retrying in %.1fs",
                    batch_index, attempt + 1, max_retries + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[EMBED] Batch %d failed after %d attempts: %s",
                    batch_index, max_retries + 1, exc,
                )
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Existing-state reader (for content-hash skip)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_existing_hashes(client, collection_name: str) -> dict[str, str]:
    """
    Scroll through the entire collection and return a mapping of
    {qdrant_point_id (str) → content_hash} for all stored points.

    Uses paginated scroll with page_size=1000 to avoid loading everything
    at once. Returns an empty dict if the collection is empty or on error.
    """
    existing: dict[str, str] = {}
    offset = None
    scroll_batch = 1000

    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=scroll_batch,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            for point in points:
                h = (point.payload or {}).get("content_hash", "")
                existing[str(point.id)] = h
            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            "[EMBED] Found %d existing points in collection '%s'",
            len(existing), collection_name,
        )
    except Exception as exc:
        logger.warning(
            "[EMBED] Could not read existing collection state: %s — will upsert all chunks",
            exc,
        )
        existing = {}

    return existing


# ─────────────────────────────────────────────────────────────────────────────
# Orphan cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_orphans(
    client,
    collection_name: str,
    current_ids: set[str],
    delete_batch_size: int = 500,
) -> int:
    """
    Delete Qdrant points whose ID is not in *current_ids* (stale chunks from
    a previous ingestion of a now-removed CFR section).

    Scrolls the collection in pages, collects orphan IDs, then deletes them
    in batches of *delete_batch_size* with wait=True.

    Returns the number of points deleted.
    """
    orphan_ids: list[str] = []
    offset = None

    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=1000,
                with_payload=False,
                with_vectors=False,
            )
            for point in points:
                pid = str(point.id)
                if pid not in current_ids:
                    orphan_ids.append(pid)
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        logger.error("[EMBED] Orphan scan failed: %s — skipping cleanup", exc)
        return 0

    if not orphan_ids:
        logger.info("[EMBED] No orphaned points to clean up")
        return 0

    logger.info("[EMBED] Cleaning up %d orphaned point(s) …", len(orphan_ids))
    deleted = 0
    for i in range(0, len(orphan_ids), delete_batch_size):
        batch = orphan_ids[i : i + delete_batch_size]
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=batch,
                wait=True,
            )
            deleted += len(batch)
        except Exception as exc:
            logger.error(
                "[EMBED] Failed to delete orphan batch starting at index %d: %s",
                i, exc,
            )

    logger.info("[EMBED] Deleted %d orphaned points", deleted)
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks_path: str | Path = "data/chunks/cfr_chunks.json",
    config: Optional[EmbedderConfig] = None,
    cleanup_orphans: bool = False,
) -> dict:
    """
    Load chunks from *chunks_path*, embed with BGE-M3 (dense + sparse),
    and upsert into Qdrant with the following guarantees:

    - Each batch is confirmed stored (wait=True) before moving to the next.
    - Batches are retried on transient errors (exponential backoff).
    - Chunks whose embed-text hash hasn't changed are skipped (no re-embedding).
    - If cleanup_orphans=True, points no longer present in the source are
      deleted after all upserts complete.

    Returns a summary dict with keys:
      total_chunks, upserted, skipped_unchanged, failed_batches,
      deleted_orphans, collection_name, duration_seconds
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    if config is None:
        config = EmbedderConfig()

    chunks = _load_chunks(chunks_path)
    logger.info("[EMBED] Loaded %d chunks from %s", len(chunks), chunks_path)

    # Connect and ensure collection exists
    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    _ensure_collection(client, config)

    # Pre-compute embed text and content hash for every chunk
    embed_texts = [_build_embed_text(c, config) for c in chunks]

    # Build per-chunk UUID list (one entry per chunk, preserving order)
    chunk_uuids = [_chunk_id_to_uuid(c["chunk_id"]) for c in chunks]

    # Warn loudly if any chunk_ids are duplicated — this means the extractor
    # produced colliding IDs and those chunks will overwrite each other in Qdrant.
    from collections import Counter
    uuid_counts = Counter(chunk_uuids)
    dup_count = sum(v - 1 for v in uuid_counts.values() if v > 1)
    if dup_count:
        logger.warning(
            "[EMBED] WARNING: %d duplicate chunk_id(s) detected in source data — "
            "re-run extraction to fix before embedding. Duplicate points will overwrite each other.",
            dup_count,
        )

    new_hashes = {uid: _content_hash(t) for uid, t in zip(chunk_uuids, embed_texts)}
    current_ids: set[str] = set(new_hashes.keys())

    # Fetch existing state to enable skip-if-unchanged
    existing_hashes = _fetch_existing_hashes(client, config.collection_name)

    # Partition chunks: skip (hash matches) vs embed (new or changed).
    # Iterate over chunk_uuids (same length as chunks) instead of new_hashes
    # (a dict that deduplicates keys) so no chunks are silently dropped.
    to_embed_indices: list[int] = []
    skipped = 0
    for i, (point_id, t) in enumerate(zip(chunk_uuids, embed_texts)):
        if existing_hashes.get(point_id) == _content_hash(t):
            skipped += 1
        else:
            to_embed_indices.append(i)

    logger.info(
        "[EMBED] %d chunks to upsert, %d unchanged (skipping)",
        len(to_embed_indices), skipped,
    )

    # Load BGE-M3 only if there's something to embed
    model = None
    if to_embed_indices:
        model = _load_bgem3_model(config)

    start = time.time()
    total_upserted = 0
    failed_batches = 0
    first_batch = True

    batches = list(_batched(to_embed_indices, config.batch_size))
    for batch_num, idx_batch in enumerate(tqdm(batches, desc="Embedding batches", unit="batch")):
        batch_chunks = [chunks[i] for i in idx_batch]
        batch_texts  = [embed_texts[i] for i in idx_batch]

        # Encode with BGE-M3
        try:
            dense_vecs, sparse_vecs = _encode_batch(model, batch_texts)
        except Exception as exc:
            logger.error(
                "[EMBED] Encoding failed for batch %d: %s — skipping batch",
                batch_num, exc,
            )
            failed_batches += 1
            if first_batch:
                logger.error("[EMBED] First batch encoding failed — aborting early")
                break
            continue

        # Build Qdrant points
        points = [
            PointStruct(
                id=_chunk_id_to_uuid(chunk["chunk_id"]),
                vector={
                    "cfr-dense": dense_vec,
                    "cfr-sparse": SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"],
                    ),
                },
                payload=_build_payload(chunk, embed_text),
            )
            for chunk, dense_vec, sparse_vec, embed_text
            in zip(batch_chunks, dense_vecs, sparse_vecs, batch_texts)
        ]

        # Upsert with retry
        success = _upsert_with_retry(
            client,
            config.collection_name,
            points,
            config.max_retries,
            config.retry_backoff_base,
            batch_num,
        )

        if success:
            total_upserted += len(points)
        else:
            failed_batches += 1
            chunk_ids = [c["chunk_id"] for c in batch_chunks]
            logger.error(
                "[EMBED] Batch %d permanently failed. Affected chunk_ids: %s … (and %d more)",
                batch_num,
                chunk_ids[:5],
                max(0, len(chunk_ids) - 5),
            )
            # Abort on first-batch permanent failure (server likely unreachable)
            if first_batch:
                logger.error("[EMBED] First batch failed all retries — aborting ingestion")
                break

        first_batch = False

    duration = round(time.time() - start, 2)

    # Orphan cleanup (only when caller has fresh source data)
    deleted_orphans = 0
    if cleanup_orphans:
        deleted_orphans = _cleanup_orphans(client, config.collection_name, current_ids)

    logger.info(
        "[EMBED] Done: %d upserted, %d skipped (unchanged), %d failed batches, "
        "%d orphans deleted, %.1fs total",
        total_upserted, skipped, failed_batches, deleted_orphans, duration,
    )

    return {
        "total_chunks": len(chunks),
        "upserted": total_upserted,
        "skipped_unchanged": skipped,
        "failed_batches": failed_batches,
        "deleted_orphans": deleted_orphans,
        "collection_name": config.collection_name,
        "duration_seconds": duration,
        # kept for backwards compat with any callers reading this key
        "total_upserted": total_upserted,
    }
