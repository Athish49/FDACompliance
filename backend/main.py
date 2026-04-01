"""
FDA Compliance AI — Root API Server
=====================================
FastAPI application that exposes ingestion pipeline and search endpoints.

Endpoints:
  POST /api/ingest          — trigger the full ingestion pipeline
  GET  /api/ingest/status   — check last pipeline run status
  POST /api/search          — hybrid search over CFR chunks
  GET  /api/chunks/{id}     — fetch a single chunk by ID
  GET  /health              — liveness probe

Run with:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ingestion.pipeline import (
    EmbedderConfig,
    PipelineConfig,
    PipelineResult,
    StepStatus,
    run_pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FDA Compliance AI",
    description="Regulatory intelligence API powered by CFR data",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory pipeline state (single-run lock + last result)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_pipeline_running = False
_last_result: Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# Retriever singleton (lazy-initialised on first search)
# ─────────────────────────────────────────────────────────────────────────────

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from backend.retrieval.retriever import CFRRetriever
        _retriever = CFRRetriever()
    return _retriever


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models — Ingestion
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Options for a pipeline run. All fields are optional — defaults are used if omitted."""

    run_extract: bool = Field(True,  description="Run the XML extraction step")
    run_embed:   bool = Field(True,  description="Run the Qdrant embedding step")
    run_graph:   bool = Field(False, description="Run the Neo4j graph building step")

    skip_extract_if_exists: bool = Field(
        False,
        description="Skip extraction if cfr_chunks.json already exists (resume mode)",
    )

    qdrant_url: str = Field("http://localhost:6333", description="Qdrant server URL")
    qdrant_collection: str = Field("cfr_chunks", description="Qdrant collection name")


class IngestResponse(BaseModel):
    message: str
    run_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models — Search
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    use_reranker: bool = Field(True, description="Apply cross-encoder reranking")

    # Optional filters
    part_number: Optional[str] = Field(None, description="Filter by CFR part number")
    chapter_number: Optional[str] = Field(None, description="Filter by chapter number")
    subpart_letter: Optional[str] = Field(None, description="Filter by subpart letter")
    section_number: Optional[str] = Field(None, description="Filter by section number")
    chunk_type: Optional[str] = Field(None, description="Filter by chunk type (section/paragraph/definition)")
    source_file: Optional[str] = Field(None, description="Filter by source XML file")


class SearchResultItem(BaseModel):
    chunk_id: str
    score: float
    reranker_score: Optional[float] = None
    text: str
    cfr_citation: Optional[str] = None
    chunk_type: Optional[str] = None
    section_preamble: Optional[str] = None
    hierarchy: dict = {}
    defines: Optional[str] = None
    overflow_chunks: list[dict] = []
    metadata: dict = {}


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResultItem]


# ─────────────────────────────────────────────────────────────────────────────
# Background task wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_background(config: PipelineConfig) -> None:
    global _pipeline_running, _last_result
    try:
        result = run_pipeline(config)
        _last_result = result.to_dict()
    except Exception as exc:
        logger.exception("Unhandled error in background pipeline: %s", exc)
        _last_result = {
            "success": False,
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        _pipeline_running = False


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — System
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    """Liveness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — Ingestion
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ingest", response_model=IngestResponse, tags=["ingestion"])
def trigger_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the CFR ingestion pipeline asynchronously.

    The pipeline runs in a background thread; poll /api/ingest/status
    to check completion.  Only one pipeline run is allowed at a time.
    """
    global _pipeline_running

    with _pipeline_lock:
        if _pipeline_running:
            raise HTTPException(
                status_code=409,
                detail="A pipeline run is already in progress. "
                       "Poll /api/ingest/status for updates.",
            )
        _pipeline_running = True

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config = PipelineConfig(
        run_extract=request.run_extract,
        run_embed=request.run_embed,
        run_graph=request.run_graph,
        skip_extract_if_exists=request.skip_extract_if_exists,
        embedder_config=EmbedderConfig(
            qdrant_url=request.qdrant_url,
            collection_name=request.qdrant_collection,
        ),
    )

    background_tasks.add_task(_run_pipeline_background, config)

    logger.info("Pipeline run %s started in background", run_id)
    return IngestResponse(
        message="Ingestion pipeline started. Poll /api/ingest/status for updates.",
        run_id=run_id,
    )


@app.get("/api/ingest/status", tags=["ingestion"])
def ingest_status():
    """
    Returns the status of the most recent pipeline run.

    While running: {status: "running"}
    After completion: full PipelineResult dict with per-step details.
    No run yet: {status: "no_run"}
    """
    if _pipeline_running:
        return {"status": "running"}
    if _last_result is None:
        return {"status": "no_run"}
    return _last_result


@app.post("/api/ingest/extract-only", tags=["ingestion"])
def trigger_extract_only(background_tasks: BackgroundTasks):
    """
    Convenience endpoint: run only the XML extraction step.
    Useful for refreshing the chunks JSON without re-embedding.
    """
    global _pipeline_running

    with _pipeline_lock:
        if _pipeline_running:
            raise HTTPException(status_code=409, detail="Pipeline already running.")
        _pipeline_running = True

    config = PipelineConfig(run_extract=True, run_embed=False, run_graph=False)
    background_tasks.add_task(_run_pipeline_background, config)
    return {"message": "Extraction step started."}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — Search
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/search", response_model=SearchResponse, tags=["search"])
def search(request: SearchRequest):
    """
    Hybrid search over CFR chunks.

    Performs BGE-M3 dense + sparse search, fuses with RRF,
    and optionally reranks with bge-reranker-v2-m3.
    """
    from backend.retrieval.retriever import SearchFilters

    retriever = _get_retriever()

    filters = SearchFilters(
        part_number=request.part_number,
        chapter_number=request.chapter_number,
        subpart_letter=request.subpart_letter,
        section_number=request.section_number,
        chunk_type=request.chunk_type,
        source_file=request.source_file,
    )

    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        filters=filters,
    )

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                score=r.score,
                reranker_score=r.reranker_score,
                text=r.text,
                cfr_citation=r.cfr_citation,
                chunk_type=r.chunk_type,
                section_preamble=r.section_preamble,
                hierarchy=r.hierarchy,
                defines=r.defines,
                overflow_chunks=r.overflow_chunks,
                metadata=r.metadata,
            )
            for r in results
        ],
    )


@app.get("/api/chunks/{chunk_id}", tags=["search"])
def get_chunk(chunk_id: str):
    """Fetch a single chunk by its chunk_id."""
    retriever = _get_retriever()
    payload = retriever.get_chunk_by_id(chunk_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
    return payload
