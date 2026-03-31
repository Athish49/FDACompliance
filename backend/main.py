"""
FDA Compliance AI — Root API Server
=====================================
FastAPI application that exposes ingestion pipeline endpoints.

Endpoints:
  POST /api/ingest          — trigger the full ingestion pipeline
  GET  /api/ingest/status   — check last pipeline run status
  GET  /health              — liveness probe

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
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
    GraphConfig,
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
    version="0.1.0",
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
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Options for a pipeline run. All fields are optional — defaults are used if omitted."""

    # Which steps to run
    run_extract: bool = Field(True,  description="Run the XML extraction step")
    run_embed:   bool = Field(True,  description="Run the Qdrant embedding step")
    run_graph:   bool = Field(True,  description="Run the Neo4j graph building step")

    # Skip extraction if the chunks file already exists
    skip_extract_if_exists: bool = Field(
        False,
        description="Skip extraction if cfr_chunks.json already exists (resume mode)",
    )

    # Qdrant settings
    qdrant_url: str = Field("http://localhost:6333", description="Qdrant server URL")
    qdrant_collection: str = Field("cfr_chunks", description="Qdrant collection name")
    embedding_backend: str = Field(
        "sentence-transformers",
        description="Embedding backend: 'sentence-transformers' or 'openai'",
    )

    # Neo4j settings
    neo4j_uri: str      = Field("bolt://localhost:7687", description="Neo4j bolt URI")
    neo4j_user: str     = Field("neo4j",    description="Neo4j username")
    neo4j_password: str = Field("password", description="Neo4j password")


class IngestResponse(BaseModel):
    message: str
    run_id: Optional[str] = None


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
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    """Liveness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


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
            embedding_backend=request.embedding_backend,
        ),
        graph_config=GraphConfig(
            neo4j_uri=request.neo4j_uri,
            neo4j_user=request.neo4j_user,
            neo4j_password=request.neo4j_password,
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
