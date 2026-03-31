"""
Ingestion Pipeline Orchestrator
================================
Ties together extractor → embedder → graph_builder into a single
sequential pipeline.  Called from main.py when the /api/ingest endpoint
is triggered.

Step flow:
  1. EXTRACT  — parse all CFR XML files → cfr_chunks.json
  2. EMBED    — embed chunks → Qdrant vector collection
  3. GRAPH    — build lexical graph → Neo4j

Each step is independently runnable and the pipeline can be resumed
from any step (e.g. skip extraction if chunks JSON already exists).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .extractor import extract_all
from .embedder import EmbedderConfig, embed_and_store
from .graph_builder import GraphConfig, build_graph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Paths
    xml_dir: str            = "backend/data/cfr_xml"
    chunks_output_path: str = "backend/data/chunks/cfr_chunks.json"

    # Which steps to run
    run_extract: bool = True
    run_embed:   bool = True
    run_graph:   bool = True

    # Skip extraction if output file already exists (resume mode)
    skip_extract_if_exists: bool = False

    # Sub-configs (use defaults if not provided)
    embedder_config: Optional[EmbedderConfig] = None
    graph_config:    Optional[GraphConfig]    = None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline status
# ─────────────────────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    SKIPPED  = "skipped"
    FAILED   = "failed"


@dataclass
class StepResult:
    name: str
    status: StepStatus = StepStatus.PENDING
    result: dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class PipelineResult:
    started_at: str = ""
    completed_at: str = ""
    total_duration_seconds: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "success": self.success,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "result": s.result,
                    "error": s.error,
                    "duration_seconds": round(s.duration_seconds, 2),
                }
                for s in self.steps
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: Optional[PipelineConfig] = None) -> PipelineResult:
    """
    Execute the full ingestion pipeline according to *config*.

    Returns a PipelineResult with per-step status and results.
    The result is safe to serialise to JSON for API responses.
    """
    from datetime import datetime, timezone

    if config is None:
        config = PipelineConfig()

    pipeline_start = time.time()
    result = PipelineResult(
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("=" * 60)
    logger.info("CFR Ingestion Pipeline starting")
    logger.info("  xml_dir          : %s", config.xml_dir)
    logger.info("  chunks_output    : %s", config.chunks_output_path)
    logger.info("  steps            : extract=%s embed=%s graph=%s",
                config.run_extract, config.run_embed, config.run_graph)
    logger.info("=" * 60)

    # ── Step 1: EXTRACT ───────────────────────────────────────────────────
    extract_step = StepResult(name="extract")
    result.steps.append(extract_step)

    if not config.run_extract:
        extract_step.status = StepStatus.SKIPPED
        logger.info("[EXTRACT] Skipped (run_extract=False)")
    elif (
        config.skip_extract_if_exists
        and Path(config.chunks_output_path).exists()
    ):
        extract_step.status = StepStatus.SKIPPED
        logger.info(
            "[EXTRACT] Skipped — chunks file already exists at %s",
            config.chunks_output_path,
        )
    else:
        extract_step.status = StepStatus.RUNNING
        step_start = time.time()
        try:
            logger.info("[EXTRACT] Starting XML extraction …")
            extract_result = extract_all(
                xml_dir=config.xml_dir,
                output_path=config.chunks_output_path,
            )
            extract_step.status = StepStatus.DONE
            extract_step.result = extract_result
            extract_step.duration_seconds = time.time() - step_start
            logger.info(
                "[EXTRACT] Done — %d chunks in %.1fs",
                extract_result["total_chunks"],
                extract_step.duration_seconds,
            )
        except Exception as exc:
            extract_step.status = StepStatus.FAILED
            extract_step.error = str(exc)
            extract_step.duration_seconds = time.time() - step_start
            logger.exception("[EXTRACT] Failed: %s", exc)
            # Stop pipeline on extraction failure — downstream steps need chunks
            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.total_duration_seconds = time.time() - pipeline_start
            result.success = False
            return result

    # ── Step 2: EMBED ─────────────────────────────────────────────────────
    embed_step = StepResult(name="embed")
    result.steps.append(embed_step)

    if not config.run_embed:
        embed_step.status = StepStatus.SKIPPED
        logger.info("[EMBED] Skipped (run_embed=False)")
    else:
        embed_step.status = StepStatus.RUNNING
        step_start = time.time()
        try:
            logger.info("[EMBED] Starting embedding and Qdrant upsert …")
            embed_result = embed_and_store(
                chunks_path=config.chunks_output_path,
                config=config.embedder_config,
            )
            embed_step.status = StepStatus.DONE
            embed_step.result = embed_result
            embed_step.duration_seconds = time.time() - step_start
            logger.info(
                "[EMBED] Done — %d points upserted in %.1fs",
                embed_result["total_upserted"],
                embed_step.duration_seconds,
            )
        except Exception as exc:
            embed_step.status = StepStatus.FAILED
            embed_step.error = str(exc)
            embed_step.duration_seconds = time.time() - step_start
            logger.exception("[EMBED] Failed: %s", exc)
            # Continue to graph step even if embedding fails

    # ── Step 3: GRAPH ─────────────────────────────────────────────────────
    graph_step = StepResult(name="graph")
    result.steps.append(graph_step)

    if not config.run_graph:
        graph_step.status = StepStatus.SKIPPED
        logger.info("[GRAPH] Skipped (run_graph=False)")
    else:
        graph_step.status = StepStatus.RUNNING
        step_start = time.time()
        try:
            logger.info("[GRAPH] Starting Neo4j graph build …")
            graph_result = build_graph(
                chunks_path=config.chunks_output_path,
                config=config.graph_config,
            )
            graph_step.status = StepStatus.DONE
            graph_step.result = graph_result
            graph_step.duration_seconds = time.time() - step_start
            logger.info(
                "[GRAPH] Done — %d chunks in %.1fs",
                graph_result["total_chunks_written"],
                graph_step.duration_seconds,
            )
        except Exception as exc:
            graph_step.status = StepStatus.FAILED
            graph_step.error = str(exc)
            graph_step.duration_seconds = time.time() - step_start
            logger.exception("[GRAPH] Failed: %s", exc)

    # ── Finalise ──────────────────────────────────────────────────────────
    failed_steps = [s for s in result.steps if s.status == StepStatus.FAILED]
    result.success = len(failed_steps) == 0
    result.completed_at = datetime.now(timezone.utc).isoformat()
    result.total_duration_seconds = time.time() - pipeline_start

    logger.info("=" * 60)
    logger.info(
        "Pipeline %s in %.1fs",
        "completed successfully" if result.success else "completed with errors",
        result.total_duration_seconds,
    )
    logger.info("=" * 60)

    return result
