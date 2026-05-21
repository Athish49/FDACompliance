"""Structured per-session logging for the FDA Compliance AI pipeline.

Log files are written to:
    backend/logs/YYYY-MM-DD/{HH-MM-SS}_{session_id[:8]}.json

Each log file captures the full pipeline trace for one query:
  - received query + timestamp
  - query analysis (intent, entities)
  - sub-question decomposition
  - per sub-question: each CRAG attempt with retrieved chunks (chunk_id + text only)
  - consistency check
  - final synthesis + elapsed time
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LOGS_ROOT = Path(__file__).parent.parent / "logs"

_active_sessions: dict[str, "SessionLogger"] = {}
_sessions_lock = threading.Lock()


def create_session(query: str) -> str:
    """Create a new pipeline session, return session_id."""
    session_id = str(uuid.uuid4())
    sl = SessionLogger(session_id, query)
    with _sessions_lock:
        _active_sessions[session_id] = sl
    return session_id


def get_session(session_id: str) -> Optional["SessionLogger"]:
    return _active_sessions.get(session_id)


def close_session(session_id: str) -> None:
    """Remove session from active map and do a final flush."""
    with _sessions_lock:
        sl = _active_sessions.pop(session_id, None)
    if sl:
        sl.flush()


class SessionLogger:
    """Thread-safe, progressively-written JSON log for one query session."""

    def __init__(self, session_id: str, query: str) -> None:
        self.session_id = session_id
        self._lock = threading.Lock()

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        log_dir = _LOGS_ROOT / date_str
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"{time_str}_{session_id[:8]}.json"

        self._data: dict = {
            "session_id": session_id,
            "started_at": now.isoformat(),
            "completed_at": None,
            "elapsed_seconds": None,
            "query": query,
            "pipeline": {
                "query_analysis": None,
                "decomposition": None,
                "sub_questions": {},
                "consistency_check": None,
                "final_synthesis": None,
            },
        }
        self.flush()

    # ── Pipeline stage loggers ─────────────────────────────────────────────────

    def log_query_analysis(
        self,
        intent: str,
        entities: dict,
        is_multi_part: bool,
        needs_clarification: bool,
    ) -> None:
        with self._lock:
            self._data["pipeline"]["query_analysis"] = {
                "intent_type": intent,
                "entities": entities,
                "is_multi_part": is_multi_part,
                "needs_clarification": needs_clarification,
            }
        self.flush()

    def log_decomposition(self, sub_questions: list[dict]) -> None:
        with self._lock:
            self._data["pipeline"]["decomposition"] = {
                "count": len(sub_questions),
                "sub_questions": [
                    {
                        "label": f"SQ-{i + 1}",
                        "index": i + 1,
                        "id": sq["id"],
                        "text": sq["text"],
                        "variants": {
                            "primary": sq.get("variants", {}).get("primary", ""),
                            "stepback": sq.get("variants", {}).get("stepback", ""),
                        },
                    }
                    for i, sq in enumerate(sub_questions)
                ],
            }
            sq_dict = self._data["pipeline"]["sub_questions"]
            for i, sq in enumerate(sub_questions):
                sq_dict[sq["id"]] = {
                    "label": f"SQ-{i + 1}",
                    "index": i + 1,
                    "id": sq["id"],
                    "text": sq["text"],
                    "status": "pending",
                    "attempts": {},
                    "final_crag_verdict": None,
                    "confidence": None,
                    "citations_count": None,
                    "answer_summary": None,
                    "caveats": [],
                    "flags": [],
                }
        self.flush()

    def get_sq_label(self, sq_id: str) -> str:
        """Return human-readable label (SQ-1, SQ-2 ...) for a sub-question id."""
        entry = self._data["pipeline"]["sub_questions"].get(sq_id)
        if entry:
            return entry.get("label", sq_id[:8])
        return sq_id[:8]

    def log_sq_attempt(
        self,
        sq_id: str,
        attempt_num: int,
        crag_verdict: str,
        top_score: float,
        retrieved_chunks: list[dict],
        reformulation: Optional[str] = None,
    ) -> None:
        with self._lock:
            sq_dict = self._data["pipeline"]["sub_questions"]
            if sq_id not in sq_dict:
                idx = len(sq_dict) + 1
                sq_dict[sq_id] = {
                    "label": f"SQ-{idx}",
                    "index": idx,
                    "id": sq_id,
                    "text": "",
                    "status": "in_progress",
                    "attempts": {},
                    "final_crag_verdict": None,
                    "confidence": None,
                    "citations_count": None,
                    "answer_summary": None,
                    "caveats": [],
                    "flags": [],
                }
            entry = sq_dict[sq_id]
            entry["status"] = "in_progress"
            attempt_key = f"attempt_{attempt_num + 1}"
            entry["attempts"][attempt_key] = {
                "attempt_number": attempt_num + 1,
                "crag_verdict": crag_verdict,
                "top_reranker_score": round(top_score, 4),
                "chunks_retrieved_count": len(retrieved_chunks),
                "chunks_retrieved": [
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "text": (c.get("text") or "")[:300],
                    }
                    for c in retrieved_chunks
                ],
                "reformulation_used": reformulation,
            }
        self.flush()

    def log_sq_answer(
        self,
        sq_id: str,
        crag_verdict: str,
        confidence: float,
        citations_count: int,
        answer_summary: str,
        caveats: list[str],
        flags: list[str],
    ) -> None:
        with self._lock:
            sq_dict = self._data["pipeline"]["sub_questions"]
            if sq_id in sq_dict:
                sq_dict[sq_id].update({
                    "final_crag_verdict": crag_verdict,
                    "confidence": round(confidence, 3),
                    "citations_count": citations_count,
                    "answer_summary": (answer_summary or "")[:500],
                    "caveats": caveats,
                    "flags": flags,
                    "status": "completed",
                })
        self.flush()

    def log_consistency(self, conflicts_found: int, unresolved: int) -> None:
        with self._lock:
            self._data["pipeline"]["consistency_check"] = {
                "conflicts_found": conflicts_found,
                "unresolved_conflicts": unresolved,
            }
        self.flush()

    def log_final_synthesis(
        self,
        ruling: str,
        confidence_level: str,
        citations_count: int,
        low_confidence_sub_questions: list[str],
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._data["pipeline"]["final_synthesis"] = {
                "ruling": ruling,
                "confidence_level": confidence_level,
                "citations_count": citations_count,
                "low_confidence_sub_questions": low_confidence_sub_questions,
            }
            self._data["completed_at"] = now.isoformat()
            try:
                started = datetime.fromisoformat(self._data["started_at"])
                self._data["elapsed_seconds"] = round((now - started).total_seconds(), 1)
            except Exception:
                pass
        self.flush()

    # ── Atomic file write ──────────────────────────────────────────────────────

    def flush(self) -> None:
        """Atomically overwrite the log file so readers never see a partial write."""
        try:
            dir_path = self._log_path.parent
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self._log_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("Session log write failed [%s]: %s", self.session_id[:8], exc)
