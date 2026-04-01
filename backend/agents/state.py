"""ComplianceState — shared state for the LangGraph multi-agent pipeline."""

from __future__ import annotations

from typing import TypedDict


class ComplianceState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────
    query: str
    intent: str  # compliance_question | definition_lookup | comparison | general

    # ── Planner ───────────────────────────────────────────────────────────
    sub_questions: list[str]
    search_filters: dict  # {part_number, section_number, chapter_number} or {}

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_chunks: list[dict]
    cross_ref_chunks: list[dict]

    # ── Definition resolver ───────────────────────────────────────────────
    definitions_resolved: dict[str, str]  # term → definition text
    definition_chunks: list[dict]

    # ── Synthesizer ───────────────────────────────────────────────────────
    draft_answer: str
    citations: list[dict]  # [{section, title, text_snippet}]
    confidence_score: float

    # ── Verifier ──────────────────────────────────────────────────────────
    verification_passed: bool
    verification_issues: list[dict]  # [{claim, issue, detail}]
    retry_count: int

    # ── Conflict detector ─────────────────────────────────────────────────
    conflicts_detected: bool
    conflict_flags: list[dict]  # [{sections, description}]

    # ── Final ─────────────────────────────────────────────────────────────
    final_response: dict
    error: str | None
