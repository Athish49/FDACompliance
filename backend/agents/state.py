"""ComplianceState — shared state for the v2 LangGraph multi-agent pipeline."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ComplianceState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    query: str

    # ── Stage 1: Query Analysis ────────────────────────────────────────────────
    analyzed_query: dict       # intent_type, entities, is_multi_part, explicit_refs
    needs_clarification: bool
    clarification_question: str | None

    # ── Stage 2: Sub-Question Decomposition ───────────────────────────────────
    sub_questions: list[dict]  # [{id, text, variants:{primary,hyde_passage,stepback}, source_query}]

    # ── Fan-out: per-sub-question Send payload ────────────────────────────────
    # Populated only in the Send branch — not in top-level state.
    sub_question: dict

    # ── Fan-in: operator.add reducer collects results from parallel branches ──
    sub_answers: Annotated[list[dict], operator.add]  # list[SubAnswer]

    # ── Stage 6: Consistency & Conflict Detection ─────────────────────────────
    resolved_answers: list[dict]
    unresolved_conflicts: list[dict]   # list[ConflictReport]

    # ── Stage 7: Final Synthesis ──────────────────────────────────────────────
    final_answer: dict      # v2 FinalAnswer schema
    final_response: dict    # API-compatible QueryResponse shape

    # ── Error ─────────────────────────────────────────────────────────────────
    error: str | None
