"""
DocumentAnalysisState — LangGraph state for the document violation analysis flow.
"""

from __future__ import annotations

from typing import TypedDict


class DocumentAnalysisState(TypedDict, total=False):
    # Inputs
    document_text: str
    document_name: str

    # Extracted claims from the document
    # Each: {claim_text, claim_type, location_hint}
    extracted_claims: list[dict]

    # Per-claim CFR mappings from retriever
    # Each: {claim_text, claim_type, top_chunks, mapping_confidence}
    claim_mappings: list[dict]

    # Per-claim violation objects from classifier
    # Each: {claim_text, claim_type, cfr_citation, violation_type, severity,
    #         explanation, relevant_text, location_hint}
    violations: list[dict]

    # Final assembled report dict (ViolationReport)
    violation_report: dict

    # Set on any fatal error to short-circuit to report_builder
    error: str | None
