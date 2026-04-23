"""
Document Analysis LangGraph — Phase 4.

Graph flow:
    claim_extractor → regulation_mapper → violation_classifier → report_builder → END

Error short-circuit:
    If claim_extractor sets state["error"], the router skips mapper+classifier
    and goes directly to report_builder.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from document_analysis.state import DocumentAnalysisState
from document_analysis.claim_extractor import claim_extractor_node
from document_analysis.regulation_mapper import regulation_mapper_node
from document_analysis.violation_classifier import violation_classifier_node
from document_analysis.report_builder import report_builder_node


def _route_after_extraction(state: DocumentAnalysisState) -> str:
    """Skip mapper + classifier if extraction failed."""
    if state.get("error"):
        return "report_builder"
    return "regulation_mapper"


def build_document_analysis_graph() -> StateGraph:
    graph = StateGraph(DocumentAnalysisState)

    graph.add_node("claim_extractor", claim_extractor_node)
    graph.add_node("regulation_mapper", regulation_mapper_node)
    graph.add_node("violation_classifier", violation_classifier_node)
    graph.add_node("report_builder", report_builder_node)

    graph.set_entry_point("claim_extractor")

    graph.add_conditional_edges(
        "claim_extractor",
        _route_after_extraction,
        {
            "regulation_mapper": "regulation_mapper",
            "report_builder": "report_builder",
        },
    )

    graph.add_edge("regulation_mapper", "violation_classifier")
    graph.add_edge("violation_classifier", "report_builder")
    graph.add_edge("report_builder", END)

    return graph.compile()


# Module-level compiled graph (lazy singleton)
_document_analysis_graph = None


def get_document_analysis_graph():
    global _document_analysis_graph
    if _document_analysis_graph is None:
        _document_analysis_graph = build_document_analysis_graph()
    return _document_analysis_graph
