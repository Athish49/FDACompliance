"""LangGraph StateGraph — v2 multi-agent compliance reasoning pipeline.

Flow:
  query_analyzer ──→ [clarification?] ──→ decomposer ──→ [fan-out per sub-question]
        │                                                          │ (parallel)
  clarification_response → END                         process_sub_question × N
                                                                   │ (fan-in via operator.add)
                                                         consistency_detector
                                                                   │
                                                          final_synthesizer → END
"""
from __future__ import annotations

import logging

from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from agents.consistency import consistency_detector_node
from agents.decomposer import decomposer_node
from agents.final_synthesizer import DISCLAIMER, final_synthesizer_node
from agents.query_analyzer import query_analyzer_node
from agents.retrieval_pipeline import process_sub_question_node
from agents.state import ComplianceState

logger = logging.getLogger(__name__)


# ── Clarification short-circuit node ─────────────────────────────────────────

def clarification_response_node(state: ComplianceState) -> dict:
    """Return a structured clarification request without further LLM calls."""
    clarification_q = state.get("clarification_question") or (
        "Could you please provide more details about your question? "
        "Specifically: what product type, what type of label claim, "
        "and which FDA regulatory area (food, drug, device, cosmetic) are you asking about?"
    )
    final_response = {
        "answer": clarification_q,
        "citations": [],
        "confidence_score": 0.0,
        "conflicts_detected": False,
        "conflict_details": [],
        "disclaimer": DISCLAIMER,
        "retrieved_sections": [],
        "verification_passed": False,
    }
    return {
        "final_answer": {"ruling": "CONDITIONAL", "ruling_summary": clarification_q},
        "final_response": final_response,
    }


# ── Conditional edge functions ────────────────────────────────────────────────

def check_clarification(state: ComplianceState) -> str:
    """Route to clarification response or decomposer."""
    if state.get("needs_clarification"):
        logger.info("Query requires clarification — short-circuiting")
        return "clarification_response"
    return "decomposer"


def dispatch_sub_questions(state: ComplianceState) -> list:
    """Fan-out: create one Send per sub-question for parallel processing."""
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        logger.warning("No sub-questions generated — sending original query as single sub-question")
        sub_questions = [{
            "id": "fallback",
            "text": state.get("query", ""),
            "variants": {
                "primary": state.get("query", ""),
                "hyde_passage": state.get("query", ""),
                "stepback": state.get("query", ""),
            },
            "source_query": state.get("query", ""),
        }]
    return [
        Send("process_sub_question", {
            "query": state["query"],
            "analyzed_query": state.get("analyzed_query", {}),
            "sub_question": sq,
            "session_id": state.get("session_id", ""),
        })
        for sq in sub_questions
    ]


# ── Build graph ───────────────────────────────────────────────────────────────

graph = StateGraph(ComplianceState)

graph.add_node("query_analyzer", query_analyzer_node)
graph.add_node("clarification_response", clarification_response_node)
graph.add_node("decomposer", decomposer_node)
graph.add_node("process_sub_question", process_sub_question_node)
graph.add_node("consistency_detector", consistency_detector_node)
graph.add_node("final_synthesizer", final_synthesizer_node)

graph.set_entry_point("query_analyzer")

graph.add_conditional_edges("query_analyzer", check_clarification, {
    "clarification_response": "clarification_response",
    "decomposer": "decomposer",
})
graph.add_edge("clarification_response", END)

graph.add_conditional_edges("decomposer", dispatch_sub_questions)
graph.add_edge("process_sub_question", "consistency_detector")

graph.add_edge("consistency_detector", "final_synthesizer")
graph.add_edge("final_synthesizer", END)

query_graph = graph.compile()
