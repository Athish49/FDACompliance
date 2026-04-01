"""LangGraph StateGraph — wires the 6 agent nodes into a compiled graph."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from agents.conflict_detector import conflict_detector_node
from agents.definition_resolver import definition_resolver_node
from agents.planner import planner_node
from agents.retriever_node import retriever_node
from agents.state import ComplianceState
from agents.synthesizer import synthesizer_node
from agents.verifier import verifier_node

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def should_retry(state: ComplianceState) -> str:
    """Conditional edge: retry via planner or proceed to conflict detector."""
    if not state.get("verification_passed", True) and state.get("retry_count", 0) < MAX_RETRIES:
        logger.info("Verifier triggered retry (%d/%d)", state["retry_count"], MAX_RETRIES)
        return "planner"
    return "conflict_detector"


# ── Build graph ───────────────────────────────────────────────────────────

graph = StateGraph(ComplianceState)

graph.add_node("planner", planner_node)
graph.add_node("retriever", retriever_node)
graph.add_node("definition_resolver", definition_resolver_node)
graph.add_node("synthesizer", synthesizer_node)
graph.add_node("verifier", verifier_node)
graph.add_node("conflict_detector", conflict_detector_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "definition_resolver")
graph.add_edge("definition_resolver", "synthesizer")
graph.add_edge("synthesizer", "verifier")

graph.add_conditional_edges("verifier", should_retry, {
    "planner": "planner",
    "conflict_detector": "conflict_detector",
})
graph.add_edge("conflict_detector", END)

query_graph = graph.compile()
