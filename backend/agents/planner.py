"""Planner agent — intent classification + query decomposition."""

from __future__ import annotations

import logging

from agents.llm import llm_completion_json, parse_llm_json
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """\
You are an FDA regulatory compliance planner. Given a user question about FDA regulations (Title 21 CFR), you must:

1. Classify the intent as one of: "compliance_question", "definition_lookup", "comparison", "general"
2. Decompose the question into 1-4 targeted search queries that will retrieve the most relevant CFR sections
3. Extract any specific CFR part/section/chapter numbers mentioned in the question

Return ONLY valid JSON in this format:
{
  "intent": "compliance_question",
  "sub_questions": ["query 1", "query 2"],
  "search_filters": {"part_number": "101", "section_number": "101.12"}
}

Rules:
- sub_questions should be specific, regulatory-focused search queries
- search_filters should only include fields explicitly mentioned (part_number, section_number, chapter_number)
- If no specific CFR references are mentioned, return search_filters as {}
- For definition_lookup intent, include a sub_question like "definition of <term>"
- Keep sub_questions to 1-4 entries; prefer fewer, more targeted queries"""

RETRY_ADDENDUM = """\

IMPORTANT: A previous attempt to answer this question had verification issues.
The following claims could not be verified against source text:
{issues}

Adjust your sub_questions to specifically find CFR sections that would support or refute these claims.
Add more targeted queries to fill the gaps."""


def planner_node(state: ComplianceState) -> ComplianceState:
    """Classify intent, decompose query into sub-questions, extract filters."""
    query = state["query"]

    user_content = f"User question: {query}"

    # On retry, append verification issues
    verification_issues = state.get("verification_issues")
    if verification_issues:
        issues_text = "\n".join(
            f"- Claim: {i.get('claim', 'N/A')} — Issue: {i.get('issue', 'N/A')}"
            for i in verification_issues
        )
        user_content += RETRY_ADDENDUM.format(issues=issues_text)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    raw = llm_completion_json(messages)
    parsed = parse_llm_json(raw, messages)

    intent = parsed.get("intent", "general")
    sub_questions = parsed.get("sub_questions", [query])
    search_filters = parsed.get("search_filters", {})

    # Ensure at least one sub-question
    if not sub_questions:
        sub_questions = [query]

    logger.info("Planner: intent=%s, %d sub-questions, filters=%s",
                intent, len(sub_questions), search_filters)

    return {
        "intent": intent,
        "sub_questions": sub_questions,
        "search_filters": search_filters,
    }
