"""Stage 2 — Sub-Question Decomposition + Query Variant Generation.

Decomposes the analyzed query into atomic sub-questions (single intent, independently
answerable).  For each sub-question, generates three retrieval variants in parallel:
  primary   — keyword-rich rephrasing for dense/sparse retrieval
  hyde      — hypothetical CFR-style passage (HyDE: query-time)
  stepback  — abstract principle-level query for broad recall
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.llm import llm_completion, llm_completion_json, parse_llm_json
from agents.session_logger import get_session
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

# ── Temperatures ───────────────────────────────────────────────────────────────
HYDE_TEMP = 0.7
STEPBACK_TEMP = 0.3

# ── System prompts ─────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
You are an FDA regulatory compliance expert. Decompose a compliance question into the MINIMUM number
of atomic sub-questions needed to fully answer it.

CRITICAL RULES — sub-question text format:
1. Each sub-question MUST be a declarative information-retrieval topic, NOT a question.
   BAD (yes/no):   "Does 20g protein per serving qualify as an 'excellent source' claim?"
   BAD (question): "What are the thresholds for an 'excellent source' of protein?"
   GOOD (topic):   "FDA nutrient content claim thresholds for 'excellent source' of protein"

2. NEVER produce yes/no questions. Rewrite them as topic statements describing WHAT
   regulatory information must be retrieved to answer them.

3. Each sub-question must cover a DISTINCT, non-overlapping regulatory topic.
   If two sub-questions would retrieve the same CFR sections, merge them into one.

4. Each sub-question must be independently answerable from CFR text alone, without
   needing the answer to any other sub-question.

5. Respect the is_multi_part flag:
   - is_multi_part = false → produce EXACTLY 1 sub-question
   - is_multi_part = true  → produce the minimum needed (max 4)

Return ONLY valid JSON:
{
  "sub_questions": [
    {"text": "declarative retrieval topic 1"},
    {"text": "declarative retrieval topic 2"}
  ]
}"""

_PRIMARY_SYSTEM = """\
Rephrase the following compliance question as a concise, keyword-rich search query optimised for
regulatory database retrieval.  Remove conversational phrasing.  Output only the rephrased query."""

_HYDE_SYSTEM = """\
You are an FDA regulatory expert.  Write a short passage (100-150 words) in the style of the Code
of Federal Regulations that would directly answer the following compliance question.  Use regulatory
language (shall, must, may not) and CFR citation style.  Do not add disclaimers.  Output only the
passage."""

_STEPBACK_SYSTEM = """\
Rewrite the following specific compliance question as a broader, more abstract question about the
underlying FDA regulatory principle it relates to.  Output only the rewritten question."""

_MIN_VARIANT_LEN = 15  # anything shorter is treated as an empty/failed response


# ── Variant generators ─────────────────────────────────────────────────────────

def _generate_primary(text: str) -> str:
    messages = [
        {"role": "system", "content": _PRIMARY_SYSTEM},
        {"role": "user", "content": text},
    ]
    return llm_completion(messages, max_tokens=128, temperature=0.1)


def _generate_hyde(text: str) -> str:
    messages = [
        {"role": "system", "content": _HYDE_SYSTEM},
        {"role": "user", "content": text},
    ]
    return llm_completion(messages, max_tokens=300, temperature=HYDE_TEMP)


def _generate_stepback(text: str) -> str:
    messages = [
        {"role": "system", "content": _STEPBACK_SYSTEM},
        {"role": "user", "content": text},
    ]
    return llm_completion(messages, max_tokens=128, temperature=STEPBACK_TEMP)


def _validate_variant(result: str, fallback: str, label: str) -> str:
    """Return result if non-empty and long enough; otherwise log and return fallback."""
    if result and len(result.strip()) >= _MIN_VARIANT_LEN:
        return result.strip()
    logger.warning(
        "%s variant invalid (got %r) — using fallback", label, (result or "")[:60]
    )
    return fallback


def _generate_variants(sub_q_text: str) -> dict:
    """Generate primary, hyde, stepback variants in parallel threads."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_primary = ex.submit(_generate_primary, sub_q_text)
        f_hyde = ex.submit(_generate_hyde, sub_q_text)
        f_stepback = ex.submit(_generate_stepback, sub_q_text)

        primary = sub_q_text
        hyde = sub_q_text
        stepback = sub_q_text

        try:
            primary = _validate_variant(f_primary.result(timeout=30), sub_q_text, "primary")
        except Exception as exc:
            logger.warning("Primary variant failed: %s", exc)
        try:
            hyde = _validate_variant(f_hyde.result(timeout=30), sub_q_text, "hyde")
        except Exception as exc:
            logger.warning("HyDE variant failed: %s", exc)
        try:
            stepback = _validate_variant(f_stepback.result(timeout=30), sub_q_text, "stepback")
        except Exception as exc:
            logger.warning("Stepback variant failed: %s", exc)

    return {"primary": primary, "hyde_passage": hyde, "stepback": stepback}


# ── Node ───────────────────────────────────────────────────────────────────────

def decomposer_node(state: ComplianceState) -> dict:
    """Decompose the analyzed query into sub-questions; generate retrieval variants."""
    query = state["query"]
    analyzed = state.get("analyzed_query", {})
    entities = analyzed.get("entities", {})

    user_content = (
        f"Original question: {query}\n"
        f"Intent type: {analyzed.get('intent_type', 'compliance_check')}\n"
        f"is_multi_part: {analyzed.get('is_multi_part', True)}\n"
        f"Product type: {entities.get('product_type')}\n"
        f"Claim type: {entities.get('claim_type')}\n"
        f"FDA domain: {entities.get('fda_domain')}\n"
        f"Explicit CFR refs mentioned: {entities.get('explicit_refs', [])}"
    )

    messages = [
        {"role": "system", "content": _DECOMPOSE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw = llm_completion_json(messages, max_tokens=512, temperature=0.1)
    try:
        decomp = parse_llm_json(raw, messages)
        raw_sqs = decomp.get("sub_questions", [])
    except ValueError as exc:
        logger.warning("Decomposer parse failed: %s", exc)
        raw_sqs = []

    sub_q_texts = [sq["text"] for sq in raw_sqs if sq.get("text")]
    if not sub_q_texts:
        sub_q_texts = [query]

    # Generate variants in parallel across sub-questions
    sub_questions: list[dict] = []
    ordered_texts = sub_q_texts  # preserve original order
    futures = {}
    with ThreadPoolExecutor(max_workers=min(4, len(ordered_texts))) as ex:
        for text in ordered_texts:
            futures[ex.submit(_generate_variants, text)] = text

        results: dict[str, dict] = {}
        for future in as_completed(futures):
            text = futures[future]
            try:
                results[text] = future.result(timeout=90)
            except Exception as exc:
                logger.warning("Variant generation failed for '%s': %s", text[:50], exc)
                results[text] = {"primary": text, "hyde_passage": text, "stepback": text}

    for text in ordered_texts:
        sub_questions.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "variants": results.get(text, {"primary": text, "hyde_passage": text, "stepback": text}),
            "source_query": query,
        })

    sq_labels = ", ".join(f"SQ-{i+1}: '{sq['text'][:60]}'" for i, sq in enumerate(sub_questions))
    logger.info("[decomposer] %d sub-question(s) created — %s", len(sub_questions), sq_labels)

    session_id = state.get("session_id", "")
    if session_id:
        sl = get_session(session_id)
        if sl:
            sl.log_decomposition(sub_questions)

    return {"sub_questions": sub_questions}
