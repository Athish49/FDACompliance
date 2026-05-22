"""Per-sub-question retrieval pipeline — invoked for each sub-question via LangGraph fan-out.

Pipeline stages (per sub-question):
  1. Encode query variants (primary, HyDE, stepback) with BGE-M3
  2. Flow A — CFR Part-routed retrieval (dense + sparse per variant, part filter)
  3. Flow B — Global retrieval (dense + sparse per variant, no filter)
  4. RRF merge across all variant/flow ranked lists
  5. Cross-encoder reranking → top 12 (RERANKER_TOP_K)
  6. CRAG evaluation → CORRECT / AMBIGUOUS / INCORRECT / LOW_CONFIDENCE
     - AMBIGUOUS: supplemental global search, re-merge, re-rank
     - INCORRECT: reformulate and retry (max 3 attempts, escalating strategies)
  7. Cross-reference resolution
     - Layer 1: pre-extracted cross_references_internal list from payloads
     - Layer 2: LLM extraction of implicit references
     - Qdrant scroll fetch by section_number (depth 1 + depth 2)
  8. Draft answer generation (preliminary LLM call)
  9. Claim-level grounding verification (claim extraction → embedding → cosine similarity)
 10. Sub-answer synthesis (final structured SubAnswer)

Note: HyPE (index-time question vectors) is not implemented — those named vectors
do not exist in the current Qdrant collection. HyDE (query-time) is fully implemented.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Optional

from agents.llm import llm_completion, llm_completion_json, parse_llm_json
from agents.session_logger import get_session
from agents.state import ComplianceState

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

RRF_K = 60
FLOW_A_TOP_N = 40
FLOW_B_TOP_N = 40
TARGET_CANDIDATE_POOL = 80
RERANKER_TOP_K = 12
CRAG_MAX_RETRIES = 3
CRAG_THRESHOLD_HIGH = 0.70       # Min top score for standard CORRECT path
CRAG_THRESHOLD_LOW = 0.45        # Min score for AMBIGUOUS / supporting chunk
CRAG_THRESHOLD_VERY_HIGH = 0.85  # Relaxed CORRECT: only 1 supporting chunk needed
CRAG_AMBIGUOUS_SOFT_CORRECT = 0.70  # AMBIGUOUS with top >= this → skip supplemental
CRAG_BEST_CHUNK_MIN_SCORE = 0.30    # Min top_score to update best_chunks_seen
PART_CLASSIFIER_CONFIDENCE_MIN = 0.60

CLAIM_SIMILARITY_THRESHOLD = 0.65
CLAIM_KEYWORD_OVERLAP_THRESHOLD = 0.50

# ── Shared retriever instance (thread-safe lazy init) ─────────────────────────

_retriever = None
_retriever_lock = threading.Lock()


def _get_retriever():
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                from retrieval.retriever import CFRRetriever, RetrieverConfig
                _retriever = CFRRetriever(RetrieverConfig())
    return _retriever


# ── CFR Part index + domain filter ────────────────────────────────────────────

_PART_INDEX_PATH = Path(__file__).parent.parent / "data" / "cfr_part_index.json"

_part_index_cache: dict | None = None
_part_index_lock = threading.Lock()

# Maps fda_domain values (from query analyzer) to the subchapter letters that
# contain relevant parts. Used to pre-filter the part list before sending to LLM.
_DOMAIN_TO_SUBCHAPTERS: dict[str, set[str]] = {
    "food":           {"B"},
    "drug":           {"C", "D"},
    "animal":         {"E"},
    "biological":     {"F"},
    "cosmetic":       {"G"},
    "device":         {"H", "I", "J"},
    "tobacco":        {"K"},
    "administrative": {"A", "L"},
}


def _load_part_index() -> dict:
    """Lazy-load and cache the CFR part index. Returns empty dict if file absent."""
    global _part_index_cache
    if _part_index_cache is None:
        with _part_index_lock:
            if _part_index_cache is None:
                if _PART_INDEX_PATH.exists():
                    with open(_PART_INDEX_PATH, encoding="utf-8") as fh:
                        _part_index_cache = json.load(fh).get("parts", {})
                    logger.info(
                        "[part_classifier] Loaded %d parts from %s",
                        len(_part_index_cache),
                        _PART_INDEX_PATH,
                    )
                else:
                    logger.warning(
                        "[part_classifier] %s not found — run ingestion to generate it",
                        _PART_INDEX_PATH,
                    )
                    _part_index_cache = {}
    return _part_index_cache


def _build_parts_context(fda_domain: str | None) -> str:
    """
    Return a compact text list of CFR parts filtered to the relevant domain.
    If fda_domain is None or unrecognised, returns all parts (no filter).
    Returns empty string if the index file hasn't been generated yet.
    """
    parts = _load_part_index()
    if not parts:
        return ""

    target_subchapters: set[str] = set()
    if fda_domain:
        target_subchapters = _DOMAIN_TO_SUBCHAPTERS.get(fda_domain.lower(), set())

    lines: list[str] = []
    for num, info in parts.items():
        if target_subchapters and info.get("subchapter") not in target_subchapters:
            continue
        lines.append(f"Part {num}: {info['title']}")

    if not lines:
        # Domain filter matched nothing — fall back to full list
        logger.debug(
            "[part_classifier] Domain filter '%s' matched no parts — using full index",
            fda_domain,
        )
        lines = [f"Part {num}: {info['title']}" for num, info in parts.items()]

    return "\n".join(lines)


# ── CFR Part classifier ────────────────────────────────────────────────────────

_PART_CLASSIFIER_SYSTEM_BASE = """\
Given the following FDA compliance sub-question, predict which Title 21 CFR Part number(s) are most
likely to contain the relevant regulations.  Return ONLY valid JSON:
{"parts": ["101", "102"], "confidence": 0.85}
- confidence is 0.0-1.0 reflecting your certainty
- parts is a list of numeric strings (without "Part" prefix)
- If unsure, return confidence < 0.60 and parts may be empty
- Hint sections (if provided below) come from the original user query — treat as context only.
  If this sub-question covers a different topic, classify based on the sub-question alone."""


def _build_classifier_messages(
    sub_question_text: str,
    explicit_refs: list[str],
    fda_domain: str | None,
) -> list[dict]:
    """Build the classifier prompt, injecting filtered part context when available."""
    parts_context = _build_parts_context(fda_domain)

    system = _PART_CLASSIFIER_SYSTEM_BASE
    if parts_context:
        system += f"\n\nAvailable CFR Parts in this regulatory domain:\n{parts_context}"

    user_content = sub_question_text
    if explicit_refs:
        user_content += f"\n\nHint — original query mentioned these CFR sections: {explicit_refs}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _classify_cfr_parts(
    sub_question_text: str,
    explicit_refs: list[str],
    fda_domain: str | None = None,
) -> tuple[list[str], float]:
    """Return (predicted_part_numbers, confidence). Empty list = skip Flow A.

    Uses a domain-filtered view of the CFR part index so the LLM receives
    only parts relevant to the query's regulatory area (~15-60 parts vs 256).
    explicit_refs are hints, not overrides — the LLM classifies based on the
    sub-question content and can ignore the hints if they are off-topic.
    """
    messages = _build_classifier_messages(sub_question_text, explicit_refs, fda_domain)
    try:
        raw = llm_completion_json(messages, max_tokens=128, temperature=0.1)
        result = parse_llm_json(raw, messages)
        parts = [str(p) for p in result.get("parts", [])]
        confidence = float(result.get("confidence", 0.0))
        return parts, confidence
    except Exception as exc:
        logger.warning("Part classifier failed: %s", exc)
        return [], 0.0


# ── Encoding ───────────────────────────────────────────────────────────────────

def _encode_texts(texts: list[str]) -> list[tuple[list[float], dict]]:
    """Batch-encode texts with BGE-M3.  Returns [(dense_vec, sparse_dict), ...]."""
    retriever = _get_retriever()
    output = retriever.model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    results = []
    for i in range(len(texts)):
        dense_vec = output["dense_vecs"][i].tolist()
        weights = output["lexical_weights"][i]
        sparse_dict = {
            "indices": [int(k) for k in sorted(weights.keys())],
            "values": [float(weights[k]) for k in sorted(weights.keys())],
        }
        results.append((dense_vec, sparse_dict))
    return results


# ── Qdrant search helpers ──────────────────────────────────────────────────────

def _search_qdrant(
    dense_vec: list[float],
    sparse_dict: dict,
    qdrant_filter,
    top_n: int,
) -> list[tuple[str, float, dict]]:
    """Run dense + sparse search and return merged list of (chunk_id, score, payload)."""
    retriever = _get_retriever()
    dense = retriever.search_dense(dense_vec, qdrant_filter, top_n)
    sparse = retriever.search_sparse(sparse_dict, qdrant_filter, top_n)
    # Simple merge by score normalisation — higher is better in both
    seen: dict[str, tuple[float, dict]] = {}
    for pid, score, payload in dense + sparse:
        if pid not in seen or score > seen[pid][0]:
            seen[pid] = (score, payload)
    # Sort by score descending, take top_n
    sorted_items = sorted(seen.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
    return [(pid, score, payload) for pid, (score, payload) in sorted_items]


def _build_part_filter(part_numbers: list[str]):
    """Build a Qdrant filter restricting to given part numbers."""
    if not part_numbers:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    return Filter(must=[
        FieldCondition(key="part_number", match=MatchAny(any=part_numbers))
    ])


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge(
    ranked_lists: list[list[tuple[str, float, dict]]],
    k: int = RRF_K,
    top_n: int = TARGET_CANDIDATE_POOL,
) -> list[tuple[str, float, dict]]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, (pid, _score, payload) in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            payloads[pid] = payload
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(pid, scores[pid], payloads[pid]) for pid, _ in sorted_items]


# ── Reranking ──────────────────────────────────────────────────────────────────

def _rerank_candidates(
    query_text: str,
    candidates: list[tuple[str, float, dict]],
    top_k: int = RERANKER_TOP_K,
) -> list[dict]:
    """Cross-encoder rerank.  Returns list of enriched chunk dicts sorted by reranker_score."""
    if not candidates:
        return []
    retriever = _get_retriever()
    pairs = [[query_text, cand[2].get("text", "")] for cand in candidates]
    try:
        reranker_scores = retriever.reranker.compute_score(pairs, normalize=True)
    except Exception as exc:
        logger.warning("Reranker failed: %s — falling back to RRF order", exc)
        reranker_scores = [cand[1] for cand in candidates]

    if isinstance(reranker_scores, float):
        reranker_scores = [reranker_scores]

    scored = []
    for cand, rs in zip(candidates, reranker_scores):
        pid, rrf_score, payload = cand
        chunk = {**payload, "chunk_id": payload.get("chunk_id", pid), "rrf_score": rrf_score, "reranker_score": float(rs)}
        scored.append(chunk)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return scored[:top_k]


# ── CRAG Evaluator ─────────────────────────────────────────────────────────────

def _crag_verdict(reranked_chunks: list[dict]) -> str:
    """Determine CRAG verdict from reranker scores.

    Two CORRECT paths:
    - Very high top score (>= CRAG_THRESHOLD_VERY_HIGH): only 1 supporting chunk needed.
    - Standard high top score (>= CRAG_THRESHOLD_HIGH): requires 2+ supporting chunks.
    """
    if not reranked_chunks:
        return "INCORRECT"
    top_score = reranked_chunks[0]["reranker_score"]
    above_low = sum(1 for c in reranked_chunks if c["reranker_score"] >= CRAG_THRESHOLD_LOW)

    if top_score >= CRAG_THRESHOLD_VERY_HIGH and above_low >= 1:
        return "CORRECT"
    if top_score >= CRAG_THRESHOLD_HIGH and above_low >= 2:
        return "CORRECT"
    if top_score >= CRAG_THRESHOLD_LOW:
        return "AMBIGUOUS"
    return "INCORRECT"


_REFORMULATE_SYNONYM_SYSTEM = """\
Rewrite the following FDA regulatory query with expanded regulatory synonyms and alternative
terminology.  Keep it concise and keyword-rich.  Output only the rewritten query."""

_REFORMULATE_REPHRASE_SYSTEM = """\
Rewrite the following FDA regulatory query from a completely different angle — focus on the
underlying requirement rather than the specific claim.  Output only the rewritten query."""

_REFORMULATE_MIN_LEN = 15


def _extract_best_section(best_chunks_seen: list[dict]) -> str:
    """Extract a CFR section number from the best chunks seen across retries.

    Used by the section anchor strategy: even low-scoring chunks carry a real
    section number that can be appended to the query to boost sparse retrieval
    toward the correct regulatory area.
    """
    for chunk in best_chunks_seen[:3]:
        sec = (chunk.get("hierarchy") or {}).get("section", {}).get("number", "")
        if sec:
            return sec
        citation = chunk.get("cfr_citation", "")
        if citation:
            m = re.search(r"(\d+\.\d+)", citation)
            if m:
                return m.group(1)
    return ""


def _reformulate_query(
    sub_q_text: str,
    retry_count: int,
    variants: dict,
    best_chunks_seen: list[dict] | None = None,
) -> str:
    """Return a reformulated query string based on retry strategy.

    Strategies by retry_count:
      0 — synonym expansion (LLM call)
      1 — full rephrase from different angle (LLM call)
      2 — section anchor (append best-seen CFR section) or HyDE/stepback fallback
    """
    if retry_count == 0:
        messages = [
            {"role": "system", "content": _REFORMULATE_SYNONYM_SYSTEM},
            {"role": "user", "content": sub_q_text},
        ]
        try:
            return llm_completion(messages, max_tokens=128, temperature=0.2)
        except Exception:
            return sub_q_text
    elif retry_count == 1:
        messages = [
            {"role": "system", "content": _REFORMULATE_REPHRASE_SYSTEM},
            {"role": "user", "content": sub_q_text},
        ]
        try:
            return llm_completion(messages, max_tokens=128, temperature=0.3)
        except Exception:
            return sub_q_text
    else:
        # Strategy 2: section anchor — use the top chunk's CFR section from the
        # best previous attempt to anchor sparse retrieval toward the right area.
        if best_chunks_seen:
            sec = _extract_best_section(best_chunks_seen)
            if sec:
                anchored = f"{sub_q_text} 21 CFR §{sec}"
                logger.info("[crag] section anchor strategy — appending §%s to query", sec)
                return anchored

        # Fallback: HyDE passage or stepback variant (pre-generated at decomposition)
        hyde = variants.get("hyde_passage", "")
        stepback = variants.get("stepback", "")
        if hyde and hyde != sub_q_text and len(hyde.strip()) >= _REFORMULATE_MIN_LEN:
            return hyde
        if stepback and len(stepback.strip()) >= _REFORMULATE_MIN_LEN:
            return stepback
        return sub_q_text


# ── Cross-Reference Resolution ─────────────────────────────────────────────────

_IMPLICIT_REF_SYSTEM = """\
Identify any implicit references to other FDA CFR sections in the following text.  Look for phrases
like "as defined in", "see also", "pursuant to", "in accordance with", "determined under",
"as specified in".  Return ONLY valid JSON: {"sections": ["101.54", "101.9"]}
If none, return {"sections": []}"""


def _extract_implicit_refs(chunk_text: str) -> list[str]:
    messages = [
        {"role": "system", "content": _IMPLICIT_REF_SYSTEM},
        {"role": "user", "content": chunk_text[:2000]},
    ]
    try:
        raw = llm_completion_json(messages, max_tokens=256, temperature=0.1)
        result = parse_llm_json(raw, messages)
        return [str(s) for s in result.get("sections", [])]
    except Exception as exc:
        logger.debug("Implicit ref extraction failed: %s", exc)
        return []


def _fetch_chunks_by_section(section_numbers: list[str], already_fetched: set[str]) -> list[dict]:
    """Fetch chunks by section_number filter via Qdrant scroll."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    retriever = _get_retriever()
    fetched = []
    for sec_num in section_numbers:
        if sec_num in already_fetched:
            continue
        try:
            results, _ = retriever.client.scroll(
                collection_name=retriever.config.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="section_number", match=MatchValue(value=sec_num))]
                ),
                limit=10,
                with_payload=True,
            )
            for point in results:
                p = point.payload
                chunk = {**p, "chunk_id": p.get("chunk_id"), "is_cross_ref": True, "reranker_score": 0.0, "rrf_score": 0.0}
                fetched.append(chunk)
            already_fetched.add(sec_num)
        except Exception as exc:
            logger.debug("Cross-ref fetch failed for §%s: %s", sec_num, exc)
    return fetched


def _resolve_cross_references(
    primary_chunks: list[dict],
    already_fetched: set[str],
    max_depth: int = 2,
) -> list[dict]:
    """Layer 1 (pre-extracted) + Layer 2 (LLM implicit) cross-reference expansion, depth 1+2."""
    all_cross_ref_chunks: list[dict] = []

    # ── Depth 1 ───────────────────────────────────────────────────────────────
    depth1_sections: set[str] = set()

    for chunk in primary_chunks:
        # Layer 1: pre-extracted cross_references_internal
        for ref in chunk.get("cross_references_internal", []):
            if isinstance(ref, str) and ref.strip():
                depth1_sections.add(ref.strip())

        # Layer 2: LLM implicit references
        text = chunk.get("text", "")
        if text:
            implicit = _extract_implicit_refs(text)
            depth1_sections.update(implicit)

    depth1_chunks = _fetch_chunks_by_section(list(depth1_sections), already_fetched)
    for c in depth1_chunks:
        c["cross_ref_depth"] = 1
    all_cross_ref_chunks.extend(depth1_chunks)

    if max_depth < 2 or not depth1_chunks:
        return all_cross_ref_chunks

    # ── Depth 2 ───────────────────────────────────────────────────────────────
    depth2_sections: set[str] = set()
    for chunk in depth1_chunks:
        for ref in chunk.get("cross_references_internal", []):
            if isinstance(ref, str) and ref.strip():
                depth2_sections.add(ref.strip())

    depth2_chunks = _fetch_chunks_by_section(list(depth2_sections), already_fetched)
    for c in depth2_chunks:
        c["cross_ref_depth"] = 2
    all_cross_ref_chunks.extend(depth2_chunks)

    return all_cross_ref_chunks


# ── Draft Answer Generation ────────────────────────────────────────────────────

_DRAFT_ANSWER_SYSTEM = """\
You are an FDA regulatory compliance expert.  Based ONLY on the provided CFR excerpts, write a
concise draft answer to the compliance question.  Include only facts present in the excerpts.
Do NOT add disclaimers or fabricate requirements.  Keep it to 2-4 sentences."""


def _generate_draft_answer(sub_question_text: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks[:12]):
        citation = c.get("cfr_citation", "")
        text = c.get("text", "")
        preamble = c.get("section_preamble", "")
        if preamble and preamble not in text:
            text = f"{preamble} {text}"
        context_parts.append(f"[{i+1}] {citation}: {text[:500]}")

    context = "\n\n".join(context_parts)
    messages = [
        {"role": "system", "content": _DRAFT_ANSWER_SYSTEM},
        {"role": "user", "content": f"Question: {sub_question_text}\n\nCFR Excerpts:\n{context}"},
    ]
    try:
        return llm_completion(messages, max_tokens=512, temperature=0.1)
    except Exception as exc:
        logger.warning("Draft answer generation failed: %s", exc)
        return ""


# ── Claim-Level Grounding Verification ────────────────────────────────────────

_CLAIM_EXTRACT_SYSTEM = """\
Extract every distinct factual regulatory claim from the following text as a JSON array of strings.
Each item must be a single atomic statement of fact.
Do NOT include opinions, uncertainty language, or procedural steps.
Return ONLY valid JSON: {"claims": ["claim 1", "claim 2"]}"""


def _extract_claims(answer_text: str) -> list[str]:
    if not answer_text.strip():
        return []
    messages = [
        {"role": "system", "content": _CLAIM_EXTRACT_SYSTEM},
        {"role": "user", "content": answer_text},
    ]
    try:
        raw = llm_completion_json(messages, max_tokens=512, temperature=0.1)
        result = parse_llm_json(raw, messages)
        return [str(c) for c in result.get("claims", []) if c]
    except Exception as exc:
        logger.warning("Claim extraction failed: %s", exc)
        return []


def _keyword_overlap(claim: str, chunk_text: str) -> float:
    """Simple unigram overlap ratio."""
    claim_words = set(re.findall(r"\b\w{4,}\b", claim.lower()))
    chunk_words = set(re.findall(r"\b\w{4,}\b", chunk_text.lower()))
    if not claim_words:
        return 0.0
    return len(claim_words & chunk_words) / len(claim_words)


def _verify_claims(
    claims: list[str],
    resolved_chunks: list[dict],
) -> tuple[list[dict], list[str]]:
    """
    For each claim, find the best-matching chunk via keyword overlap.
    Returns (claim_verification_list, unverified_claims).
    """
    if not claims or not resolved_chunks:
        return [], claims[:]

    chunk_texts = [c.get("text", "") for c in resolved_chunks]

    claim_results: list[dict] = []
    unverified: list[str] = []

    for claim in claims:
        best_score = 0.0
        best_chunk = None

        for chunk, text in zip(resolved_chunks, chunk_texts):
            overlap = _keyword_overlap(claim, text)
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk

        supported = best_score >= CLAIM_KEYWORD_OVERLAP_THRESHOLD

        if not supported and best_chunk is not None:
            # Try cosine similarity for borderline cases
            try:
                claim_enc = _encode_texts([claim])
                chunk_enc = _encode_texts([best_chunk.get("text", "")])
                if claim_enc and chunk_enc:
                    cv = claim_enc[0][0]
                    tv = chunk_enc[0][0]
                    dot = sum(a * b for a, b in zip(cv, tv))
                    nc = (sum(a * a for a in cv) ** 0.5) or 1.0
                    nt = (sum(a * a for a in tv) ** 0.5) or 1.0
                    cosine = dot / (nc * nt)
                    supported = cosine >= CLAIM_SIMILARITY_THRESHOLD
            except Exception:
                pass

        claim_result = {
            "claim_text": claim,
            "is_supported": supported,
            "supporting_chunk_id": best_chunk.get("chunk_id") if supported and best_chunk else None,
            "supporting_cfr_section": (
                best_chunk.get("hierarchy", {}).get("section", {}).get("number")
                if supported and best_chunk else None
            ),
        }
        claim_results.append(claim_result)

        if not supported:
            unverified.append(claim)

    return claim_results, unverified


# ── Sub-Answer Synthesis ───────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are an FDA regulatory compliance expert.  Write a structured sub-answer to the compliance
question using ONLY the provided verified CFR excerpts.

Format the answer according to the intent type:
- compliance_check → numbered requirements list with inline [§X.XX] citations
- definition        → bold defined term, then formal regulatory definition
- procedure        → numbered step-by-step procedure with citations
- penalty          → description of violations and penalties with citations

Rules:
- Every factual sentence MUST end with [§X.XX] citing the source section
- Do not fabricate requirements not present in the excerpts
- Do not add disclaimers
- Be specific and cite the actual CFR text

Return ONLY valid JSON:
{
  "answer": "formatted answer with inline citations",
  "citations": [
    {"cfr_section": "21 CFR 101.54", "text_snippet": "relevant excerpt..."}
  ]
}"""


def _synthesize_sub_answer(
    sub_question_text: str,
    intent_type: str,
    verified_answer: str,
    resolved_chunks: list[dict],
    claim_verification: list[dict],
) -> tuple[str, list[dict]]:
    """Generate the final structured sub-answer text and citations list."""
    # Build context from verified claims' supporting chunks
    supporting_ids = {
        cv["supporting_chunk_id"]
        for cv in claim_verification
        if cv.get("is_supported") and cv.get("supporting_chunk_id")
    }
    evidence_chunks = [c for c in resolved_chunks if c.get("chunk_id") in supporting_ids]
    if not evidence_chunks:
        evidence_chunks = resolved_chunks[:8]

    context_parts = []
    for c in evidence_chunks[:10]:
        sec = (c.get("hierarchy", {}).get("section", {}) or {})
        sec_num = sec.get("number", "")
        citation = c.get("cfr_citation", f"21 CFR {sec_num}" if sec_num else "")
        text = c.get("text", "")
        context_parts.append(f"Section {citation}:\n{text[:600]}")

    context = "\n\n".join(context_parts)
    user_content = (
        f"Intent type: {intent_type}\n"
        f"Sub-question: {sub_question_text}\n\n"
        f"Verified draft: {verified_answer}\n\n"
        f"CFR Excerpts:\n{context}"
    )

    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = llm_completion_json(messages, max_tokens=1024, temperature=0.1)
        result = parse_llm_json(raw, messages)
        answer = result.get("answer", verified_answer)
        citations = result.get("citations", [])
        return answer, citations
    except Exception as exc:
        logger.warning("Sub-answer synthesis failed: %s", exc)
        return verified_answer, []


# ── Confidence Scoring ─────────────────────────────────────────────────────────

def _compute_confidence(crag_verdict: str, unverified_claims: list[str]) -> float:
    table = {
        ("CORRECT", 0): 1.0,
        ("CORRECT", 1): 0.8,
        ("AMBIGUOUS", 0): 0.6,
        ("AMBIGUOUS", 1): 0.4,
        ("LOW_CONFIDENCE", 1): 0.2,
    }
    has_unverified = 1 if unverified_claims else 0
    key = (crag_verdict if crag_verdict in ("CORRECT", "AMBIGUOUS") else "LOW_CONFIDENCE", has_unverified)
    return table.get(key, 0.2)


# ── Main per-sub-question orchestrator ────────────────────────────────────────

def run_sub_question_pipeline(
    query: str,
    analyzed_query: dict,
    sub_question: dict,
    session_id: str = "",
) -> dict:
    """
    Full per-sub-question retrieval, evaluation, and synthesis pipeline.
    Returns a SubAnswer dict.
    """
    sub_q_id = sub_question["id"]
    sub_q_text = sub_question["text"]
    variants = sub_question.get("variants", {})
    intent_type = analyzed_query.get("intent_type", "compliance_check")
    entities = analyzed_query.get("entities", {})
    explicit_refs = entities.get("explicit_refs", [])
    fda_domain = entities.get("fda_domain")

    sl = get_session(session_id) if session_id else None
    sq_label = sl.get_sq_label(sub_q_id) if sl else sub_q_id[:8]

    logger.info("[retrieval] Starting pipeline for %s — '%s'", sq_label, sub_q_text[:100])

    reformulation_log: list[str] = []
    crag_verdict = "INCORRECT"
    reranked_chunks: list[dict] = []
    best_chunks_seen: list[dict] = []   # best reranked result across all retries
    caveat_flag = False

    current_primary = variants.get("primary", sub_q_text)

    for retry_count in range(CRAG_MAX_RETRIES + 1):
        # ── 1. Encode variants ────────────────────────────────────────────────
        variant_texts = [
            current_primary,
            variants.get("hyde_passage", current_primary),
            variants.get("stepback", current_primary),
        ]
        try:
            encodings = _encode_texts(variant_texts)
        except Exception as exc:
            logger.error("Encoding failed: %s", exc)
            encodings = [([], {}) for _ in variant_texts]

        # ── 2. Flow A — Routed retrieval ──────────────────────────────────────
        flow_a_lists: list[list[tuple[str, float, dict]]] = []
        parts, part_confidence = _classify_cfr_parts(sub_q_text, explicit_refs, fda_domain)

        if parts and part_confidence >= PART_CLASSIFIER_CONFIDENCE_MIN:
            part_filter = _build_part_filter(parts)
            logger.debug("Flow A: parts=%s, confidence=%.2f", parts, part_confidence)
            with ThreadPoolExecutor(max_workers=3) as ex:
                futures_a = []
                for enc in encodings:
                    dense_vec, sparse_dict = enc
                    if dense_vec:
                        futures_a.append(ex.submit(_search_qdrant, dense_vec, sparse_dict, part_filter, FLOW_A_TOP_N))
                for f in futures_a:
                    try:
                        flow_a_lists.append(f.result(timeout=30))
                    except Exception as exc:
                        logger.warning("Flow A search failed: %s", exc)
        else:
            logger.debug("Flow A skipped: confidence=%.2f", part_confidence)

        # ── 3. Flow B — Global retrieval ──────────────────────────────────────
        flow_b_lists: list[list[tuple[str, float, dict]]] = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures_b = []
            for enc in encodings:
                dense_vec, sparse_dict = enc
                if dense_vec:
                    futures_b.append(ex.submit(_search_qdrant, dense_vec, sparse_dict, None, FLOW_B_TOP_N))
            for f in futures_b:
                try:
                    flow_b_lists.append(f.result(timeout=30))
                except Exception as exc:
                    logger.warning("Flow B search failed: %s", exc)

        # ── 4. RRF merge ──────────────────────────────────────────────────────
        all_lists = flow_a_lists + flow_b_lists
        if not all_lists:
            logger.warning("No search results for sub-question: %s", sub_q_text[:80])
            break

        fused = _rrf_merge(all_lists, k=RRF_K, top_n=TARGET_CANDIDATE_POOL)

        # ── 5. Rerank ─────────────────────────────────────────────────────────
        reranked_chunks = _rerank_candidates(sub_q_text, fused, top_k=RERANKER_TOP_K)

        # ── 6. CRAG evaluation ────────────────────────────────────────────────
        crag_verdict = _crag_verdict(reranked_chunks)
        top_score = reranked_chunks[0]["reranker_score"] if reranked_chunks else 0.0
        reformulation_used = reformulation_log[-1] if reformulation_log and retry_count > 0 else None
        logger.info(
            "[crag] %s | attempt=%d | verdict=%s | top_score=%.3f | chunks=%d",
            sq_label, retry_count + 1, crag_verdict, top_score, len(reranked_chunks),
        )
        if sl:
            sl.log_sq_attempt(
                sq_id=sub_q_id,
                attempt_num=retry_count,
                crag_verdict=crag_verdict,
                top_score=top_score,
                retrieved_chunks=reranked_chunks,
                reformulation=reformulation_used,
            )

        # ── Track best chunks across all attempts ─────────────────────────────
        if reranked_chunks and top_score > CRAG_BEST_CHUNK_MIN_SCORE:
            if not best_chunks_seen or top_score > best_chunks_seen[0]["reranker_score"]:
                best_chunks_seen = reranked_chunks[:]

        if crag_verdict == "CORRECT":
            caveat_flag = False
            break

        if crag_verdict == "AMBIGUOUS":
            caveat_flag = True
            if top_score >= CRAG_AMBIGUOUS_SOFT_CORRECT:
                # Top chunk is already strong — supplemental search risks introducing
                # noise. Treat as soft-CORRECT and proceed.
                logger.info(
                    "[crag] %s | AMBIGUOUS soft-CORRECT (top_score=%.3f >= %.2f) — skipping supplemental",
                    sq_label, top_score, CRAG_AMBIGUOUS_SOFT_CORRECT,
                )
                break
            # Genuinely ambiguous: run supplemental broad search to augment pool
            enc = encodings[0]
            if enc[0]:
                try:
                    supplemental = _search_qdrant(enc[0], enc[1], None, FLOW_B_TOP_N)
                    fused2 = _rrf_merge([fused, supplemental], k=RRF_K, top_n=TARGET_CANDIDATE_POOL)
                    reranked_chunks = _rerank_candidates(sub_q_text, fused2, top_k=RERANKER_TOP_K)
                except Exception as exc:
                    logger.warning("Supplemental search failed: %s", exc)
            break

        if crag_verdict == "INCORRECT":
            if retry_count >= CRAG_MAX_RETRIES:
                crag_verdict = "LOW_CONFIDENCE"
                caveat_flag = True
                # Use best chunks seen if they scored higher than the current attempt
                if best_chunks_seen:
                    current_top = reranked_chunks[0]["reranker_score"] if reranked_chunks else 0.0
                    if best_chunks_seen[0]["reranker_score"] > current_top:
                        reranked_chunks = best_chunks_seen
                        logger.info(
                            "[crag] %s | LOW_CONFIDENCE — restoring best_chunks_seen (top=%.3f)",
                            sq_label, reranked_chunks[0]["reranker_score"],
                        )
                if not reranked_chunks:
                    reranked_chunks = fused[:RERANKER_TOP_K]
                break
            # Reformulate and retry
            new_primary = _reformulate_query(sub_q_text, retry_count, variants, best_chunks_seen)
            # Validate reformulation: reject empty, too-short, or identical to current query
            if (
                not new_primary
                or len(new_primary.strip()) < _REFORMULATE_MIN_LEN
                or new_primary.strip() == current_primary.strip()
            ):
                logger.warning(
                    "[crag] %s | reformulation invalid (got %r) — keeping previous query",
                    sq_label, (new_primary or "")[:60],
                )
                new_primary = current_primary
            reformulation_log.append(new_primary)
            logger.info(
                "[crag] %s | reformulating for attempt %d → '%s'",
                sq_label, retry_count + 2, new_primary[:80],
            )
            current_primary = new_primary

    # ── Skip synthesis when retrieval found nothing usable ────────────────────
    # Condition: exhausted all retries (LOW_CONFIDENCE) AND best_chunks_seen is
    # empty (no attempt ever scored above CRAG_BEST_CHUNK_MIN_SCORE=0.30).
    # These sub-answers are pure noise — they drag down confidence averaging and
    # inject irrelevant citations into the final answer. Return immediately.
    if crag_verdict == "LOW_CONFIDENCE" and not best_chunks_seen:
        logger.warning(
            "[retrieval] %s | SKIPPED — LOW_CONFIDENCE with no usable chunks found across all retries",
            sq_label,
        )
        skipped_sub_answer = {
            "sub_question_id": sub_q_id,
            "sub_question_text": sub_q_text,
            "answer": "",
            "confidence": 0.0,
            "crag_verdict": "LOW_CONFIDENCE",
            "caveat_flag": True,
            "chunks_used": [],
            "cross_refs_resolved": [],
            "claim_verification": [],
            "unverified_claims": [],
            "citations": [],
            "caveats": ["LOW_CONFIDENCE_RETRIEVAL"],
            "flags": ["MAX_RETRIES_EXCEEDED", "SKIPPED_NO_EVIDENCE"],
            "reformulation_log": reformulation_log,
            "skipped": True,
        }
        if sl:
            sl.log_sq_answer(
                sq_id=sub_q_id,
                crag_verdict="LOW_CONFIDENCE",
                confidence=0.0,
                citations_count=0,
                answer_summary="SKIPPED — no usable chunks found",
                caveats=["LOW_CONFIDENCE_RETRIEVAL"],
                flags=["MAX_RETRIES_EXCEEDED", "SKIPPED_NO_EVIDENCE"],
            )
        return {"sub_answers": [skipped_sub_answer]}

    # ── 7. Cross-reference resolution ─────────────────────────────────────────
    already_fetched: set[str] = {
        c.get("hierarchy", {}).get("section", {}).get("number", "")
        for c in reranked_chunks
        if c.get("hierarchy", {}).get("section", {}).get("number")
    }
    cross_ref_chunks = _resolve_cross_references(reranked_chunks, already_fetched, max_depth=2)

    resolved_chunks = reranked_chunks + cross_ref_chunks
    cross_refs_resolved = [
        c.get("hierarchy", {}).get("section", {}).get("number", "")
        for c in cross_ref_chunks
        if c.get("hierarchy", {}).get("section", {}).get("number")
    ]

    # ── 8. Draft answer ────────────────────────────────────────────────────────
    draft_answer = _generate_draft_answer(sub_q_text, resolved_chunks)

    # ── 9. Claim-level grounding verification ──────────────────────────────────
    claims = _extract_claims(draft_answer)
    claim_verification, unverified_claims = _verify_claims(claims, resolved_chunks)

    # Prune unsupported claims from draft if most are unsupported
    all_unverified = len(unverified_claims) == len(claims) and claims
    verified_answer = draft_answer  # keep full draft; pruning is tracked via verification list

    # ── 10. Sub-answer synthesis ───────────────────────────────────────────────
    final_answer_text, citations = _synthesize_sub_answer(
        sub_q_text, intent_type, verified_answer, resolved_chunks, claim_verification
    )

    # ── Confidence ─────────────────────────────────────────────────────────────
    confidence = _compute_confidence(crag_verdict, unverified_claims)

    # ── Caveats & flags ────────────────────────────────────────────────────────
    caveats: list[str] = []
    flags: list[str] = []
    if caveat_flag:
        caveats.append("AMBIGUOUS_EVIDENCE" if crag_verdict == "AMBIGUOUS" else "LOW_CONFIDENCE_RETRIEVAL")
    if crag_verdict == "LOW_CONFIDENCE":
        flags.append("MAX_RETRIES_EXCEEDED")
    if unverified_claims:
        caveats.append("UNVERIFIED_CLAIMS_REMOVED")
    if all_unverified:
        flags.append("ALL_CLAIMS_UNVERIFIED")

    chunks_used = [
        c for c in resolved_chunks
        if c.get("chunk_id") in {cv.get("supporting_chunk_id") for cv in claim_verification if cv.get("is_supported")}
    ] or resolved_chunks[:8]

    sub_answer = {
        "sub_question_id": sub_q_id,
        "sub_question_text": sub_q_text,
        "answer": final_answer_text,
        "confidence": confidence,
        "crag_verdict": crag_verdict,
        "caveat_flag": caveat_flag,
        "chunks_used": [
            {
                "chunk_id": c.get("chunk_id"),
                "cfr_citation": c.get("cfr_citation"),
                "text": c.get("text", "")[:300],
                "reranker_score": c.get("reranker_score", 0.0),
            }
            for c in chunks_used
        ],
        "cross_refs_resolved": cross_refs_resolved,
        "claim_verification": claim_verification,
        "unverified_claims": unverified_claims,
        "citations": citations,
        "caveats": caveats,
        "flags": flags,
        "reformulation_log": reformulation_log,
    }

    logger.info(
        "[retrieval] %s complete | crag=%s | confidence=%.2f | citations=%d | cross_refs=%d",
        sq_label, crag_verdict, confidence, len(citations), len(cross_refs_resolved),
    )
    if sl:
        sl.log_sq_answer(
            sq_id=sub_q_id,
            crag_verdict=crag_verdict,
            confidence=confidence,
            citations_count=len(citations),
            answer_summary=final_answer_text,
            caveats=caveats,
            flags=flags,
        )
    return sub_answer


# ── LangGraph node ─────────────────────────────────────────────────────────────

def process_sub_question_node(state: dict) -> dict:
    """LangGraph node — receives a Send payload and processes one sub-question."""
    query = state["query"]
    analyzed_query = state.get("analyzed_query", {})
    sub_question = state["sub_question"]
    session_id = state.get("session_id", "")

    try:
        sub_answer = run_sub_question_pipeline(query, analyzed_query, sub_question, session_id)
    except Exception as exc:
        logger.exception("Sub-question pipeline failed for %s: %s", sub_question.get("id", "?"), exc)
        sub_answer = {
            "sub_question_id": sub_question.get("id", ""),
            "sub_question_text": sub_question.get("text", ""),
            "answer": "",
            "confidence": 0.0,
            "crag_verdict": "LOW_CONFIDENCE",
            "caveat_flag": True,
            "chunks_used": [],
            "cross_refs_resolved": [],
            "claim_verification": [],
            "unverified_claims": [],
            "citations": [],
            "caveats": ["PIPELINE_ERROR"],
            "flags": ["PIPELINE_ERROR"],
            "reformulation_log": [],
        }

    return {"sub_answers": [sub_answer]}
