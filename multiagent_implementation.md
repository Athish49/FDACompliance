# FDA Compliance AI — Multi-Agent Layer Implementation Reference

## Overview

The multi-agent reasoning layer is a LangGraph `StateGraph` that takes a natural language compliance question and runs it through a structured pipeline of 6 nodes to produce a grounded, cited, conflict-checked answer. Sub-questions are processed in parallel using LangGraph's `Send`-based fan-out pattern.

The pipeline is exposed via two FastAPI endpoints:
- `POST /api/query` — blocking, returns a `QueryResponse` JSON object
- `POST /api/query/stream` — SSE streaming, emits one event per completed node

---

## Top-Level Graph Flow

```
POST /api/query  {"question": "..."}
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph (ComplianceState)                              │
│                                                                      │
│  query_analyzer                                                      │
│       │                                                              │
│       ├─── needs_clarification=true ──→ clarification_response → END│
│       │                                                              │
│       └─── needs_clarification=false                                 │
│                    │                                                 │
│               decomposer                                             │
│                    │  (1–4 sub-questions, each with 3 variants)      │
│                    │                                                 │
│            [fan-out via Send]                                        │
│         ┌──────┬──────┬──────┐                                       │
│         ▼      ▼      ▼      ▼   (all in parallel)                   │
│       process_sub_question × N                                       │
│         └──────┴──────┴──────┘                                       │
│                    │  (fan-in via operator.add on sub_answers)       │
│                    ▼                                                 │
│          consistency_detector                                        │
│                    │                                                 │
│           final_synthesizer                                          │
│                    │                                                 │
│                   END                                                │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
{answer, citations, confidence_score, conflicts_detected,
 conflict_details, disclaimer, retrieved_sections, verification_passed}
```

---

## Code Files

### `backend/agents/state.py`

Defines `ComplianceState`, the shared TypedDict that flows through the graph. Uses `operator.add` as a reducer on `sub_answers` so that all parallel `process_sub_question` branches can write their results back without collisions.

**Key fields:**

| Field | Type | Set by |
|---|---|---|
| `query` | `str` | API caller |
| `analyzed_query` | `dict` | `query_analyzer` |
| `needs_clarification` | `bool` | `query_analyzer` |
| `clarification_question` | `str \| None` | `query_analyzer` |
| `sub_questions` | `list[dict]` | `decomposer` |
| `sub_question` | `dict` | `Send` payload (fan-out branch only) |
| `sub_answers` | `Annotated[list[dict], operator.add]` | `process_sub_question` (each branch appends) |
| `resolved_answers` | `list[dict]` | `consistency_detector` |
| `unresolved_conflicts` | `list[dict]` | `consistency_detector` |
| `final_answer` | `dict` | `final_synthesizer` |
| `final_response` | `dict` | `final_synthesizer` (API-compatible shape) |
| `error` | `str \| None` | any node on failure |

---

### `backend/agents/graph.py`

Wires all nodes into the compiled LangGraph `StateGraph`. Exports `query_graph` which is the compiled graph invoked by the API endpoints.

**Nodes registered:**
- `query_analyzer` → `query_analyzer_node`
- `clarification_response` → `clarification_response_node` (inline in graph.py)
- `decomposer` → `decomposer_node`
- `process_sub_question` → `process_sub_question_node`
- `consistency_detector` → `consistency_detector_node`
- `final_synthesizer` → `final_synthesizer_node`

**Edge logic:**
- `query_analyzer` → conditional: `check_clarification()` routes to `clarification_response` or `decomposer`
- `clarification_response` → `END`
- `decomposer` → conditional: `dispatch_sub_questions()` returns `[Send("process_sub_question", {...}), ...]` — one per sub-question
- `process_sub_question` → `consistency_detector` (LangGraph waits for all parallel branches)
- `consistency_detector` → `final_synthesizer`
- `final_synthesizer` → `END`

**Fan-out mechanism:** `dispatch_sub_questions()` creates one `Send` object per sub-question. Each `Send` carries `{query, analyzed_query, sub_question}` as a partial state for that branch. Results are collected by the `operator.add` reducer on `sub_answers`.

**Fallback:** If `decomposer` produces no sub-questions, `dispatch_sub_questions()` creates a single fallback `Send` using the original query unchanged.

---

### `backend/agents/query_analyzer.py`

**Stage 1 — Query Analysis**

One LLM call that parses the user's raw question into a structured object. Detects ambiguity before any expensive retrieval begins.

**Input:** `state["query"]`

**Output:** `analyzed_query`, `needs_clarification`, `clarification_question`

**`analyzed_query` shape:**
```json
{
  "intent_type": "compliance_check" | "definition" | "procedure" | "penalty",
  "entities": {
    "product_type": "string | null",
    "claim_type": "string | null",
    "fda_domain": "string | null",
    "explicit_refs": ["101.54"]
  },
  "is_multi_part": false,
  "needs_clarification": false,
  "clarification_question": null
}
```

**Intent types and downstream effect:**

| `intent_type` | Sub-answer synthesis format |
|---|---|
| `compliance_check` | Numbered requirements list with `[§X.XX]` citations |
| `definition` | Bold defined term + formal regulatory definition |
| `procedure` | Numbered step-by-step procedure with citations |
| `penalty` | Description of violations and penalties with citations |

**Clarification gate:** If `needs_clarification=true`, the graph short-circuits to `clarification_response` without calling any retrieval or synthesis nodes. The LLM is instructed to be conservative — only flag truly unanswerable questions (missing product type, claim type, AND regulatory domain simultaneously).

**Error handling:** Parse failure defaults to `intent_type="compliance_check"` and `needs_clarification=false` so the pipeline always proceeds.

---

### `backend/agents/decomposer.py`

**Stage 2 — Sub-Question Decomposition + Query Variant Generation**

Two-phase node. First decomposes the question into atomic sub-questions (1–4 max). Then generates three retrieval variants for each sub-question in parallel using separate LLM calls.

**Input:** `state["query"]`, `state["analyzed_query"]`

**Output:** `sub_questions` — list of `SubQuestion` dicts

**Phase 1 — Decomposition:**

LLM produces the minimum number of atomic, non-overlapping sub-questions. Each must be independently answerable. If the original question is already atomic, a single sub-question is produced.

**Phase 2 — Variant generation (parallel per sub-question):**

For each sub-question, three variants are generated concurrently via `ThreadPoolExecutor(max_workers=3)`:

| Variant | Temp | Purpose |
|---|---|---|
| `primary` | 0.1 | Keyword-rich rephrasing for dense/sparse retrieval |
| `hyde_passage` | 0.7 | Hypothetical CFR-style regulatory passage (HyDE, query-time) |
| `stepback` | 0.3 | Abstract principle-level query for broad recall |

Multiple sub-questions also decompose in parallel via a second `ThreadPoolExecutor(max_workers=min(4, N))`.

**`SubQuestion` shape:**
```json
{
  "id": "uuid-string",
  "text": "original atomic sub-question text",
  "variants": {
    "primary": "keyword-rich rephrasing",
    "hyde_passage": "hypothetical CFR passage...",
    "stepback": "broader abstract principle question"
  },
  "source_query": "original user question"
}
```

**Error handling:** If any single variant LLM call fails, that variant falls back to the sub-question text. If the decomposition LLM call fails entirely, a single sub-question is created from the original query.

---

### `backend/agents/retrieval_pipeline.py`

**Per-Sub-Question Pipeline — the core retrieval and synthesis engine**

This is the largest and most complex file. `process_sub_question_node` is the LangGraph node function that receives one `SubQuestion` via `Send`. It calls `run_sub_question_pipeline()` which runs all 10 stages below, then returns `{"sub_answers": [sub_answer]}`.

#### System Constants

```python
RRF_K = 60                        # Reciprocal Rank Fusion constant
FLOW_A_TOP_N = 40                 # Results per variant from Flow A (part-filtered)
FLOW_B_TOP_N = 40                 # Results per variant from Flow B (global)
TARGET_CANDIDATE_POOL = 80        # Pool size after RRF, before reranking
RERANKER_TOP_K = 12               # Final chunks passed downstream after reranking
CRAG_MAX_RETRIES = 3              # Max reformulation attempts on INCORRECT verdict
CRAG_THRESHOLD_HIGH = 0.70        # Reranker score for CORRECT verdict (top chunk)
CRAG_THRESHOLD_LOW = 0.45         # Reranker score for AMBIGUOUS verdict threshold
PART_CLASSIFIER_CONFIDENCE_MIN = 0.60  # Below this, skip Flow A entirely
CLAIM_SIMILARITY_THRESHOLD = 0.65     # BGE-M3 cosine similarity for claim support
CLAIM_KEYWORD_OVERLAP_THRESHOLD = 0.50 # Unigram overlap ratio for claim support
```

---

#### Stage 1 — Encode Query Variants

All three variants (`primary`, `hyde_passage`, `stepback`) are batch-encoded in a single BGE-M3 forward pass via `_encode_texts()`. Each encoding produces:
- A 1024-dimensional dense vector
- A sparse lexical-weight dict (indices + values, like a learned BM25)

The model instance is shared as a module-level singleton (`_retriever`) lazy-loaded on first use.

---

#### Stage 2 — Flow A: CFR Part-Routed Retrieval

**Purpose:** High-precision retrieval restricted to predicted CFR Part(s). Reduces noise by narrowing the search space before hybrid search.

**CFR Part Classifier (`_classify_cfr_parts`):**
- If the user typed an explicit CFR reference (e.g., "21 CFR 101.54"), the part number is extracted directly with regex → confidence = 1.0
- Otherwise, one LLM call predicts the most likely Part number(s) and a confidence score
- If confidence < `PART_CLASSIFIER_CONFIDENCE_MIN` (0.60), Flow A is skipped entirely and only Flow B runs

**When Flow A runs:**
- Builds a Qdrant `MatchAny` filter on the `part_number` payload index
- Runs dense + sparse search for each of the 3 variants in parallel (`ThreadPoolExecutor(max_workers=3)`)
- Each search returns up to `FLOW_A_TOP_N` (40) results per variant
- Each variant's dense and sparse results are merged by best score, keeping top 40

---

#### Stage 3 — Flow B: Global Retrieval

**Purpose:** Full-collection retrieval with no Part filter. Guarantees recall when Flow A's classifier predicted incorrectly or the answer spans multiple Parts.

- Runs dense + sparse search for each of the 3 variants in parallel
- No Qdrant filter — searches all 59,105 chunks
- Each search returns up to `FLOW_B_TOP_N` (40) results per variant
- Always runs regardless of whether Flow A ran

---

#### Stage 4 — RRF Merge

`_rrf_merge()` applies Reciprocal Rank Fusion across all variant/flow ranked lists:

```
RRF_score(chunk) = Σ_list  1 / (RRF_K + rank_in_list + 1)
```

- Maximum 6 ranked lists (2 flows × 3 variants); 3 if Flow A was skipped
- Chunks appearing in multiple lists are rewarded
- All unique chunks are collected, scored, and sorted
- Top `TARGET_CANDIDATE_POOL` (80) candidates proceed to reranking

---

#### Stage 5 — Cross-Encoder Reranking

`_rerank_candidates()` uses `bge-reranker-v2-m3` as a cross-encoder:
- Takes `(sub_question_text, chunk_text)` pairs for all 80 candidates
- Scores each pair with sigmoid normalisation → range [0, 1]
- Sorts by `reranker_score` descending
- Returns top `RERANKER_TOP_K` (12) chunks with both `rrf_score` and `reranker_score` attached

**Fallback:** If the reranker fails, falls back to RRF score order with a warning log.

---

#### Stage 6 — CRAG Evaluator

`_crag_verdict()` implements a three-way quality verdict:

| Condition | Verdict | Action |
|---|---|---|
| Top chunk score ≥ 0.70 **AND** ≥ 2 chunks score ≥ 0.45 | `CORRECT` | Proceed — `caveat_flag=false` |
| Top chunk score ≥ 0.45 but doesn't meet CORRECT | `AMBIGUOUS` | Supplemental search + re-rank; `caveat_flag=true` |
| All chunks score < 0.45 | `INCORRECT` | Reformulate and retry |
| Max retries reached and still INCORRECT | `LOW_CONFIDENCE` | Continue with available chunks; `caveat_flag=true` |

**AMBIGUOUS path:**
- Runs one supplemental global search using the primary variant
- RRF-merges supplemental results with existing fused pool
- Re-runs cross-encoder reranking on the enlarged pool
- Does NOT loop — continues after this single augmentation

**INCORRECT path — reformulation retry loop (max `CRAG_MAX_RETRIES=3` attempts):**

Each retry uses a different escalating strategy based on `retry_count`:

| Retry | Strategy |
|---|---|
| 0 | Synonym expansion: LLM rewrites query with expanded regulatory synonyms |
| 1 | Full rephrase: LLM rewrites query from a completely different angle |
| 2 | Variant switch: uses `hyde_passage` if available, else `stepback` as the query |

After each reformulation, the full pipeline (encode → Flow A → Flow B → RRF → rerank → CRAG) runs again with the new primary query. The reformulation is logged in `reformulation_log`.

If `CRAG_MAX_RETRIES` is exhausted, `crag_verdict` is set to `LOW_CONFIDENCE` and the pipeline continues with whatever chunks are available.

---

#### Stage 7 — Cross-Reference Resolution

`_resolve_cross_references()` discovers CFR sections referenced within retrieved chunks and fetches them from Qdrant to augment the chunk pool. Runs to a maximum depth of 2.

**Layer 1 — Pre-extracted references (regex-free):**
- Reads `cross_references_internal` from each chunk's Qdrant payload — a pre-extracted list of section number strings produced at ingestion time
- No regex needed; the list is already clean

**Layer 2 — LLM extraction of implicit references:**
- For each retrieved chunk, an LLM call scans the text for implicit reference phrases: "as defined in", "see also", "pursuant to", "in accordance with", "determined under", "as specified in"
- Returns a JSON list of referenced CFR section numbers

**Fetch step:**
- Qdrant `scroll()` by `section_number` payload index (direct metadata lookup, not vector search)
- Up to 10 chunks per section
- New chunks are marked `is_cross_ref=True` and `cross_ref_depth=1`

**Depth 2:**
- Depth-1 fetched chunks are themselves scanned for `cross_references_internal`
- Those sections are fetched and marked `cross_ref_depth=2`
- LLM extraction is not repeated at depth 2 (only Layer 1 regex at depth 2)

`already_fetched` set prevents re-fetching sections already in the primary pool.

---

#### Stage 8 — Draft Answer Generation

`_generate_draft_answer()` makes a preliminary LLM call to produce an unconstrained draft answer from the resolved chunk pool (primary + cross-ref chunks).

- Takes top 12 chunks from `resolved_chunks`
- Prepends `section_preamble` to chunk text when available
- The draft is intentionally loose — it exists only to feed the claim verifier
- `max_tokens=512`, `temperature=0.1`

---

#### Stage 9 — Claim-Level Grounding Verification

`_verify_claims()` checks every factual claim in the draft answer against the actual source chunk texts. Produces a `ClaimVerification` record for each claim.

**Step 1 — Claim extraction:**
`_extract_claims()` sends the draft answer to the LLM and asks it to extract every distinct atomic factual regulatory statement as a JSON list.

**Step 2 — Claim-to-chunk matching:**
For each claim, `_keyword_overlap()` computes unigram overlap with each chunk in `resolved_chunks`:
```
overlap = |claim_words ∩ chunk_words| / |claim_words|
```
Only words with ≥ 4 characters are counted.

If `overlap >= CLAIM_KEYWORD_OVERLAP_THRESHOLD (0.50)` → claim is `supported`.

**Cosine similarity fallback:** If keyword overlap fails for the best-scoring chunk, BGE-M3 is used to encode both the claim and the chunk text, and cosine similarity is computed:
```
supported = cosine(claim_embedding, chunk_embedding) >= CLAIM_SIMILARITY_THRESHOLD (0.65)
```

**Output per claim:**
```json
{
  "claim_text": "Products labeled 'excellent source' must contain...",
  "is_supported": true,
  "supporting_chunk_id": "21-I-SCA-1-B-101-54-para-a",
  "supporting_cfr_section": "101.54"
}
```

Unsupported claims are collected in `unverified_claims`. The draft answer text is kept as-is (pruning is tracked via the verification list rather than modifying the text).

---

#### Stage 10 — Sub-Answer Synthesis

`_synthesize_sub_answer()` generates the final structured sub-answer using only evidence-backed chunks.

- Selects chunks whose `chunk_id` appears in at least one `is_supported=True` ClaimVerification record
- Falls back to `resolved_chunks[:8]` if no supported claims exist
- Passes `intent_type` to instruct the LLM on formatting
- Returns JSON with `answer` (inline `[§X.XX]` citations) and `citations` list
- `max_tokens=1024`, `temperature=0.1`

---

#### Confidence Scoring

`_compute_confidence()` maps CRAG verdict + presence of unverified claims to a 0.0–1.0 float:

| CRAG Verdict | Unverified Claims | Confidence |
|---|---|---|
| `CORRECT` | None | 1.0 |
| `CORRECT` | Any | 0.8 |
| `AMBIGUOUS` | None | 0.6 |
| `AMBIGUOUS` | Any | 0.4 |
| `LOW_CONFIDENCE` | Any | 0.2 |

---

#### SubAnswer Shape

The final output of `run_sub_question_pipeline()`:

```json
{
  "sub_question_id": "uuid",
  "sub_question_text": "...",
  "answer": "formatted answer with inline [§X.XX] citations",
  "confidence": 0.8,
  "crag_verdict": "CORRECT",
  "caveat_flag": false,
  "chunks_used": [
    {"chunk_id": "...", "cfr_citation": "21 CFR § 101.54(b)", "text": "..."}
  ],
  "cross_refs_resolved": ["101.9", "101.13"],
  "claim_verification": [
    {"claim_text": "...", "is_supported": true, "supporting_chunk_id": "...", "supporting_cfr_section": "101.54"}
  ],
  "unverified_claims": [],
  "citations": [
    {"cfr_section": "21 CFR 101.54", "text_snippet": "..."}
  ],
  "caveats": [],
  "flags": [],
  "reformulation_log": []
}
```

**Caveats and flags attached by the pipeline:**

| Value | Type | Meaning |
|---|---|---|
| `AMBIGUOUS_EVIDENCE` | caveat | CRAG verdict was AMBIGUOUS |
| `LOW_CONFIDENCE_RETRIEVAL` | caveat | CRAG verdict was LOW_CONFIDENCE |
| `UNVERIFIED_CLAIMS_REMOVED` | caveat | At least one claim could not be grounded |
| `MAX_RETRIES_EXCEEDED` | flag | All 3 CRAG reformulation retries exhausted |
| `ALL_CLAIMS_UNVERIFIED` | flag | Every claim in the draft was unverified |
| `PIPELINE_ERROR` | both | Unhandled exception in the pipeline (graceful degradation) |

---

### `backend/agents/consistency.py`

**Stage 6 — Consistency & Conflict Detection**

Receives all collected `sub_answers` after the fan-in. Cross-examines claims across sub-answer pairs to find contradictions, then attempts automatic resolution.

**Input:** `state["sub_answers"]` (full list from all parallel branches)

**Output:** `resolved_answers`, `unresolved_conflicts`

**Step 1+2 — Cross-examination:**
- For each pair of sub-answers (i, j), samples up to 3 supported claims from each
- Each pair of claims (one from each sub-answer) is sent to the LLM for contradiction check: "Do these two regulatory claims contradict each other? YES/NO + explanation"
- Contradiction pairs are collected as potential `ConflictReport` entries

**Step 3 — Automatic resolution (by section specificity):**
Two resolution rules are applied in order:

**Rule 1 — Section specificity:**
- Compares the specificity of conflicting CFR sections using a heuristic: a more specific section has a longer section number string (e.g., `§101.54` is more specific than `§101`)
- Specificity score = `len(section_parts) × 10 + len(section_string)`
- If equal specificity: LLM is asked which section is more specific
- Winner overrides the loser; loser sub-answer gets `CONFLICT_RESOLVED` flag added

**Rule 2 — Recency:** Not implemented. The `effective_date` field is not present in the Qdrant payload schema.

If no rule resolves the conflict, it remains in `unresolved_conflicts` and is surfaced in the final response.

**`ConflictReport` shape:**
```json
{
  "sub_question_ids": ["uuid-a", "uuid-b"],
  "conflicting_sections": ["21 CFR 101.13", "21 CFR 101.54"],
  "description": "LLM explanation of the contradiction",
  "resolved": true,
  "resolution": "101.54 is more specific than 101.13",
  "winning_section": "21 CFR 101.54"
}
```

Only truly unresolved conflicts (`resolved=false`) are passed to `final_synthesizer`.

---

### `backend/agents/final_synthesizer.py`

**Stage 7 — Final Synthesis**

Merges all resolved sub-answers into a single `FinalAnswer` and maps it to the API-compatible `QueryResponse` shape.

**Input:** `state["resolved_answers"]`, `state["unresolved_conflicts"]`, `state["query"]`

**Output:** `final_answer` (v2 shape), `final_response` (API-compatible shape)

**Step 1 — Ruling determination:**
One LLM call receives a summary of all sub-answers and produces:
- `ruling`: `"YES"` | `"NO"` | `"CONDITIONAL"` — a direct answer to the original question
- `ruling_summary`: 1–2 sentence plain-language explanation
- `requirements`: list of `{item, citation}` — specific regulatory requirements with `[§X.XX]` citations
- `compliance_checklist`: ordered list of steps the user must take to comply

**Step 2 — Confidence level computation:**

| Average sub-answer confidence | `confidence_level` |
|---|---|
| ≥ 0.80 | `HIGH` |
| ≥ 0.50 | `MEDIUM` |
| < 0.50 | `LOW` |

Sub-questions with `confidence < 0.5` are listed in `low_confidence_sub_questions`.

**Step 3 — Regulation excerpts:**
Deduplicates all `chunks_used` across sub-answers by `cfr_citation`. Includes verbatim chunk text (truncated to 400 chars) for each unique section.

**Step 4 — Full FinalAnswer assembly:**
```json
{
  "ruling": "YES",
  "ruling_summary": "...",
  "requirements": [{"item": "...", "citation": "[§101.54]"}],
  "confidence_level": "HIGH",
  "confidence_explanation": "Average confidence 88% across 2 sub-questions.",
  "regulation_excerpts": [{"cfr_section": "21 CFR § 101.54(b)", "text": "..."}],
  "unresolved_conflicts": [],
  "low_confidence_sub_questions": [],
  "caveats": [],
  "compliance_checklist": ["Step 1: ...", "Step 2: ..."],
  "all_citations": ["21 CFR 101.54", "21 CFR 101.13"]
}
```

**API response mapping (`final_response`):**

`final_response` maps `FinalAnswer` to the existing `QueryResponse` shape consumed by the frontend. This preserves backward compatibility:

| QueryResponse field | Source |
|---|---|
| `answer` | `ruling` bold heading + `ruling_summary` + formatted `requirements` + `compliance_checklist` |
| `citations` | One entry per item in `all_citations` |
| `confidence_score` | HIGH→0.9 / MEDIUM→0.7 / LOW→0.4 |
| `conflicts_detected` | `len(unresolved_conflicts) > 0` |
| `conflict_details` | `[{sections, description}]` from unresolved conflicts |
| `disclaimer` | Static disclaimer string |
| `retrieved_sections` | `all_citations` list |
| `verification_passed` | `confidence_level in ("HIGH", "MEDIUM")` |

---

### `backend/agents/llm.py`

Unchanged from the previous version. Provides the LiteLLM wrapper with automatic model fallback chain.

**Functions:**
- `llm_completion(messages, max_tokens, temperature) → str` — plain text output
- `llm_completion_json(messages, ...) → str` — requests JSON mode; falls back if unsupported
- `llm_with_tools(messages, tools, ...) → (text, tool_calls[])` — tool calling with graceful fallback
- `parse_llm_json(raw, messages) → dict` — JSON parser; retries once on malformed output

**Model chain (tried left-to-right):**
1. `groq/<GROQ_MODEL>` if `GROQ_API_KEY` is set
2. `gemini/<GEMINI_MODEL>` if `GEMINI_API_KEY` is set
3. `nvidia_nim/<NVIDIA_MODEL>` if `NVIDIA_API_KEY` is set
4. `ollama/<OLLAMA_MODEL>` — always present as final local fallback

---

## LLM Call Budget Per Query

| Stage | Node | LLM calls | Notes |
|---|---|---|---|
| Query analysis | `query_analyzer` | 1 | Always |
| Decomposition | `decomposer` | 1 + (3 × N) | 1 decompose + 3 variants per sub-question |
| Part classification | `retrieval_pipeline` | 1 per sub-Q | Skipped if explicit refs present |
| CRAG reformulation | `retrieval_pipeline` | 0–3 per sub-Q | Only on INCORRECT verdict |
| Implicit cross-ref extraction | `retrieval_pipeline` | 1 per primary chunk | Layer 2 cross-ref |
| Draft answer | `retrieval_pipeline` | 1 per sub-Q | Always |
| Claim extraction | `retrieval_pipeline` | 1 per sub-Q | Always |
| Sub-answer synthesis | `retrieval_pipeline` | 1 per sub-Q | Always |
| Contradiction check | `consistency` | up to 9 per pair | 3 claims × 3 claims per sub-Q pair |
| Conflict resolution | `consistency` | 0–1 per conflict | Only for equal-specificity conflicts |
| Ruling + requirements | `final_synthesizer` | 1 | Always |

**Approximate total for a 2 sub-question query with no retries:** ~15–20 LLM calls.

---

## Qdrant Collection Schema (actual)

The pipeline reads from the `FDAComplianceAI` Qdrant collection. Field names used:

| Payload field | Type | Used by |
|---|---|---|
| `chunk_id` | string | Deduplication, claim verification |
| `chunk_type` | string | Definition chunk filtering (payload index) |
| `cfr_citation` | string | Display and citations (e.g., "21 CFR § 101.54(b)") |
| `text` | string | Retrieval, reranking, claim verification, synthesis |
| `section_preamble` | string | Prepended to text in draft answer generation |
| `cross_references_internal` | list[string] | Layer 1 cross-ref expansion (section numbers) |
| `hierarchy.section.number` | string | Cross-ref fetch, specificity resolution |
| `hierarchy.section.name` | string | Display |
| `hierarchy.part.number` | string | Part classifier cross-check |
| `part_number` | string | Flow A filter (payload index) |
| `section_number` | string | Cross-ref fetch filter (payload index) |
| `defines` | string | Definition chunk lookup (payload index) |
| `is_overflow_chunk` | bool | Overflow tracking |
| `overflow_sequence.next_chunk_id` | string | Overflow chain expansion |

**Named vectors in the collection:**
- `cfr-dense` — 1024-dimensional BGE-M3 dense vector (cosine similarity)
- `cfr-sparse` — BGE-M3 sparse lexical weights (dot product)

Note: HyPE question vectors (`hyde_q_0`…`hyde_q_4`) are NOT present in the collection and are not used.

---

## Retriever Changes (`backend/retrieval/retriever.py`)

The following methods were exposed as public to support the v2 retrieval pipeline's need for direct access to search primitives:

| New public method | Previous name | Purpose |
|---|---|---|
| `encode_query(query)` | `_encode_query` | BGE-M3 encode → (dense_vec, sparse_dict) |
| `search_dense(dense_vec, filter, top_k)` | `_search_dense` | Dense vector search in Qdrant |
| `search_sparse(sparse_dict, filter, top_k)` | `_search_sparse` | Sparse vector search in Qdrant |

Private `_search_dense` and `_search_sparse` are kept as aliases for backward compatibility with any code that calls them directly.

---

## SSE Event Schema (`POST /api/query/stream`)

Each SSE event has the form `data: {JSON}\n\n`. Terminal event: `data: [DONE]\n\n`.

| `event` field | Additional fields | Emitted when |
|---|---|---|
| `query_analyzer` | `intent_type`, `needs_clarification`, `entities` | Query analysis complete |
| `clarification_response` | `answer` (full QueryResponse) | Clarification needed (pipeline stops) |
| `decomposer` | `sub_question_count`, `sub_questions[]` | Decomposition complete |
| `process_sub_question` | `sub_question_text`, `crag_verdict`, `confidence`, `citation_count` | Each parallel branch completes |
| `consistency_detector` | `conflict_count` | Consistency check complete |
| `final_synthesizer` | `answer` (full QueryResponse), `ruling`, `confidence_level`, `conflicts_detected` | Final synthesis complete |
| `error` | `detail` | Unhandled exception in graph |

---

## What Was Not Implemented

| Feature | Reason |
|---|---|
| **HyPE (index-time question vectors)** | `hyde_q_0`…`hyde_q_4` named vectors don't exist in the Qdrant collection. Requires re-embedding all 59,105 chunks. HyDE (query-time) is fully implemented instead. |
| **Conflict resolution by recency (Rule 2)** | `effective_date` field is not present in the chunk payload schema. Rule 1 (section specificity) is implemented. |

---

## File Inventory

```
backend/agents/
  __init__.py              — Package init (unchanged)
  state.py                 — ComplianceState TypedDict with fan-in reducer
  query_analyzer.py        — Stage 1: intent, entities, clarification detection
  decomposer.py            — Stage 2: sub-question decomposition + HyDE/stepback variants
  retrieval_pipeline.py    — Per-sub-Q: Flow A+B, RRF, CRAG, cross-ref, claim verify, synthesis
  consistency.py           — Stage 6: cross-claim contradiction detection + resolution
  final_synthesizer.py     — Stage 7: ruling + requirements + API response mapping
  graph.py                 — LangGraph StateGraph wiring + compiled query_graph
  llm.py                   — LiteLLM wrapper with model fallback chain (unchanged)

backend/retrieval/
  retriever.py             — CFRRetriever (search_dense, search_sparse, encode_query now public)
```

**Deleted files (replaced by v2 architecture):**
- `planner.py` — replaced by `query_analyzer.py` + `decomposer.py`
- `retriever_node.py` — replaced by `retrieval_pipeline.py`
- `definition_resolver.py` — integrated into `retrieval_pipeline.py` (cross-ref resolution + LLM implicit extraction)
- `synthesizer.py` — replaced by sub-answer synthesis in `retrieval_pipeline.py` + `final_synthesizer.py`
- `verifier.py` — replaced by claim-level grounding verification in `retrieval_pipeline.py`
- `conflict_detector.py` — replaced by `consistency.py` + conflict handling in `final_synthesizer.py`
