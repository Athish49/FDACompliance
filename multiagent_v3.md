# Multi-Agent FDA 21 CFR Compliance RAG Pipeline — v2 Technical Specification

## System Scope

- Corpus: 21 CFR regulations only, indexed in Qdrant
- Optimise for accuracy over latency
- All sub-question processing runs in parallel

---

## Qdrant Collection Schema

Every chunk stored in Qdrant must carry the following payload fields. Fields marked `[filterable]` must be indexed as Qdrant payload indexes.

```json
{
  "chunk_id": "uuid-string",
  "cfr_part": 101,
  "cfr_subpart": "B",
  "cfr_section": "101.54",
  "section_title": "Nutrient content claims for protein",
  "effective_date": "2024-01-01",
  "text": "full chunk text content",
  "token_count": 312
}
```

| Field | Type | Filterable | Purpose |
|---|---|---|---|
| `chunk_id` | string | yes | Deduplication key across retrieval runs |
| `cfr_part` | integer | yes | Flow A metadata filter — Part-level routing |
| `cfr_subpart` | string | yes | Optional narrower filter |
| `cfr_section` | string | yes | Cross-reference direct lookup key |
| `section_title` | string | no | Display and synthesis context |
| `effective_date` | string | yes | Conflict resolution by recency |
| `text` | string | no | Chunk content passed to LLM |
| `token_count` | integer | no | Context window budgeting |

**HyPE vectors (index-time enhancement):**
For each chunk, generate 3–5 hypothetical questions that the chunk answers. Store their embeddings as additional named vectors in the same Qdrant point alongside the primary chunk embedding. Label them `hyde_q_0` through `hyde_q_4`. At query time, search against both the primary vector and these question vectors.

---

## System Constants

```python
RRF_K = 60                      # RRF ranking constant (standard value)
FLOW_A_TOP_N = 40               # Chunks returned per query variant from Flow A
FLOW_B_TOP_N = 40               # Chunks returned per query variant from Flow B
TARGET_CANDIDATE_POOL = 80      # Pool size after RRF merge, before reranking
RERANKER_TOP_K = 12             # Final chunks passed downstream after reranking
CRAG_MAX_RETRIES = 3            # Max reformulation attempts on INCORRECT verdict
CROSS_REF_MAX_DEPTH = 2         # Max recursive hops in cross-reference resolution
CRAG_THRESHOLD_HIGH = 0.70      # Min reranker score for standard CORRECT verdict
CRAG_THRESHOLD_VERY_HIGH = 0.85 # Min top score for single-chunk CORRECT verdict
CRAG_AMBIGUOUS_SOFT_CORRECT = 0.70  # If AMBIGUOUS and top >= this, skip supplemental search
CRAG_BEST_CHUNK_MIN_SCORE = 0.30    # Min top score to update best_chunks_seen carry-forward
CRAG_THRESHOLD_LOW = 0.45       # Min reranker score for AMBIGUOUS verdict
REFORMULATE_MIN_LEN = 15        # Min char length for a valid reformulation string
VARIANT_MIN_LEN = 15            # Min char length for a valid query variant
HYDE_GENERATION_TEMP = 0.7      # LLM temperature for HyDE passage generation
STEPBACK_GENERATION_TEMP = 0.3  # LLM temperature for step-back query generation
PART_CLASSIFIER_CONFIDENCE_MIN = 0.60  # Below this, skip Flow A and use Flow B only
CONSISTENCY_CONF_THRESHOLD = 0.3  # Sub-answers below this skip contradiction detection
SYNTHESIZER_WEAK_THRESHOLD = 0.3  # Below this, sub-answer excluded from confidence avg
```

---

## Data Models

```typescript
interface UserQuery {
  raw_text: string
  timestamp: string
}

interface AnalyzedQuery {
  raw_text: string
  intent_type: "compliance_check" | "definition" | "procedure" | "penalty"
  entities: {
    product_type: string | null
    claim_type: string | null
    fda_domain: string | null
    explicit_refs: string[]          // e.g. ["101.54", "21 CFR 101.9"]
  }
  is_multi_part: boolean
  needs_clarification: boolean
  clarification_question: string | null
}

interface QueryVariants {
  primary: string                    // Direct sub-question
  hyde_passage: string               // Hypothetical CFR answer passage
  stepback: string                   // Abstract principle query
}

interface SubQuestion {
  id: string                         // uuid
  text: string
  variants: QueryVariants
  source_query: string               // Original user query
}

interface RetrievedChunk {
  chunk_id: string
  cfr_part: number
  cfr_subpart: string
  cfr_section: string
  section_title: string
  effective_date: string
  text: string
  score: number                      // Similarity score from retrieval
  source_flow: "A" | "B"
  source_variant: "primary" | "hyde" | "stepback"
}

interface RankedChunk extends RetrievedChunk {
  rrf_score: number                  // Score after RRF merge
  reranker_score: number             // Score after cross-encoder reranking
}

interface CRAGResult {
  verdict: "CORRECT" | "AMBIGUOUS" | "INCORRECT" | "LOW_CONFIDENCE"
  chunks: RankedChunk[]              // Final chunk pool (may be augmented)
  caveat_flag: boolean               // True if verdict was AMBIGUOUS
  retry_count: number
  reformulation_log: string[]        // Record of each reformulation attempt
}

interface CrossRef {
  section: string                    // e.g. "101.9"
  source_chunk_id: string            // Which chunk referenced this
  extraction_method: "regex" | "llm"
  depth: number                      // 1 or 2
}

interface ResolvedChunk extends RankedChunk {
  is_cross_ref: boolean
  cross_ref_source: string | null    // section that referenced this chunk
}

interface ClaimVerification {
  claim_text: string
  is_supported: boolean
  supporting_chunk_id: string | null
  supporting_cfr_section: string | null
}

interface SubAnswer {
  sub_question_id: string
  sub_question_text: string
  answer: string
  confidence: number                 // 0.0–1.0, see Confidence Scoring section
  crag_verdict: string
  caveat_flag: boolean
  chunks_used: ResolvedChunk[]
  cross_refs_resolved: string[]
  claim_verification: ClaimVerification[]
  unverified_claims: string[]
  citations: string[]
  caveats: string[]
  flags: string[]                    // e.g. ["LOW_CONFIDENCE", "AMBIGUOUS_EVIDENCE",
                                     //       "SKIPPED_NO_EVIDENCE",
                                     //       "LOW_CONF_EXCLUDED_FROM_CONFLICT_CHECK",
                                     //       "RETRIEVAL_DOMAIN_MISMATCH"]
  reformulation_log: string[]
  skipped: boolean                   // True when LOW_CONFIDENCE + best_chunks_seen empty;
                                     // excluded from synthesis and confidence averaging
}

interface ConflictReport {
  sub_question_ids: string[]
  conflicting_sections: string[]
  description: string
  resolved: boolean
  resolution: string | null          // How it was resolved, if resolved
  winning_section: string | null
}

interface DomainMismatchReport {
  sub_question_ids: string[]         // Two sub-answers with incompatible retrieved domains
  domain_a: string                   // e.g. "food"
  domain_b: string                   // e.g. "drug"
  description: string
}

interface FinalAnswer {
  ruling: "YES" | "NO" | "CONDITIONAL"
  ruling_summary: string
  reasoning_steps: string[]          // Numeric derivation steps; [] for qualitative rulings
  requirements: {
    item: string
    citation: string                 // e.g. "[§101.54]"
  }[]
  confidence_level: "HIGH" | "MEDIUM" | "LOW"
  confidence_explanation: string
  regulation_excerpts: {
    cfr_section: string
    text: string
  }[]
  unresolved_conflicts: ConflictReport[]
  domain_mismatches: DomainMismatchReport[]  // Retrieval-domain failures; separate from
                                             // regulatory conflicts
  low_confidence_sub_questions: string[]
  caveats: string[]
  compliance_checklist: string[]
  all_citations: string[]            // Deduplicated list of all §refs used
  partial_information_note: string | null  // Informational flag when sub-questions were
                                           // skipped or had weak retrieval; not a penalty
}
```

---

## Confidence Scoring Rules

### Per-Sub-Answer Confidence

Applied in Sub-Answer Synthesis Agent. Used to populate `SubAnswer.confidence`.

| CRAG Verdict | Unverified Claims | Confidence Score |
|---|---|---|
| CORRECT | 0 | 1.0 |
| CORRECT | > 0 | 0.8 |
| AMBIGUOUS | 0 | 0.6 |
| AMBIGUOUS | > 0 | 0.4 |
| LOW_CONFIDENCE | any | 0.2 |

Sub-questions where `verdict = LOW_CONFIDENCE` AND `best_chunks_seen` is empty are
**not synthesized** — they return `SubAnswer(skipped=true, confidence=0.0)` and are
excluded from the final synthesizer's confidence average entirely.

### Final `confidence_level` (Weighted Scoring)

Computed by the Final Synthesis Agent from non-skipped sub-answers only:

```python
strong = [sa for sa in resolved_answers if sa.confidence >= SYNTHESIZER_WEAK_THRESHOLD]  # 0.3
weak   = [sa for sa in resolved_answers if sa.confidence <  SYNTHESIZER_WEAK_THRESHOLD]
pool   = strong if strong else resolved_answers   # fallback if all are weak
avg    = mean(sa.confidence for sa in pool)
```

Thresholds applied to `avg`:
- `>= 0.8` → HIGH
- `>= 0.5` → MEDIUM
- `< 0.5` → LOW

Rationale: a single strong sub-answer (e.g. `confidence=0.6`) is not dragged down to
LOW by weak sub-answers (`confidence=0.2`) that had poor retrieval. The
`confidence_explanation` reports the count of strong sub-answers used and notes when
weak ones were excluded from scoring. Skipped and weak sub-answers are surfaced via
`partial_information_note` without penalising the confidence level.

---

## Pipeline Flow

```
UserQuery
  │
  ▼
[Stage 1] Query Analysis Agent
  │  Output: AnalyzedQuery
  │  If needs_clarification=true → return clarification_question to user, STOP
  │
  ▼
[Stage 2] Sub-Question Decomposition Agent
  │  Output: List[SubQuestion]  (each with QueryVariants)
  │
  ▼ (parallel for each SubQuestion)
  ├─────────────────────────────────────────────────────────┐
  │                  Per-Sub-Question Loop                   │
  │                                                          │
  │  [Query Expansion]                                       │
  │       │  Confirms QueryVariants (primary, hyde, step)    │
  │       │                                                  │
  │       ├──────────────────────┐                           │
  │       ▼                      ▼                           │
  │  [Flow A: Routed]      [Flow B: Global]                  │
  │       │  3 variants          │  3 variants               │
  │       └──────────┬───────────┘                           │
  │                  ▼                                        │
  │         [RRF Merge & Dedup]                               │
  │                  │  ~80 candidate pool                    │
  │                  ▼                                        │
  │         [Cross-Encoder Reranker]                          │
  │                  │  Top-K chunks (12)                     │
  │                  ▼                                        │
  │         [CRAG Evaluator]                                  │
  │          ├─ CORRECT (2-path: very-high OR standard) ─────┤
  │          ├─ AMBIGUOUS → top>=0.70 break; else suppl. ────┤
  │          └─ INCORRECT → reformulate & retry              │
  │                          retry 2: section-anchor first    │
  │                          best_chunks_seen carry-forward   │
  │                  │  CRAGResult                            │
  │                  ▼                                        │
  │         [Skip-No-Evidence Gate]                           │
  │          ├─ LOW_CONFIDENCE + best_chunks_seen empty       │
  │          │   → SubAnswer(skipped=true), bypass synthesis  │
  │          └─ otherwise → continue                          │
  │                  ▼                                        │
  │         [Cross-Reference Resolution Agent]                │
  │                  │  ResolvedChunk[]                       │
  │                  ▼                                        │
  │         [Claim-Level Grounding Verification]              │
  │                  │  ClaimVerification[]                   │
  │                  ▼                                        │
  │         [Sub-Answer Synthesis Agent]                      │
  │                  │  SubAnswer                             │
  └─────────────────────────────────────────────────────────┘
  │
  ▼ (collect all SubAnswer objects, including skipped=true)
[Stage 6] Consistency & Conflict Detection Agent
  │  active/skipped split → conf threshold filter → domain pre-check
  │  Output: {resolved_answers, unresolved_conflicts, domain_mismatches}
  │
  ▼
[Stage 7] Final Synthesis Agent
  │  filter skipped → primary sub-answer → weighted conf
  │  → reasoning_steps → partial_information_note
  │  Output: FinalAnswer
  │
  ▼
Response to user
```

---

## Stage 1 — Query Analysis Agent

### Objective
Parse the raw user question into a structured object that downstream agents can operate on. Detect ambiguity before decomposition begins.

### Input
```typescript
input: UserQuery
```

### Process
1. Send `raw_text` to LLM with a structured extraction prompt
2. LLM classifies `intent_type` from: `compliance_check`, `definition`, `procedure`, `penalty`
3. LLM extracts entities: `product_type`, `claim_type`, `fda_domain`
4. LLM extracts any explicit §refs or "21 CFR X.XX" patterns from the raw text into `explicit_refs`
5. LLM flags `is_multi_part = true` if the question contains more than one distinct compliance question
6. LLM flags `needs_clarification = true` if the question is too vague to decompose (e.g., missing product type, claim type, or regulatory context)
7. If `needs_clarification = true`, LLM generates a specific `clarification_question` targeting the missing information

### LLM Prompt Structure
- System: FDA regulatory compliance expert that extracts structured query metadata
- User: raw query text
- Output format: JSON matching `AnalyzedQuery` interface, no additional text

### Output
```typescript
output: AnalyzedQuery
```

### Pipeline Gate
- If `needs_clarification = true`: return `clarification_question` to the user, halt pipeline. Resume when user responds with clarifying input.
- If `needs_clarification = false`: pass `AnalyzedQuery` to Stage 2.

`needs_clarification` and `clarification_question` are **also hoisted** to top-level
`ComplianceState` keys so the conditional edge in `graph.py` can route without
unpacking the `analyzed_query` dict. `dispatch_sub_questions` forwards the full
`AnalyzedQuery` to every parallel branch so each sub-question pipeline has access to
`intent_type`, `entities` (including `fda_domain` and `explicit_refs`), and
`is_multi_part`.

### Error Handling
- LLM parse failure → retry once with explicit JSON schema in prompt
- Second failure → return generic clarification request to user, halt pipeline

---

## Stage 2 — Sub-Question Decomposition Agent

### Objective
Decompose the analyzed query into atomic, single-intent sub-questions. Generate three retrieval query variants for each sub-question.

### Input
```typescript
input: AnalyzedQuery
```

### Process

**Step 1 — Decomposition (single LLM call)**

Sub-questions are produced by a **single** LLM call (not parallel). The decomposition
prompt enforces these hard rules passed as system instructions:

1. Sub-question text MUST be a **declarative retrieval topic**, not a yes/no question.
   - Bad: "Does 20g protein per serving qualify as an 'excellent source' claim?"
   - Good: "FDA criteria for 'excellent source' nutrient content claims for protein"
   Yes/no phrasings pull unrelated regulatory text because semantic search finds
   "criteria" and "claim" in drug/device docs instead of the correct food-labeling Part.
2. Two sub-questions that would retrieve the same CFR sections must be merged.
3. Each sub-question must have a single, non-overlapping intent.
4. Each sub-question must be self-contained (independently answerable without others).
5. `is_multi_part = false` → produce **exactly 1** sub-question (hard constraint).
   `is_multi_part = true`  → produce the minimum needed, maximum 4.
6. Assign a unique `id` (uuid) to each sub-question.

The LLM `user_content` for decomposition MUST include all of: `intent_type`,
`product_type`, `claim_type`, `fda_domain`, `explicit_refs`, and `is_multi_part` from
`AnalyzedQuery`. Omitting `fda_domain` or `is_multi_part` removes the signals the LLM
needs to pick domain-anchored topics and to gate single vs. multi sub-question output.

**Step 2 — Query Variant Generation (per sub-question, parallel)**

For each sub-question, run three separate LLM calls in a `ThreadPoolExecutor` (variant
generation is the **only** parallel step in this stage). Each sub-question's three
variants are independent of other sub-questions' variants.

**Variant 1 — Primary query**
- Input: sub-question text as-is
- Output: `variants.primary` — rephrased for optimal retrieval (concise, keyword-rich)

**Variant 2 — HyDE passage**
- Input: sub-question text
- System prompt: "You are an FDA regulatory expert. Write a short passage (100–150 words) in the style of the Code of Federal Regulations that would directly answer the following compliance question. Use regulatory language and structure. Do not add disclaimers."
- Temperature: `HYDE_GENERATION_TEMP` (0.7)
- Output: `variants.hyde_passage` — a hypothetical CFR-style passage

**Variant 3 — Step-back query**
- Input: sub-question text
- System prompt: "Rewrite the following specific compliance question as a broader, more abstract question about the underlying FDA regulatory principle it relates to. Output only the rewritten question."
- Temperature: `STEPBACK_GENERATION_TEMP` (0.3)
- Output: `variants.stepback` — abstract principle-level query

**Step 3 — Variant Validation (`_validate_variant`)**

After each `future.result()` call, validate before attaching to `SubQuestion.variants`:
- Empty or `None` → invalid
- `len(stripped) < VARIANT_MIN_LEN` (15) → invalid
- Else → accept

On invalid: log WARNING and fall back to the sub-question text as the safe default.
A successful API call returning `""` must never reach the retrieval pipeline — vacuous
queries pull wrong-domain chunks and waste CRAG retries on noise.

### Output
```typescript
output: SubQuestion[]
```
Each `SubQuestion` contains `id`, `text`, `variants` (primary, hyde_passage, stepback), `source_query`.

### Error Handling
- If decomposition LLM call fails → retry once
- If variant generation fails (exception or validator rejects) → fall back to
  sub-question text for that variant, log warning
- If the original query is already atomic (`is_multi_part = false`) → produce a single
  `SubQuestion` with `text = AnalyzedQuery.raw_text`

---

## Per-Sub-Question Loop

The following components execute for each `SubQuestion` independently and in parallel. All components share a single `SubQuestion` as their root context.

---

### Component: Query Expansion

### Objective
Confirm and expose the three query variants for use by both retrieval flows.

### Input
```typescript
input: SubQuestion
```

### Process
1. Extract `variants.primary`, `variants.hyde_passage`, `variants.stepback` from the `SubQuestion`
2. Generate embeddings for all three variants using the primary embedding model
3. Package as `QueryVariants` with pre-computed embedding vectors

### Output
```typescript
output: QueryVariants & {
  primary_embedding: float[]
  hyde_embedding: float[]
  stepback_embedding: float[]
}
```

---

### Component: Flow A — Routed Retrieval

### Objective
High-precision retrieval restricted to the predicted CFR Part(s). Reduces noise by narrowing the search space before hybrid search.

### Input
```typescript
input: {
  query_variants: QueryVariants,
  embeddings: { primary, hyde, stepback },
  sub_question_text: string
}
```

### Process

**Step 1 — CFR Part Classification (grounded LLM classifier)**

Unanchored LLM knowledge of "which 21 CFR Part covers which topic" is unreliable.
The classifier is grounded in a knowledge base built at ingestion time.

**Knowledge base (`backend/data/cfr_part_index.json`):**
- Generated by `backend/ingestion/part_index.py` from `cfr_chunks.json` — always in
  sync with the indexed corpus (no manual maintenance)
- ~256 active CFR Parts (~3.2 K tokens), each entry:
  `{ "title": "...", "subchapter": "<A-L>", "subchapter_name": "..." }`
- Loaded once at startup (lazy, cached, thread-safe)
- Regenerated automatically as Step 3 of `run_pipeline()` on every ingestion run

**Domain pre-filter (pure Python, no LLM call):**

Uses `AnalyzedQuery.entities.fda_domain` to narrow the part list sent to the LLM:

| fda_domain | Subchapter(s) |
|---|---|
| `food` | B |
| `drug` | C, D |
| `animal` | E |
| `biological` | F |
| `cosmetic` | G |
| `device` | H, I, J |
| `tobacco` | K |
| `administrative` | A, L |
| `null` / unknown | (no filter — all 256 parts) |

**Classifier call:**
1. Build LLM `user_content` from `sub_question_text` and the filtered part list in
   compact `"Part 101: Food Labeling\nPart 102: ..."` format
2. If `explicit_refs` are non-empty, append as **hint only** (not a hard override):
   `"Hint — original query mentioned these CFR sections: [...]"`
   The system prompt instructs the LLM to classify independently if the sub-question
   covers a different regulatory topic than the hinted sections. (Prior implementation
   hard-overrode the classifier to `confidence=1.0` on any non-empty `explicit_refs`,
   which forced every sub-question into the originally-cited Part even when the
   sub-question covered an unrelated regulatory area.)
3. Classifier returns predicted `cfr_part[]` and a confidence score in `[0, 1]`
4. If classifier confidence < `PART_CLASSIFIER_CONFIDENCE_MIN` (0.60):
   - Skip Flow A entirely
   - Log: `"Flow A skipped: classifier confidence below threshold"`
   - Return empty result set; Flow B covers full retrieval

**Fallback:** if `cfr_part_index.json` is absent (e.g. first run before ingestion),
the classifier silently omits part-list context and falls back to unconstrained LLM
classification.

**Step 2 — Qdrant Metadata Filter**
1. Build Qdrant filter: `cfr_part IN [predicted_parts]`
2. This restricts all searches to only chunks belonging to the predicted Part(s)

**Step 3 — Hybrid Search (per variant)**
Run the following for each of the 3 query variants independently:

Dense search:
```
qdrant.search(
  collection_name="cfr_21",
  query_vector=variant_embedding,
  query_filter=part_filter,
  limit=FLOW_A_TOP_N,
  with_payload=true
)
```

Sparse search (BM25):
```
qdrant.search(
  collection_name="cfr_21",
  query=variant_text,
  search_type="sparse",
  query_filter=part_filter,
  limit=FLOW_A_TOP_N,
  with_payload=true
)
```

Combine dense and sparse for each variant using score normalization (min-max) then average. Tag each result with `source_flow="A"` and `source_variant`.

### Output
```typescript
output: RetrievedChunk[]    // Up to FLOW_A_TOP_N × 3 variants, tagged
```

### Error Handling
- Qdrant search failure → retry once after 500ms
- Second failure → return empty set, log error, Flow B provides full coverage

---

### Component: Flow B — Global Retrieval

### Objective
Full-collection retrieval with no Part filter. Acts as recall safety net for cases where Flow A's classifier predicted incorrectly or the answer spans multiple Parts.

### Input
```typescript
input: {
  query_variants: QueryVariants,
  embeddings: { primary, hyde, stepback }
}
```

### Process
Run hybrid search (dense + sparse) for each of the 3 query variants against the full Qdrant collection with no metadata filter. Identical process to Flow A Step 3, without the `query_filter`. Tag each result with `source_flow="B"` and `source_variant`.

```
qdrant.search(
  collection_name="cfr_21",
  query_vector=variant_embedding,
  limit=FLOW_B_TOP_N,
  with_payload=true
)
```

### Output
```typescript
output: RetrievedChunk[]    // Up to FLOW_B_TOP_N × 3 variants, tagged
```

### Error Handling
- Qdrant search failure → retry once after 500ms
- Second failure → propagate error upstream; pipeline cannot continue without retrieval

---

### Component: RRF Merge & Deduplication

### Objective
Combine all retrieval results from Flow A and Flow B (across all 3 query variants) into a single deduplicated ranked list using Reciprocal Rank Fusion.

### Input
```typescript
input: {
  flow_a_results: RetrievedChunk[],
  flow_b_results: RetrievedChunk[]
}
```

### Process

**Step 1 — Collect all ranked lists**
Treat each (flow, variant) combination as a separate ranked list. Maximum 6 ranked lists (2 flows × 3 variants). If Flow A was skipped, there are 3 ranked lists from Flow B only.

**Step 2 — Deduplicate**
Identify all unique `chunk_id` values across all lists.

**Step 3 — Apply RRF**
For each unique chunk, compute its RRF score across all lists it appears in:

```
RRF_score(chunk) = Σ_list  1 / (RRF_K + rank_in_list(chunk))
```

Where:
- `RRF_K = 60`
- `rank_in_list(chunk)` is the 1-indexed position of the chunk in that ranked list
- If the chunk does not appear in a list, it contributes 0 to the sum

**Step 4 — Sort and trim**
Sort all unique chunks by `RRF_score` descending. Keep top `TARGET_CANDIDATE_POOL` (80) chunks. Attach `rrf_score` to each chunk.

### Output
```typescript
output: RankedChunk[]    // Top 80, sorted by rrf_score descending
```

---

### Component: Cross-Encoder Reranker

### Objective
Score each candidate chunk against the original sub-question using a cross-encoder model. Produce the final top-K chunk list for downstream quality evaluation.

### Input
```typescript
input: {
  candidate_pool: RankedChunk[],    // ~80 chunks
  sub_question_text: string          // original primary sub-question (not HyDE or stepback)
}
```

### Process
1. For each chunk in `candidate_pool`, create an input pair: `(sub_question_text, chunk.text)`
2. Pass all pairs through the cross-encoder model (e.g., BGE-Reranker-v2, Cohere Rerank)
3. Receive a relevance score `[0.0, 1.0]` per pair
4. Attach `reranker_score` to each `RankedChunk`
5. Sort by `reranker_score` descending
6. Return top `RERANKER_TOP_K` (12) chunks

### Output
```typescript
output: RankedChunk[]    // Top 12, sorted by reranker_score descending
```

### Error Handling
- Reranker API failure → retry once
- Second failure → fall back to RRF-ranked order, use `rrf_score` as proxy, attach flag `"RERANKER_UNAVAILABLE"` to `SubAnswer.flags`

---

### Component: CRAG Evaluator

### Objective
Assess whether the retrieved top-K chunks contain sufficient evidence to answer the sub-question. Trigger the appropriate action based on a three-way verdict. Manage the reformulation retry loop on failure.

### Input
```typescript
input: {
  top_k_chunks: RankedChunk[],
  sub_question: SubQuestion,
  retry_count: number,                   // starts at 0
  reformulation_log: string[],           // starts as []
  best_chunks_seen: RankedChunk[]        // carry-forward across retries; starts as []
}
```

### Process

**Step 1 — Verdict Determination**

Let `top_score = top_k_chunks[0].reranker_score` and
`above_low = count(c for c in top_k_chunks if c.reranker_score >= CRAG_THRESHOLD_LOW)`.

| Condition | Verdict |
|---|---|
| `top_score >= CRAG_THRESHOLD_VERY_HIGH` (0.85) AND `above_low >= 1` | CORRECT |
| `top_score >= CRAG_THRESHOLD_HIGH` (0.70) AND `above_low >= 2` | CORRECT |
| `top_score >= CRAG_THRESHOLD_LOW` but neither CORRECT condition met | AMBIGUOUS |
| All chunk scores `< CRAG_THRESHOLD_LOW` | INCORRECT |
| `retry_count >= CRAG_MAX_RETRIES` and still INCORRECT | LOW_CONFIDENCE |

The two CORRECT paths replace the single prior condition. A single very strongly
relevant chunk (`top_score >= 0.85`) is sufficient evidence on its own; requiring a
second chunk above 0.45 incorrectly downgraded those results to AMBIGUOUS.

**Step 1b — Best-Chunks Carry-Forward**

After every rerank pass (initial and each retry), before routing:

```python
if top_k_chunks and top_score > CRAG_BEST_CHUNK_MIN_SCORE:  # 0.30
    if not best_chunks_seen or top_score > best_chunks_seen[0].reranker_score:
        best_chunks_seen = top_k_chunks
```

This preserves the highest-scoring chunk set seen across all retries. Later retries on
a failing query tend to retrieve similar irrelevant content, so without this the
LOW_CONFIDENCE path would return the *worst* iteration.

**Step 2 — CORRECT path**
- Set `verdict = "CORRECT"`, `caveat_flag = false`
- Pass `top_k_chunks` to Cross-Reference Resolution Agent unchanged

**Step 3 — AMBIGUOUS path**

If `top_score >= CRAG_AMBIGUOUS_SOFT_CORRECT` (0.70):
- The top chunk already meets the HIGH threshold; adding supplemental chunks risks
  diluting a good result with noise
- Skip supplemental search, break the retry loop immediately
- Set `verdict = "AMBIGUOUS"`, `caveat_flag = true`
- Pass `top_k_chunks` unchanged to Cross-Reference Resolution Agent

Otherwise (genuinely ambiguous, `top_score < 0.70`):
1. Run supplemental broad search: Flow B with `variants.primary` only, no Part filter, `limit = FLOW_B_TOP_N`
2. Merge supplemental results with existing `top_k_chunks` using RRF
3. Re-run cross-encoder reranker on merged pool
4. Set `verdict = "AMBIGUOUS"`, `caveat_flag = true`
5. Pass augmented chunk list to Cross-Reference Resolution Agent

**Step 4 — INCORRECT path (retry loop)**
If `retry_count < CRAG_MAX_RETRIES`:

Select reformulation strategy based on `retry_count`:
| retry_count | Strategy |
|---|---|
| 0 | Synonym expansion: prompt LLM to rewrite `variants.primary` with expanded regulatory synonyms |
| 1 | Full rephrase: prompt LLM to rewrite the sub-question from a different angle |
| 2 | Section-anchor (see Step 4a below) |

**Step 4a — Section-Anchor Reformulation (retry_count == 2)**

Priority order for the reformulation string:

1. **Section anchor** (preferred when `best_chunks_seen` is non-empty):
   - Helper `_extract_best_section(best_chunks_seen)` checks top 3 chunks; tries
     `chunk.hierarchy.section.number` first, then regex on `chunk.cfr_citation`
   - Build query: `f"{sub_q_text} 21 CFR §{section}"`
   - Example: `"FDA protein labeling requirements 21 CFR §101.9"`
   - The section number is grounded in actually-retrieved content (not LLM-invented).
     BGE-M3 sparse weights strongly match any indexed chunk that mentions that section,
     narrowing retrieval to the correct regulatory neighbourhood without an LLM call.
2. **HyDE passage** — only if non-empty and `len >= REFORMULATE_MIN_LEN` (15)
3. **Stepback query** — only if non-empty and `len >= REFORMULATE_MIN_LEN` (15)
4. **`sub_q_text` unchanged** — last resort

**Step 4b — Reformulation Validation**

Before using the reformulated query:
- Empty or `None` → invalid
- `len(stripped) < REFORMULATE_MIN_LEN` (15) → invalid
- `stripped == current_primary.strip()` → invalid (no change)

On invalid: log WARNING with the rejected string, keep `current_primary` unchanged.
The rejected string is still appended to `reformulation_log` for session analysis.

After a valid reformulation:
1. Log to `reformulation_log`
2. Increment `retry_count`
3. Re-run Flow A and Flow B with the new query text
4. Re-run RRF Merge
5. Re-run Cross-Encoder Reranker (then update `best_chunks_seen` per Step 1b)
6. Re-enter CRAG Evaluator with incremented `retry_count`

**Step 5 — LOW_CONFIDENCE (retries exhausted)**

If `retry_count >= CRAG_MAX_RETRIES`:
- Set `verdict = "LOW_CONFIDENCE"`, `caveat_flag = true`
- If `best_chunks_seen` is non-empty AND `best_chunks_seen[0].reranker_score > current top_score`:
  restore `best_chunks_seen` as final chunk pool
- Attach flag `"MAX_RETRIES_EXCEEDED"`
- Continue pipeline — the Skip-No-Evidence Gate (see Sub-Answer Synthesis) decides
  whether to synthesize or skip based on whether `best_chunks_seen` is empty

### Output
```typescript
output: CRAGResult {
  verdict: "CORRECT" | "AMBIGUOUS" | "INCORRECT" | "LOW_CONFIDENCE"
  chunks: RankedChunk[]
  best_chunks_seen: RankedChunk[]    // best chunk pool seen across all retries
  caveat_flag: boolean
  retry_count: number
  reformulation_log: string[]
}
```

### Retry Specification
- Maximum retries: `CRAG_MAX_RETRIES = 3`
- Each retry uses a different strategy (strategies are ordered and not repeated)
- Pipeline never blocks; after max retries the LOW_CONFIDENCE path continues

---

### Component: Cross-Reference Resolution Agent

### Objective
Discover all CFR sections referenced within retrieved chunks — both explicitly and implicitly — and fetch those sections from Qdrant to append to the chunk pool.

### Input
```typescript
input: {
  crag_result: CRAGResult,         // chunks from CRAG Evaluator
  already_fetched: Set<string>     // cfr_section values already in pool (avoid re-fetch)
}
```

### Process

**Layer 1 — Regex Extraction (explicit refs)**
1. For each chunk in `crag_result.chunks`, run regex scan on `chunk.text`
2. Patterns to match:
   - `§\d+\.\d+` (e.g., §101.54)
   - `21 CFR \d+\.\d+` (e.g., 21 CFR 101.9)
   - `part \d+` or `subpart [A-Z]` (less specific, lower priority)
3. Collect all matched section strings, normalize to format `"XXX.XX"`
4. Remove any sections already in `already_fetched`

**Layer 2 — LLM Extraction (implicit refs)**
1. For each chunk, send `chunk.text` to LLM with prompt:
   - "Identify any implicit references to other FDA regulations in this text. Look for phrases like 'as defined in', 'see also', 'pursuant to', 'in accordance with', 'determined under', 'as specified in'. Return only the referenced CFR section numbers as a JSON array. If none, return []."
2. Parse the returned JSON array
3. Add to the set of sections to fetch
4. Remove duplicates and sections already in `already_fetched`
5. Label each with `extraction_method = "llm"`

**Fetch Step**
For each identified section reference (depth 1):
```
qdrant.scroll(
  collection_name="cfr_21",
  scroll_filter={ "cfr_section": { "$eq": "101.9" } },
  with_payload=true,
  limit=10
)
```
This is a direct metadata lookup — not a vector search.

**Recursive Resolution (depth 2)**
1. For each fetched chunk (depth-1 results), repeat Layer 1 (regex) and Layer 2 (LLM) extraction
2. Fetch the newly identified sections
3. Mark fetched chunks with `depth = 2`
4. Do not recurse beyond depth 2

**Deduplication**
Add all fetched chunks to `already_fetched`. Remove any chunk already in the existing pool. Mark all resolved chunks with `is_cross_ref = true` and `cross_ref_source = <originating_section>`.

### Output
```typescript
output: ResolvedChunk[]    // Original CRAG chunks + resolved cross-ref chunks, all tagged
```

### Error Handling
- Qdrant fetch failure for a specific section → log, skip that section, continue
- LLM extraction failure → fall back to regex results only for that chunk, log warning
- If a depth-2 fetch fails → skip, log, do not block

---

### Component: Claim-Level Grounding Verification

### Objective
Verify that every factual claim in the candidate sub-answer is traceable to a specific retrieved chunk. Remove or flag any unsupported claims.

### Input
```typescript
input: {
  candidate_answer: string,          // Draft answer text (generated inline before this step)
  resolved_chunks: ResolvedChunk[]   // Full augmented chunk pool
}
```

Note: The `candidate_answer` is generated by a preliminary LLM call immediately before this component runs, using the same prompt as Sub-Answer Synthesis but without citation enforcement. This draft is what gets verified.

### Process

**Step 1 — Claim Extraction**
1. Send `candidate_answer` to LLM with prompt:
   - "Extract every distinct factual claim from the following text as a JSON array of strings. Each item should be a single atomic statement. Do not include opinions, uncertainty language, or procedural steps — only factual regulatory claims."
2. Parse result as `string[]`

**Step 2 — Claim-to-Chunk Matching**
For each extracted claim:
1. Embed the claim text
2. Run cosine similarity against all chunks in `resolved_chunks` using their embeddings
3. Also run keyword overlap check between claim and chunk text
4. If max similarity score >= 0.65 OR keyword overlap > 50%:
   - Mark claim as `is_supported = true`
   - Record `supporting_chunk_id` and `supporting_cfr_section`
5. If no chunk meets threshold:
   - Mark claim as `is_supported = false`
   - Add claim text to `unverified_claims`

**Step 3 — Answer Pruning**
1. Reconstruct the answer using only supported claims
2. Remove sentences containing unsupported claims entirely
3. If the answer after pruning is empty or fewer than 2 supported claims:
   - Do not return an empty answer
   - Set entire sub-answer to verdict `LOW_CONFIDENCE`
   - Attach flag `"ALL_CLAIMS_UNVERIFIED"`
   - Keep the original candidate answer with all claims marked `[UNVERIFIED]`

### Output
```typescript
output: {
  verified_answer: string,
  claim_verification: ClaimVerification[],
  unverified_claims: string[],
  all_claims_unverified: boolean
}
```

---

### Component: Sub-Answer Synthesis Agent

### Objective
Produce the final structured sub-answer for a single sub-question, grounded strictly in verified and cross-reference resolved chunks.

### Input
```typescript
input: {
  sub_question: SubQuestion,
  crag_result: CRAGResult,
  resolved_chunks: ResolvedChunk[],
  claim_verification_result: {
    verified_answer: string,
    claim_verification: ClaimVerification[],
    unverified_claims: string[],
    all_claims_unverified: boolean
  }
}
```

### Process

**Step 0 — Skip-No-Evidence Early Return (gate before synthesis)**

Immediately after the CRAG loop, before any synthesis work:

```python
if crag_result.verdict == "LOW_CONFIDENCE" and not crag_result.best_chunks_seen:
    return SubAnswer(
        sub_question_id=...,
        sub_question_text=...,
        answer="",
        confidence=0.0,
        crag_verdict="LOW_CONFIDENCE",
        caveat_flag=True,
        chunks_used=[],
        cross_refs_resolved=[],
        claim_verification=[],
        unverified_claims=[],
        citations=[],
        caveats=["LOW_CONFIDENCE_RETRIEVAL"],
        flags=["MAX_RETRIES_EXCEEDED", "SKIPPED_NO_EVIDENCE"],
        reformulation_log=crag_result.reformulation_log,
        skipped=True,
    )
```

`best_chunks_seen` empty after `MAX_RETRIES` means the retrieval pipeline genuinely
found no supporting regulatory text. Synthesizing from nothing produces a fabricated
answer with `confidence=0.2` that:
- Drags down the final-answer confidence average
- Triggers spurious contradiction checks in Stage 6 against claims with no evidence

`skipped=true` is the sentinel used by Stage 6 and Stage 7 to exclude these sub-answers
from conflict detection and confidence averaging while still passing them through for
count reporting.

**Step 1 — Standard synthesis** (runs only when `skipped` is false)
1. Use `claim_verification_result.verified_answer` as the base answer text
2. Inline §citations: for each sentence, append `cfr_section` of `supporting_chunk_id` as `[§X.XX]`
3. Compute `confidence` score using the Confidence Scoring table above
4. Build `chunks_used` from all `ResolvedChunk` objects referenced by `ClaimVerification.supporting_chunk_id`
5. Build `cross_refs_resolved` from all `ResolvedChunk` where `is_cross_ref = true`
6. Set `caveat_flag` from `crag_result.caveat_flag`
7. Populate `caveats` array:
   - Add `"AMBIGUOUS_EVIDENCE"` if `caveat_flag = true`
   - Add `"LOW_CONFIDENCE_RETRIEVAL"` if `crag_result.verdict = "LOW_CONFIDENCE"`
   - Add `"UNVERIFIED_CLAIMS_REMOVED"` if `unverified_claims` is non-empty
8. Populate `flags` array from all flag conditions encountered in this sub-question's processing
9. Set `skipped = false`

### Output
```typescript
output: SubAnswer
```

---

## Stage 6 — Consistency & Conflict Detection Agent

### Objective
Identify and resolve contradictions across sub-answers, filtering out evidence-free
and weak sub-answers before conflict analysis, and separately detecting cross-domain
retrieval failures (which are not regulatory conflicts).

### Input
```typescript
input: SubAnswer[]    // All sub-answers including skipped=true ones
```

### Process

**Step 1 — Active / Skipped Split**

```python
active  = [sa for sa in sub_answers if not sa.skipped]
skipped = [sa for sa in sub_answers if sa.skipped]
```

Skipped sub-answers bypass all conflict detection and are appended unchanged to
`resolved_answers`. Log skipped count at INFO level.

If `len(active) <= 1`: return immediately with `resolved_answers = active + skipped`,
`unresolved_conflicts = []`, `domain_mismatches = []`.

**Step 2 — Confidence Threshold Filter**

```python
eligible          = [sa for sa in active if sa.confidence >= CONSISTENCY_CONF_THRESHOLD]  # 0.3
low_conf_excluded = [sa for sa in active if sa.confidence <  CONSISTENCY_CONF_THRESHOLD]
```

Sub-answers below 0.3 have `claim_verification` entries based on noise chunks. LLM
contradiction calls against those claims waste tokens and produce spurious conflicts.

`low_conf_excluded` sub-answers:
- Each gets flag `"LOW_CONF_EXCLUDED_FROM_CONFLICT_CHECK"` appended
- Pass through to `resolved_answers` unchanged (they still reach Stage 7)
- NOT included in contradiction LLM calls

If `len(eligible) <= 1` after this filter: return early with no conflicts, returning
`resolved_answers = eligible + low_conf_excluded + skipped`.

**Step 3 — Domain Coherence Pre-Check**

When retrieval returns chunks from the wrong FDA domain, two sub-answers may cite
incompatible domains. This is a retrieval failure, not a regulatory conflict, and must
be reported separately so the frontend can render appropriate messaging.

Part-number → domain lookup (`_PART_RANGES`):

| Part range | Domain |
|---|---|
| 1–99 | administrative |
| 100–199 | food |
| 200–499 | drug |
| 500–599 | animal |
| 600–699 | biological |
| 700–799 | cosmetic |
| 800–999 | device |
| 1000–1099 | device |
| 1100–1199 | tobacco |
| 1200–1299 | administrative |

Helpers: `_part_number_from_section(section)` (regex), `_domain_from_part(int)`,
`_dominant_domain(sub_answer)` (Counter.most_common), `_domains_compatible(d1, d2)`
(True if same, either None, or either is "administrative").

For each eligible pair `(i, j)`:
- If `_domains_compatible(...)` is False:
  - Add `(i, j)` to `domain_mismatch_pairs`
  - Append `DomainMismatchReport` to `domain_mismatches`
  - Flag both sub-answers with `"RETRIEVAL_DOMAIN_MISMATCH"` (idempotent)

Log WARNING with mismatch count if any found.

**Step 4 — Claim Cross-Examination**

```python
for i, j in itertools.combinations(range(len(eligible)), 2):
    if (i, j) in domain_mismatch_pairs:
        continue   # skip — retrieval failure, not a conflict
    # run pairwise LLM contradiction check between eligible[i] and eligible[j]
```

1. Extract all supported factual claims for each pair
2. For each claim pair across different sub-answers, run LLM comparison:
   - "Do these two regulatory claims contradict each other? Answer YES or NO, then briefly explain."
3. Flag pairs where LLM returns YES as potential conflicts

**Step 5 — Conflict Identification**
For each flagged contradiction pair:
1. Identify `cfr_section` values cited by each claim
2. Build `ConflictReport` with `sub_question_ids`, `conflicting_sections`,
   `description`, `resolved = false`

**Step 6 — Conflict Resolution (automatic)**

Rule 1 — Section specificity:
- More specific section (e.g., §101.54) overrides more general (e.g., Part 101 intro)
- If resolved: set `resolved = true`, `winning_section`, `resolution = "Resolved by section specificity"`

Rule 2 — Recency:
- Compare `effective_date` of conflicting chunks; more recent wins
- If resolved: set `resolved = true`, `winning_section`, `resolution = "Resolved by effective_date"`

Rule 3 — Unresolvable:
- Set `resolved = false`, keep in `unresolved_conflicts`

**Step 7 — Sub-Answer Update**
For resolved conflicts: update losing sub-answer's `answer` and `caveats`. Add flag `"CONFLICT_RESOLVED"`.

### Output
```typescript
output: {
  resolved_answers: SubAnswer[],          // eligible + low_conf_excluded + skipped
  unresolved_conflicts: ConflictReport[],
  domain_mismatches: DomainMismatchReport[]
}
```

`domain_mismatches` is also stored in `ComplianceState.domain_mismatches` and always
returned (empty list when no mismatches).

### Error Handling
- LLM claim comparison failure → skip that pair, log warning, do not block
- All comparisons fail → pass sub-answers through unmodified, attach flag `"CONSISTENCY_CHECK_FAILED"` to final answer

---

## Stage 7 — Final Synthesis Agent

### Objective
Merge all resolved sub-answers into a single structured compliance response. Every
claim must carry an inline §citation. Apply weighted confidence scoring, designate a
primary sub-answer to drive the ruling, derive numeric/threshold answers explicitly,
and surface all flags, conflicts, and confidence signals including partial-information
context.

### Input
```typescript
input: {
  resolved_answers: SubAnswer[],           // includes skipped and low-conf
  unresolved_conflicts: ConflictReport[],
  domain_mismatches: DomainMismatchReport[],
  original_query: string
}
```

### Process

**Step 0 — Filter Skipped Sub-Answers**

```python
all_answers      = state.get("resolved_answers", state.get("sub_answers", []))
resolved_answers = [sa for sa in all_answers if not sa.skipped]
skipped_count    = len(all_answers) - len(resolved_answers)
```

All subsequent steps operate on `resolved_answers` only. If `resolved_answers` is empty
(every sub-question was skipped): return `ruling = "NO"` with `confidence_level = "LOW"`.

**Step 1 — Partial Information Note (`_build_partial_info_note`)**

```python
total    = len(all_answers)
answered = len(resolved_answers)
reliable = sum(1 for sa in resolved_answers if sa.confidence >= 0.3)
```

Generates a human-readable note when any gap exists, e.g.:
> "1 of 3 sub-question(s) found no supporting evidence; 1 of 2 answered
>  sub-question(s) had low retrieval confidence (<30%). Ruling is based on
>  partial information."

Returns `""` when all sub-questions were answered with acceptable confidence.

**Step 2 — Primary Sub-Answer Designation**

```python
primary = max(resolved_answers, key=lambda sa: sa.confidence)
```

`_build_ruling_user_content(query, resolved_answers, primary, partial_info)` formats
sub-answers for the ruling LLM with explicit labels:

```
[PRIMARY — rule primarily from this] Sub-question (confidence=0.63): ...
[SUPPORTING] Sub-question (confidence=0.21): ...

Context note: {partial_info if present}
```

**Step 3 — Ruling Determination**

Ruling LLM system prompt includes two key clauses:

**PRIORITY clause:** "Rule primarily from the [PRIMARY] sub-answer. Use SUPPORTING
sub-answers only if their content is consistent and non-contradictory with the primary."

**REASONING STEP clause (conditional):** "If the PRIMARY sub-answer retrieved relevant
numeric values (DRV, percentage threshold, serving size limit), derive the answer
mathematically and state the calculation in `reasoning_steps`. Example: 'Protein
DRV=50g. Excellent source threshold=20% DV=10g. 20g>10g → YES.' Only compute from
values explicitly present in the sub-answers. Do NOT invent numbers. If no calculation
is needed, return `reasoning_steps` as []."

LLM produces:
- `ruling`: `"YES"`, `"NO"`, or `"CONDITIONAL"`
- `ruling_summary` (1–2 sentences)
- `reasoning_steps`: string array (empty `[]` for qualitative rulings)

`max_tokens` set to 1200 (raised from 1024) to accommodate reasoning output.

**Step 4 — Requirements Assembly**
1. Extract all compliance requirements across `resolved_answers`
2. Deduplicate overlapping requirements
3. Format each as `{ item: string, citation: "[§X.XX]" }`

**Step 5 — Weighted Confidence Level Computation**

```python
strong = [sa for sa in resolved_answers if sa.confidence >= SYNTHESIZER_WEAK_THRESHOLD]  # 0.3
weak   = [sa for sa in resolved_answers if sa.confidence <  SYNTHESIZER_WEAK_THRESHOLD]
pool   = strong if strong else resolved_answers
avg    = mean(sa.confidence for sa in pool)
```

Apply thresholds to `avg` → `confidence_level`. The `confidence_explanation` reports
the count of strong sub-answers and notes when weak ones were excluded from scoring.

No automatic downgrade for skipped sub-questions: the `partial_information_note` is
informational, not a penalty.

**Step 6 — Regulation Excerpts**
1. Collect all `ResolvedChunk` objects used across `resolved_answers`
2. Deduplicate by `cfr_section`
3. Include verbatim `text` and `cfr_section` for each unique section

**Step 7 — Full Answer Assembly**

Assemble `FinalAnswer` with all fields populated:
- `ruling` and `ruling_summary`
- `reasoning_steps` (from ruling LLM; `[]` when not applicable)
- `requirements` with §citations
- `confidence_level` and `confidence_explanation`
- `regulation_excerpts`
- `unresolved_conflicts` (pass through from Stage 6)
- `domain_mismatches` (pass through from Stage 6; distinguishes retrieval failures
  from regulatory conflicts)
- `low_confidence_sub_questions` (sub-questions where `confidence < 0.5`)
- `caveats` (aggregated from `resolved_answers`, deduplicated)
- `compliance_checklist` (LLM-generated numbered list of what the user must do)
- `all_citations` (deduplicated list of every `cfr_section` referenced)
- `partial_information_note` (string or null; see Step 1)

`_build_query_response(final_answer, unresolved_conflicts, domain_mismatches)` renders:
- `reasoning_steps` between `ruling_summary` and `requirements` as `**Reasoning:** - <step>` lines
- `partial_information_note` as an italic footnote at the bottom

**Inline citation rule:** Every sentence containing a factual regulatory claim must
end with `[§X.XX]`. No factual claim appears without a citation. The LLM is explicitly
instructed to enforce this.

### Output
```typescript
output: FinalAnswer
```

### Error Handling
- LLM synthesis failure → retry once
- Second failure → return raw sub-answers as a numbered list with original citations, attach flag `"SYNTHESIS_FAILED"`

---

## Index-Time Enhancement — HyPE

### Objective
Improve retrieval recall at query time by embedding hypothetical questions per chunk at index time. Complements HyDE (query-time) with no added query latency.

### Process (runs once during indexing, not at query time)

For each chunk being indexed:
1. Send `chunk.text` to LLM with prompt:
   - "Generate 4 distinct compliance questions that a food manufacturer might ask that this FDA regulation text directly answers. Output as a JSON array of strings."
2. Parse the 4 generated questions
3. Generate embeddings for each question using the primary embedding model
4. Store these embeddings as named vectors `hyde_q_0` through `hyde_q_3` in the same Qdrant point as the chunk's primary embedding

### At Query Time
In Flow A and Flow B, in addition to searching against the primary chunk vector, also search against the HyPE question vectors:
```
qdrant.search(
  query_vector=("hyde_q_0", query_embedding),
  ...
)
```
Merge HyPE results with primary results before RRF.

---

## Agent Summary

| Component | Stage | Input | Output | Retry / Fallback |
|---|---|---|---|---|
| Query Analysis Agent | 1 | `UserQuery` | `AnalyzedQuery` (+ hoisted clarification keys) | 1 retry on LLM failure |
| Sub-Question Decomposition Agent | 2 | `AnalyzedQuery` (incl. `fda_domain`, `is_multi_part`) | `SubQuestion[]` with validated variants | 1 retry on LLM; per-variant fallback to sub-q text |
| Query Expansion | Per sub-Q | `SubQuestion` | Embedded `QueryVariants` | None |
| CFR Part Classifier | Per sub-Q (Flow A) | sub-q text + domain-filtered part index + `explicit_refs` hint | predicted parts + confidence | Falls back to no-context if part index missing |
| Flow A — Routed Retrieval | Per sub-Q | `QueryVariants` + embeddings | `RetrievedChunk[]` | 1 retry on Qdrant failure; empty set if classifier skips |
| Flow B — Global Retrieval | Per sub-Q | `QueryVariants` + embeddings | `RetrievedChunk[]` | 1 retry on Qdrant failure |
| RRF Merge & Deduplication | Per sub-Q | Flow A + Flow B results | `RankedChunk[]` (~80) | None |
| Cross-Encoder Reranker | Per sub-Q | `RankedChunk[]` (~80) | `RankedChunk[]` (top 12) | 1 retry; fallback to RRF order |
| CRAG Evaluator | Per sub-Q | Top-K chunks + sub-question + `best_chunks_seen` | `CRAGResult` (incl. `best_chunks_seen`) | Max 3 reformulation retries; retry 2 = section-anchor strategy |
| Skip-No-Evidence Gate | Per sub-Q | `CRAGResult` + `best_chunks_seen` | `SubAnswer(skipped=true)` OR continues | n/a — single check |
| Cross-Reference Resolution Agent | Per sub-Q | `CRAGResult` | `ResolvedChunk[]` | Per-section skip on failure |
| Claim-Level Grounding Verification | Per sub-Q | Draft answer + `ResolvedChunk[]` | Verified answer + `ClaimVerification[]` | None; pruning handles failures |
| Sub-Answer Synthesis Agent | Per sub-Q | All above outputs | `SubAnswer` | 1 retry on LLM failure |
| Consistency & Conflict Detection Agent | 6 | `SubAnswer[]` (active + skipped) | `{ resolved_answers, unresolved_conflicts, domain_mismatches }` | Skip failed pairs; continue |
| Final Synthesis Agent | 7 | Resolved answers + conflicts + domain mismatches | `FinalAnswer` (weighted conf, primary sub-answer, `reasoning_steps`, `partial_information_note`) | 1 retry; fallback to raw sub-answers |
