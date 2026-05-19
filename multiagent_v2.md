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
CRAG_THRESHOLD_HIGH = 0.70      # Min reranker score for CORRECT verdict
CRAG_THRESHOLD_LOW = 0.45       # Min reranker score for AMBIGUOUS verdict
HYDE_GENERATION_TEMP = 0.7      # LLM temperature for HyDE passage generation
STEPBACK_GENERATION_TEMP = 0.3  # LLM temperature for step-back query generation
PART_CLASSIFIER_CONFIDENCE_MIN = 0.60  # Below this, skip Flow A and use Flow B only
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
  caveats: string[]
  flags: string[]                    // e.g. ["LOW_CONFIDENCE", "AMBIGUOUS_EVIDENCE"]
}

interface ConflictReport {
  sub_question_ids: string[]
  conflicting_sections: string[]
  description: string
  resolved: boolean
  resolution: string | null          // How it was resolved, if resolved
  winning_section: string | null
}

interface FinalAnswer {
  ruling: "YES" | "NO" | "CONDITIONAL"
  ruling_summary: string
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
  low_confidence_sub_questions: string[]
  caveats: string[]
  compliance_checklist: string[]
  all_citations: string[]            // Deduplicated list of all §refs used
}
```

---

## Confidence Scoring Rules

Applied in Sub-Answer Synthesis Agent. Used to populate `SubAnswer.confidence`.

| CRAG Verdict | Unverified Claims | Confidence Score |
|---|---|---|
| CORRECT | 0 | 1.0 |
| CORRECT | > 0 | 0.8 |
| AMBIGUOUS | 0 | 0.6 |
| AMBIGUOUS | > 0 | 0.4 |
| LOW_CONFIDENCE | any | 0.2 |

Final answer `confidence_level` is derived from the average of all `SubAnswer.confidence` values:
- `>= 0.8` → HIGH
- `>= 0.5` → MEDIUM
- `< 0.5` → LOW

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
  │          ├─ CORRECT ──────────────────────────┐          │
  │          ├─ AMBIGUOUS → supplemental search ──┤          │
  │          └─ INCORRECT → reformulate & retry ──┘          │
  │                  │  CRAGResult                            │
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
  ▼ (collect all SubAnswer objects)
[Stage 6] Consistency & Conflict Detection Agent
  │  Output: {resolved_answers: SubAnswer[], conflicts: ConflictReport[]}
  │
  ▼
[Stage 7] Final Synthesis Agent
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

**Step 1 — Decomposition**
1. Send `AnalyzedQuery` to LLM
2. LLM produces the minimum number of atomic sub-questions needed to fully answer the original query
3. Each sub-question must have a single, non-overlapping intent
4. Each sub-question must be self-contained (independently answerable without the others)
5. Assign a unique `id` (uuid) to each sub-question

**Step 2 — Query Variant Generation (per sub-question)**

For each sub-question, run three separate LLM calls in parallel:

**Variant 1 — Primary query**
- Input: sub-question text as-is
- Output: `variants.primary` — the sub-question rephrased for optimal retrieval (concise, keyword-rich, removes conversational phrasing)

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

### Output
```typescript
output: SubQuestion[]
```
Each `SubQuestion` contains `id`, `text`, `variants` (primary, hyde_passage, stepback), `source_query`.

### Error Handling
- If decomposition LLM call fails → retry once
- If variant generation fails for one variant → use `primary` for all three, log warning
- If the original query is already atomic (single intent, `is_multi_part = false`) → produce a single `SubQuestion` with `text = AnalyzedQuery.raw_text`

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

**Step 1 — CFR Part Classification**
1. Send `sub_question_text` (and any `explicit_refs` from `AnalyzedQuery`) to the Part classifier
2. Part classifier returns predicted `cfr_part[]` values and a confidence score
3. If classifier confidence < `PART_CLASSIFIER_CONFIDENCE_MIN` (0.60):
   - Skip Flow A entirely
   - Log: `"Flow A skipped: classifier confidence below threshold"`
   - Return empty result set; Flow B covers full retrieval

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
  reformulation_log: string[]            // starts as []
}
```

### Process

**Step 1 — Verdict Determination**

Evaluate the `reranker_score` values of `top_k_chunks`:

| Condition | Verdict |
|---|---|
| Top chunk score `>= CRAG_THRESHOLD_HIGH` AND at least 2 chunks `>= CRAG_THRESHOLD_LOW` | CORRECT |
| Top chunk score `>= CRAG_THRESHOLD_LOW` but does not meet CORRECT criteria | AMBIGUOUS |
| All chunk scores `< CRAG_THRESHOLD_LOW` | INCORRECT |
| `retry_count >= CRAG_MAX_RETRIES` and still INCORRECT | LOW_CONFIDENCE |

**Step 2 — CORRECT path**
- Set `verdict = "CORRECT"`, `caveat_flag = false`
- Pass `top_k_chunks` to Cross-Reference Resolution Agent unchanged

**Step 3 — AMBIGUOUS path**
1. Run a supplemental broad search: Flow B with `variants.primary` only, no Part filter, `limit = FLOW_B_TOP_N`
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
| 2 | Variant switch: use `variants.hyde_passage` as query if not yet tried; else use `variants.stepback` |

After reformulation:
1. Log the reformulation attempt to `reformulation_log`
2. Increment `retry_count`
3. Re-run Flow A and Flow B with the new query text
4. Re-run RRF Merge
5. Re-run Cross-Encoder Reranker
6. Re-enter CRAG Evaluator with `retry_count` incremented

If `retry_count >= CRAG_MAX_RETRIES`:
- Set `verdict = "LOW_CONFIDENCE"`, `caveat_flag = true`
- Use whatever chunks are available (even low-scoring)
- Attach flag `"MAX_RETRIES_EXCEEDED"` to the result
- Continue pipeline — do not block

### Output
```typescript
output: CRAGResult {
  verdict: "CORRECT" | "AMBIGUOUS" | "INCORRECT" | "LOW_CONFIDENCE"
  chunks: RankedChunk[]
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
1. Use `claim_verification_result.verified_answer` as the base answer text
2. Inline §citations: for each sentence in the answer, append the `cfr_section` of its `supporting_chunk_id` in format `[§X.XX]`
3. Compute `confidence` score using the Confidence Scoring table above
4. Build `chunks_used` list from all `ResolvedChunk` objects whose `chunk_id` appears in at least one `ClaimVerification.supporting_chunk_id`
5. Build `cross_refs_resolved` list from all `ResolvedChunk` where `is_cross_ref = true`
6. Set `caveat_flag` from `crag_result.caveat_flag`
7. Populate `caveats` array:
   - Add `"AMBIGUOUS_EVIDENCE"` if `caveat_flag = true`
   - Add `"LOW_CONFIDENCE_RETRIEVAL"` if `crag_result.verdict = "LOW_CONFIDENCE"`
   - Add `"UNVERIFIED_CLAIMS_REMOVED"` if `unverified_claims` is non-empty
8. Populate `flags` array from all flag conditions encountered in this sub-question's processing

### Output
```typescript
output: SubAnswer
```

---

## Stage 6 — Consistency & Conflict Detection Agent

### Objective
Identify and resolve contradictions across all sub-answers before final synthesis. Ensure conflicting regulatory sections are reconciled rather than silently passed to the user.

### Input
```typescript
input: SubAnswer[]    // All completed sub-answers from the parallel loop
```

### Process

**Step 1 — Claim Cross-Examination**
1. Extract all supported factual claims across all `SubAnswer` objects
2. For each pair of claims from different sub-answers, run LLM comparison:
   - "Do these two regulatory claims contradict each other? Answer YES or NO, then briefly explain."
3. Flag pairs where LLM returns YES as potential conflicts

**Step 2 — Conflict Identification**
For each flagged contradiction pair:
1. Identify the `cfr_section` values cited by each claim
2. Build a `ConflictReport` with:
   - `sub_question_ids` of the affected sub-answers
   - `conflicting_sections` (the two §refs in conflict)
   - `description` (LLM-generated explanation of the contradiction)
   - `resolved = false` initially

**Step 3 — Conflict Resolution (automatic)**

Apply resolution rules in order:

Rule 1 — Section specificity:
- A more specific section (lower-level, e.g., §101.54) overrides a more general one (e.g., Part 101 introductory text)
- Specificity determined by section number depth and `section_title`
- If resolved: set `resolved = true`, `winning_section`, `resolution = "Resolved by section specificity"`

Rule 2 — Recency:
- Compare `effective_date` of the conflicting chunks
- More recent section wins
- If resolved: set `resolved = true`, `winning_section`, `resolution = "Resolved by effective_date"`

Rule 3 — Unresolvable:
- If neither rule applies or both sections have equal priority:
- Set `resolved = false`
- Keep in `unresolved_conflicts` for surfacing in the final answer

**Step 4 — Sub-Answer Update**
For resolved conflicts: update the losing sub-answer's `answer` text and `caveats` to reflect the resolution. Add flag `"CONFLICT_RESOLVED"`.

### Output
```typescript
output: {
  resolved_answers: SubAnswer[],
  unresolved_conflicts: ConflictReport[]
}
```

### Error Handling
- LLM claim comparison failure → skip that pair, log warning, do not block
- All comparisons fail → pass sub-answers through unmodified, attach flag `"CONSISTENCY_CHECK_FAILED"` to final answer

---

## Stage 7 — Final Synthesis Agent

### Objective
Merge all resolved sub-answers into a single structured compliance response. Every claim must carry an inline §citation. Surface all flags, conflicts, and confidence signals.

### Input
```typescript
input: {
  resolved_answers: SubAnswer[],
  unresolved_conflicts: ConflictReport[],
  original_query: string
}
```

### Process

**Step 1 — Ruling Determination**
1. Send all sub-answers to LLM
2. LLM produces a direct `ruling`: `"YES"`, `"NO"`, or `"CONDITIONAL"`
3. LLM produces a `ruling_summary` (1–2 sentences)

**Step 2 — Requirements Assembly**
1. Extract all compliance requirements across sub-answers
2. Deduplicate overlapping requirements
3. Format each as `{ item: string, citation: "[§X.XX]" }`

**Step 3 — Confidence Level Computation**
1. Average `confidence` scores across all `SubAnswer` objects
2. Apply Confidence Scoring thresholds to produce `confidence_level`
3. Generate `confidence_explanation` that references which sub-questions were LOW_CONFIDENCE if any

**Step 4 — Regulation Excerpts**
1. Collect all `ResolvedChunk` objects used across all sub-answers
2. Deduplicate by `cfr_section`
3. Include verbatim `text` and `cfr_section` for each unique section

**Step 5 — Full Answer Assembly**
Assemble `FinalAnswer` with all fields populated:
- `ruling` and `ruling_summary`
- `requirements` with §citations
- `confidence_level` and `confidence_explanation`
- `regulation_excerpts`
- `unresolved_conflicts` (pass through from Stage 6)
- `low_confidence_sub_questions` (sub-questions where `confidence < 0.5`)
- `caveats` (aggregated from all sub-answers, deduplicated)
- `compliance_checklist` (LLM-generated numbered list of what the user must do to comply)
- `all_citations` (deduplicated list of every `cfr_section` referenced across all sub-answers)

**Inline citation rule:** Every sentence in the final answer that contains a factual regulatory claim must end with `[§X.XX]`. No factual claim appears without a citation. The LLM is explicitly instructed to enforce this.

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

| Component | Stage | Input | Output | Retry |
|---|---|---|---|---|
| Query Analysis Agent | 1 | `UserQuery` | `AnalyzedQuery` | 1 retry on LLM failure |
| Sub-Question Decomposition Agent | 2 | `AnalyzedQuery` | `SubQuestion[]` | 1 retry on LLM failure |
| Query Expansion | Per sub-Q | `SubQuestion` | Embedded `QueryVariants` | None |
| Flow A — Routed Retrieval | Per sub-Q | `QueryVariants` + embeddings | `RetrievedChunk[]` | 1 retry on Qdrant failure |
| Flow B — Global Retrieval | Per sub-Q | `QueryVariants` + embeddings | `RetrievedChunk[]` | 1 retry on Qdrant failure |
| RRF Merge & Deduplication | Per sub-Q | Flow A + Flow B results | `RankedChunk[]` (~80) | None |
| Cross-Encoder Reranker | Per sub-Q | `RankedChunk[]` (~80) | `RankedChunk[]` (top 12) | 1 retry; fallback to RRF order |
| CRAG Evaluator | Per sub-Q | Top-K chunks + sub-question | `CRAGResult` | Max 3 reformulation retries |
| Cross-Reference Resolution Agent | Per sub-Q | `CRAGResult` | `ResolvedChunk[]` | Per-section skip on failure |
| Claim-Level Grounding Verification | Per sub-Q | Draft answer + `ResolvedChunk[]` | Verified answer + `ClaimVerification[]` | None; pruning handles failures |
| Sub-Answer Synthesis Agent | Per sub-Q | All above outputs | `SubAnswer` | 1 retry on LLM failure |
| Consistency & Conflict Detection Agent | 6 | `SubAnswer[]` | Resolved answers + `ConflictReport[]` | Skip failed pairs, continue |
| Final Synthesis Agent | 7 | Resolved answers + conflicts | `FinalAnswer` | 1 retry; fallback to raw sub-answers |
