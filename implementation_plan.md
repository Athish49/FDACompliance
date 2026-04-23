# FDA Food Labeling Compliance RAG — Implementation Plan

> **Version**: 2.0  
> **Last Updated**: April 2026  
> **Status**: Phases 1–4 Complete; Phase 5 (Observability & Evaluation) pending

> **Note**: This document reflects what was *actually implemented*. Key deviations from v1.0: (1) single BGE-M3 model only — Kanon 2 and HyDE dropped; (2) single Qdrant collection with named vectors — no separate sparse/kanon collections; (3) Neo4j optional and off by default — cross-reference and definition resolution use Qdrant filtered search; (4) Celery/Redis/Langfuse/RAGAS not yet implemented.  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Deliverables](#3-deliverables)
4. [Architecture Overview Diagram](#4-architecture-overview-diagram)
5. [Embedding Model Decision](#5-embedding-model-decision)
6. [Layer 1 — Data Source](#6-layer-1--data-source)
7. [Layer 2 — Ingestion Pipeline](#7-layer-2--ingestion-pipeline)
8. [Layer 3 — Hybrid Storage (Qdrant + Neo4j)](#8-layer-3--hybrid-storage-qdrant--neo4j)
9. [Layer 4 — Retrieval Engine](#9-layer-4--retrieval-engine)
10. [Layer 5 — Multi-Agent Reasoning Layer](#10-layer-5--multi-agent-reasoning-layer)
11. [Flow A — Query Flow](#11-flow-a--query-flow)
12. [Flow B — Document Violation Analysis Flow](#12-flow-b--document-violation-analysis-flow)
13. [Layer 6 — Response Synthesis](#13-layer-6--response-synthesis)
14. [Observability & Evaluation](#14-observability--evaluation)
15. [Technology Stack Summary](#15-technology-stack-summary)
16. [Data Models & Schemas](#16-data-models--schemas)
17. [Implementation Phasing](#17-implementation-phasing)
18. [Local Hardware Requirements](#18-local-hardware-requirements)
19. [Directory Structure](#19-directory-structure)

---

## 1. Project Overview

**Project Name**: FDA Labeling Compliance Intelligence Platform  
**Domain**: Regulatory Compliance — Food & Drug Administration  
**Target Regulation**: 21 CFR Part 101 (Food Labeling), Title 21, Chapter I  
**Primary Users**: Food manufacturers, packaging teams, label designers, import/export compliance officers, QA teams, regulatory consultants  

### What This System Does

This is a **Retrieval-Augmented Generation (RAG) system** backed by a knowledge graph that:

1. **Answers natural language compliance questions** about FDA food labeling regulations and returns grounded, cited answers with CFR section references.
2. **Analyzes uploaded documents** (food labels, product spec sheets) and returns a structured point-wise list of detected regulatory violations, their severity, and remediation suggestions.

### Why This Matters

Every food brand that sells packaged goods in the United States must comply with 21 CFR Part 101. Labeling errors are one of the top causes of FDA warning letters and product recalls. Tens of thousands of brands reformulate or repackage every year and need to continuously verify compliance. This system assists compliance managers, packaging engineers, and regulatory affairs teams with their daily queries — work that currently requires expensive consultants or manual research.

---

## 2. Problem Statement

### The Core Challenge

21 CFR Part 101 is a dense, hierarchical regulatory document with deep cross-references. A user asking *"Can I label my product as 'low fat' and 'heart healthy'?"* needs:

- §101.62 — definition of "low fat" nutrient content claim
- §101.13 — general rules for nutrient content claims
- §101.76 — approved health claim for cardiovascular disease
- §101.14 — general requirements for health claims (linked from §101.76)

Standard single-vector RAG retrieval would likely return only one of these sections. This system is designed to retrieve **all of them**, traverse their cross-references, and assemble a complete, accurate answer.

### The Secondary Problem

Companies produce food labels that may unintentionally violate regulations — using unapproved health claim language, omitting required disclosures, misstating serving sizes, etc. There is no easy automated tool to scan a label document against the full regulatory corpus and surface every violation with its specific CFR reference.

---

## 3. Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| D1 | Ingestion pipeline | Automated eCFR XML download, contextual chunking, dual-encoding, graph extraction |
| D2 | Hybrid vector + graph store | Qdrant (dense + sparse) + Neo4j (cross-reference graph) |
| D3 | Retrieval engine | HyDE + hybrid search + cross-encoder reranking + graph expansion |
| D4 | Multi-agent reasoning | LangGraph state machine with Planner, Definition Resolver, Synthesizer, Verifier, Conflict Detector |
| D5 | Query flow API | FastAPI endpoint for natural language compliance questions |
| D6 | Document analysis API | FastAPI endpoint accepting uploaded label docs; returns structured violation report |
| D7 | Evaluation suite | RAGAS-based golden test set of 100 Q&A pairs with citation validation |
| D8 | Observability | Langfuse tracing for all pipeline steps |
| D9 | Docker Compose | Full local stack: Qdrant + Neo4j + Redis + FastAPI + Langfuse |

---

## 4. Architecture Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA SOURCE                                                         │
│  eCFR API — 21 CFR Part 101 · XML/JSON · No auth required          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  INGESTION PIPELINE                                                  │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │ XML parser  │──▶│ Contextual       │──▶│ Metadata tagger      │  │
│  │ (lxml)      │   │ chunker          │   │ part·subpart·section │  │
│  └─────────────┘   └──────────────────┘   └──────────┬───────────┘  │
│                                                       │              │
│                    ┌──────────────────────────────────▼────────┐    │
│                    │  BGE-M3 + Kanon 2 Embedder (dual model)   │    │
│                    │  Dense vector + SPLADE sparse weights      │    │
│                    └──────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Graph extractor — parse §ref patterns → Neo4j edge list      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  HYBRID STORAGE  (2 databases only)                                  │
│  ┌────────────────────────────┐   ┌────────────────────────────────┐ │
│  │ Qdrant                     │   │ Neo4j                          │ │
│  │ · Dense vectors            │   │ · Section nodes                │ │
│  │ · Sparse SPLADE vectors    │   │ · references / defined_in /    │ │
│  │ · Chunk text payload       │   │   exempted_by edges            │ │
│  │ · All metadata             │   │ · Standards registry           │ │
│  └────────────────────────────┘   └────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  RETRIEVAL ENGINE                                                    │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────────────────┐  │
│  │Query decom-│  │HyDE expander │  │Qdrant hybrid search         │  │
│  │poser       │  │(hypothetical │  │(dense + sparse, single call) │  │
│  │            │  │ doc embed)   │  │                             │  │
│  └──────┬─────┘  └──────┬───────┘  └──────────────┬────────────┘  │
│         └───────────────┴──────────────────────────┘               │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │  RRF rank fusion → bge-reranker-v2-m3 cross-encoder reranker   │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │  Neo4j cross-reference resolver (1-hop graph expansion)        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  MULTI-AGENT LAYER  (LangGraph state machine)                        │
│                                                                      │
│             ┌─────────────────────────────────┐                     │
│             │  Orchestrator / Planner agent   │                     │
│             └──┬──────┬──────┬──────┬────────┘                     │
│                │      │      │      │                               │
│         ┌──────▼┐ ┌───▼──┐ ┌▼─────┐ ┌▼──────────┐                │
│         │Defini-│ │Synth-│ │Verif-│ │Conflict   │                │
│         │tion   │ │esizer│ │ier   │ │detector   │                │
│         │resolv.│ │agent │ │agent │ │agent      │                │
│         └───────┘ └──────┘ └──────┘ └───────────┘                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Two output flows
            ┌──────────────────┴──────────────────┐
            │                                      │
┌───────────▼────────────┐         ┌──────────────▼────────────────┐
│  FLOW A: Query answer  │         │  FLOW B: Document violation   │
│  Answer + CFR cites +  │         │  analysis                     │
│  confidence score      │         │  Doc parser → Claim extractor │
└────────────────────────┘         │  → Reg mapper → Gap analyzer  │
                                   │  → Violation classifier       │
                                   │  → Structured report          │
                                   └───────────────────────────────┘
```

---

## 5. Embedding Model Decision

### The Challenge: FDA Text is Dual-Domain

21 CFR Part 101 sits at the intersection of two specialized vocabularies:

1. **Legal/regulatory language** — "nutrient content claim", "safe harbor level", "misbranded", "adulterated", "characterizing flavor", "reference amount customarily consumed (RACC)"
2. **Biomedical/chemical language** — "saturated fatty acids", "trans fatty acids", "cholesterol", "thiamin", "riboflavin", "niacin", "folate", "DV (Daily Value)", "mg/dL", "omega-3 fatty acids", "phytosterols"

General-purpose embedding models like `text-embedding-3-large` or `e5-large-v2` do not have deep representations for either vocabulary. They treat "fat" as a general word, not as the regulated term with a specific threshold (≤3g per RACC) that triggers a "low fat" claim eligibility.

### Model Comparison

| Model | Domain | Context Window | Local? | Sparse? | Verdict for this project |
|-------|--------|---------------|--------|---------|--------------------------|
| `e5-base-v2` / `e5-large-v2` | General | 512 tokens | ✅ | ❌ | Too small context; no domain adaptation; ruled out |
| `PubMedBERT` / `BiomedBERT` | Biomedical only | 512 tokens | ✅ | ❌ | Excellent for bio terms, blind to legal structure; single-domain; ruled out |
| `BGE-M3` (BAAI) | General multilingual | 8,192 tokens | ✅ | ✅ (SPLADE) | Strong baseline; dense + sparse in one model; fine-tunable; **selected as primary local model** |
| `Kanon 2 Embedder` (Isaacus) | Legal (regulations) | 16,384 tokens | Via API/AWS | ❌ | #1 on MLEB NDCG@10=86.03 on regulatory tasks; 9% better than OpenAI te3-large; **selected as production upgrade model** |
| `Qwen3-Embedding-8B` | General (MTEB #1 open) | 8,192 tokens | ✅ (8B params) | ❌ | Best general open model; too large for CPU laptop; GPU required |

### Decision: BGE-M3 Single-Model Strategy *(implemented)*

The Kanon 2 Embedder and dual-model ensemble were not implemented. **BGE-M3 is the sole embedding model**, providing both dense and sparse vectors in a single call. This covers both regulatory structure and biomedical terminology adequately for the current system.

#### Embedding: BGE-M3 (BAAI/bge-m3)
- Runs fully locally (CPU or GPU) via `FlagEmbedding`; ~570 MB model (fp16)
- Single `encode()` call returns dense vector (1024-dim) + sparse lexical weights
- `batch_size=64`, `use_fp16=True`
- Context window of 8,192 tokens — handles full CFR sections
- MIT license

#### Why Kanon 2 Was Dropped
The Isaacus API added an external dependency and API cost. BGE-M3 + cross-encoder reranking (`bge-reranker-v2-m3`) provides sufficient retrieval quality for the current scope without the added complexity.

#### Why HyDE Was Dropped
HyDE (hypothetical document embedding) requires an LLM call per sub-question before retrieval, adding latency and complexity. The multi-agent Planner node achieves equivalent query decomposition directly.

### Retrieval Strategy: Dual-Vector Fusion *(implemented)*

```
User query
    │
    ├── BGE-M3 dense vector    → Qdrant "FDAComplianceAI" collection (dense) → top-60
    └── BGE-M3 sparse vector   → Qdrant "FDAComplianceAI" collection (sparse) → top-60

    Both lists → RRF fusion (k=60) → top-20 merged candidates
    → bge-reranker-v2-m3 cross-encoder → final top-10
    → Qdrant filtered search for cross-reference expansion (no Neo4j)
```

#### Fine-Tuning Plan (future)
Collect 500–1000 query-passage pairs from the CFR corpus using:
- Real compliance questions + their authoritative section answers
- Negative pairs (plausible but incorrect section matches)
Fine-tune BGE-M3 using `sentence-transformers` with MultipleNegativesRankingLoss. Expected gain: 15–25% improvement on domain-specific recall.

---

## 6. Layer 1 — Data Source

### Primary Source: eCFR API

**Base URL**: `https://www.ecfr.gov/api/versioner/v1/`  
**Authentication**: None required  
**Format**: XML (full text) and JSON (structure)  

### Endpoints Used

```bash
# Get hierarchical structure of Title 21 (as JSON tree)
GET https://www.ecfr.gov/api/versioner/v1/structure/current/title-21.json

# Get full text of Title 21 (XML)
GET https://www.ecfr.gov/api/versioner/v1/full/current/title-21.xml

# Get just Part 101 (scoped download — preferred for this project)
GET https://www.ecfr.gov/api/versioner/v1/full/current/title-21.xml?part=101
```

### Scope

| Component | Coverage |
|-----------|----------|
| **Primary** | 21 CFR Part 101 — Food Labeling (all subparts A–J, ~300 sections) |
| **Secondary** | 21 CFR Part 102 — Common/Usual Name for Non-Standardized Foods |
| **Reference** | 21 CFR Part 104 — Nutritional Quality Guidelines |

---

## 7. Layer 2 — Ingestion Pipeline

### 7.1 XML Parsing *(implemented)*

The actual CFR XML files (10 files, Title 21 vols 1–9) use the `CFRGRANULE` structure, **not** the `DIV`-based structure in the original plan. The parser is `backend/ingestion/extractor.py`, implemented using the stdlib `xml.etree.ElementTree`.

**Two XML structural variants handled:**
- **chapI files (vols 1–8)**: `CFRGRANULE > CHAPTER > SUBCHAP > PART > SUBPART > SECTION`
- **chapII/III files (vol 9)**: `CFRGRANULE > CHAPTER > PART > SUBPART > SECTION`

Both share: `SECTION > SECTNO + SUBJECT + P + GPOTABLE + CITA`

**Tags skipped** (non-regulatory): `TOC, TOCHD, CONTENTS, CHAPTI, PTHD, PGHD, SECHD, EAR, PRTPAGE, LRH, RRH, FAIDS, ALPHLIST, EXPLA, IPAR, STUB, SIDEHED, SIG, NAME, POSITION, EDNOTE`

### 7.2 Contextual Chunking *(implemented — paragraph-level strategy)*

Chunking is paragraph-level, not section-level. Each `SECTION` is decomposed into one or more typed chunks by `extractor.py`:

**Chunk types produced per section:**
- `section` — SECTION has one `<P>` element (no letter labels) → single chunk for the whole section
- `paragraph` — one chunk per lettered paragraph `(a)`, `(b)`, `(c)` … in multi-paragraph sections
- `definition` — one chunk per `<E>`-tagged defined term in Definitions sections (e.g., §101.2)

**Paragraph label hierarchy (4-level CFR structure):**
| Level | Type | Examples |
|-------|------|---------|
| 1 | `letter` | (a) (b) (c) |
| 2 | `number` | (1) (2) (3) |
| 3 | `roman` | (i) (ii) (iii) |
| 4 | `cap_letter` | (A) (B) (C) |

**Overflow splitting**: Paragraphs exceeding 400 tokens are split at sentence boundaries with a ~2-sentence overlap tail. Overflow chunks are linked via `overflow_sequence.next_chunk_id` and flagged with `is_overflow_chunk: true`.

**GPOTABLE handling**: Tables are serialized as markdown and embedded in the containing chunk.

**Embed text construction**: At embedding time, `embedder.py` optionally prepends the `section_preamble` (governing intro sentence) to each paragraph chunk so it is never missing context — this replaces the `[CONTEXT]` prefix approach from the original plan.

**Verified output** (smoke tested): 59,105 total chunks from 10 XML files.

### 7.3 Metadata Schema per Chunk *(actual implemented schema)*

```python
{
    # Identity
    "chunk_id":     str,   # e.g. "21-I-SCA-1-A-1-A-1-para-a"
    "cfr_citation": str,   # e.g. "21 CFR § 101.62(a)"
    "chunk_type":   str,   # "section" | "paragraph" | "definition"

    # Hierarchy
    "hierarchy": {
        "title":       str,          # "21"
        "chapter":     str,          # "I"
        "subchapter":  str | None,   # "A" (chapI files only)
        "part":        str,          # "101"
        "subpart":     str | None,   # "B"
        "section":     str,          # "101.62"
        "paragraph":   str | None,   # "a"  (top letter label)
        "subparagraph": str | None,  # deepest sub-label found
    },

    # Flat index fields (for Qdrant payload filters)
    "part_number":    str,
    "chapter_number": str,
    "section_number": str,
    "source_file":    str,          # XML filename
    "is_overflow_chunk": bool,

    # Text
    "text":             str,        # raw chunk text
    "section_preamble": str | None, # governing intro sentence of the section
    "defines":          str | None, # defined term (definition chunks only)

    # Structure
    "paragraph_labels": list[dict], # [{label, level, level_type, topic}]
    "overflow_sequence": {
        "chunk_index": int,
        "next_chunk_id": str | None,
        "prev_chunk_id": str | None,
    } | None,

    # Extracted metadata
    "cross_references_internal": list[str],  # § X.YZ refs found in text
    "metrics": list[dict],                   # [{value, unit, context}]
}
```

### 7.4 Single-Model Embedding *(implemented — BGE-M3 only)*

Kanon 2 was not implemented. BGE-M3 produces both dense and sparse vectors in a single call.

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# embed_text optionally prepends section_preamble
output = model.encode(
    [embed_text],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
    batch_size=64,
)

dense_vec  = output['dense_vecs'][0]       # list[float], 1024-dim
sparse_raw = output['lexical_weights'][0]  # dict {token_id: weight}

# Convert to Qdrant SparseVector format
sparse_vec = SparseVector(
    indices=list(sparse_raw.keys()),
    values=list(sparse_raw.values()),
)
```

Both vectors are stored as named vectors in a **single Qdrant collection** (`cfr_chunks`). `batch_size=64` with fp16 for memory efficiency.

### 7.5 Graph Extraction *(optional — disabled by default)*

`backend/ingestion/graph_builder.py` exists and can write CFR structure to Neo4j. It is **not run by default** (`run_graph: bool = False` in `PipelineConfig` and the `POST /api/ingest` request schema).

Neo4j is commented out of `requirements.txt`. Cross-reference resolution in the multi-agent layer uses Qdrant filtered search instead of graph traversal (see Section 9 and Section 10).

The schema design (Section, Subpart, DefinedTerm nodes; REFERENCES, DEFINED_IN, EXEMPTED_BY, CONTAINED_IN edges) remains valid if Neo4j is ever activated.

---

## 8. Layer 3 — Hybrid Storage (Qdrant only; Neo4j optional)

### 8.1 Qdrant Collection *(implemented — single collection with named vectors)*

One collection (`cfr_chunks`) stores both vector types as named vectors:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams, SparseIndexParams
)

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

client.create_collection(
    collection_name="FDAComplianceAI",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(
        index=SparseIndexParams(on_disk=False)
    )},
)
```

**Payload indexes created** (for filtered search):
`chunk_id`, `part_number`, `chapter_number`, `chunk_type`, `cfr_citation` (text index), `section_number`, `defines`, `is_overflow_chunk`, `source_file`

**Each Qdrant point**:
```python
PointStruct(
    id=sha256_uuid(chunk["chunk_id"]),
    vector={
        "dense": dense_vec,           # list[float], 1024-dim
        "sparse": SparseVector(...)   # lexical weights
    },
    payload={
        # flattened chunk metadata — all filter fields at top level
        "chunk_id": "...",
        "cfr_citation": "21 CFR § 101.62(a)",
        "chunk_type": "paragraph",
        "part_number": "101",
        "section_number": "101.62",
        "text": "...",
        "section_preamble": "...",
        "defines": None,
        "is_overflow_chunk": False,
        "cross_references_internal": ["101.13", "101.14"],
        # + full hierarchy, paragraph_labels, metrics, overflow_sequence
    }
)
```

**Metadata filters available at search time** — filter by `part_number`, `chapter_number`, `section_number`, `chunk_type`, `subpart_letter`, `source_file`.

### 8.2 Neo4j Schema *(optional — not in active use)*

`backend/ingestion/graph_builder.py` implements Neo4j ingestion. Neo4j is commented out of `requirements.txt` and `run_graph` defaults to `False`. The schema below is correct for future activation:

```cypher
// Node types: Title, Chapter, Subchapter, Part, Subpart, Section, Chunk, DefinedTerm, ExternalCitation
// Edge types: CONTAINS (hierarchy), CROSS_REFERENCES, DEFINES_TERM, USES_TERM, CITES, OVERFLOW_CONTINUES

MERGE (s:Section {id: "101.62"})
SET s.title = "Nutrient content claims for fat...", s.part = "101", s.subpart = "B"

MATCH (a:Section {id:"101.62"}), (b:Section {id:"101.13"})
MERGE (a)-[:CROSS_REFERENCES]->(b)
```

**Current replacement**: Cross-reference expansion and definition lookup are performed using Qdrant payload filters (`section_number` and `chunk_type="definition"` filters respectively) in the multi-agent layer.

---

## 9. Layer 4 — Retrieval Engine *(implemented)*

The retrieval engine (`backend/retrieval/retriever.py`) runs in 4 steps. HyDE and the Kanon 2 collection were not implemented.

### Step 1 — Query Encoding

```python
output = bge_model.encode([query], return_dense=True, return_sparse=True)
dense_vec  = output['dense_vecs'][0]       # 1024-dim float list
sparse_vec = SparseVector(
    indices=list(output['lexical_weights'][0].keys()),
    values=list(output['lexical_weights'][0].values()),
)
```

Optional `SearchFilters` (part_number, chapter_number, subpart_letter, section_number, chunk_type, source_file) are converted to a Qdrant `Filter` object.

### Step 2 — Qdrant Hybrid Search (single collection)

Both named vectors are searched against the same `FDAComplianceAI` collection:

```python
dense_hits  = client.query_points(collection_name, using="dense",  query=dense_vec,  limit=60, query_filter=filter)
sparse_hits = client.query_points(collection_name, using="sparse", query=sparse_vec, limit=60, query_filter=filter)
```

### Step 3 — RRF Rank Fusion + Cross-Encoder Reranking

**RRF** (k=60) merges the two ranked lists → top-20 candidates:

```python
scores[chunk_id] += 1.0 / (60 + rank + 1)  # summed over both lists
```

**Cross-encoder reranking** with `bge-reranker-v2-m3` (normalize=True) → final top-10:

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
scores = reranker.compute_score([(query, chunk_text) for ...], normalize=True)
```

### Step 4 — Overflow Expansion

For chunks with `overflow_sequence.next_chunk_id`, the retriever follows the chain via Qdrant scroll (max 5 hops) and attaches the continuation text in `overflow_chunks` on each `SearchResult`.

**Cross-reference expansion** (replacing Neo4j): The retriever node in the multi-agent layer collects `cross_references_internal` from retrieved chunks and performs targeted Qdrant searches filtered by `section_number` — no graph database required (see Section 10).

---

## 10. Layer 5 — Multi-Agent Reasoning Layer *(implemented)*

All agents run inside a **LangGraph** state machine (`backend/agents/`). The state object flows through nodes and conditional edges.

### LangGraph State *(actual `ComplianceState` TypedDict)*

```python
class ComplianceState(TypedDict, total=False):
    # Input
    query: str
    intent: str   # "compliance_question" | "definition_lookup" | "comparison" | "general"

    # Planner
    sub_questions: list[str]
    search_filters: dict   # {part_number, section_number, chapter_number} or {}

    # Retrieval
    retrieved_chunks: list[dict]
    cross_ref_chunks: list[dict]

    # Definition resolver
    definitions_resolved: dict[str, str]   # term → definition text
    definition_chunks: list[dict]

    # Synthesizer
    draft_answer: str
    citations: list[dict]          # [{section, title, text_snippet}]
    confidence_score: float

    # Verifier
    verification_passed: bool
    verification_issues: list[dict]   # [{claim, issue, detail}]
    retry_count: int

    # Conflict detector
    conflicts_detected: bool
    conflict_flags: list[dict]   # [{sections, description}]

    # Final
    final_response: dict
    error: str | None
```

`total=False` — each node returns only the keys it updates.

### LangGraph Graph Wiring *(backend/agents/graph.py)*

```
planner → retriever → definition_resolver → synthesizer → verifier
                                                              │
                           ┌──────── retry (max 2) ──────────┘
                           ▼
                        planner
                           │ (passes or max retries)
                           ▼
                    conflict_detector → END
```

Conditional edge `should_retry`: if `verification_passed=False` and `retry_count < 2`, routes back to planner; otherwise proceeds to conflict_detector.

### LLM Wrapper *(backend/agents/llm.py)*

LiteLLM with model fallback chain:
- `PRIMARY_MODEL` (env) → default `"ollama/llama3.1:8b"`
- `FALLBACK_MODEL_1` (env) → default `"groq/llama-3.1-70b-versatile"`
- `FALLBACK_MODEL_2` (env) → default `"openai/gpt-4o"`
- `OLLAMA_API_BASE` (env) → default `"http://localhost:11434"`

Two functions: `llm_completion(messages, max_tokens, temperature)` and `llm_completion_json(...)` (sets `response_format={"type": "json_object"}`). Every `json.loads()` on LLM output is wrapped in try/except with one retry on parse failure.

### Agent Roles *(all implemented)*

#### Planner Agent (`planner.py`)
- Input: `query`, optionally `verification_issues` (on retry)
- Output: `intent`, `sub_questions` (1–4 queries), `search_filters`
- On retry: appends extra context instructing LLM to find source text for unsupported claims

#### Retriever Node (`retriever_node.py`) — no LLM
- Input: `query`, `sub_questions`, `search_filters`
- Output: `retrieved_chunks`, `cross_ref_chunks`
- Calls `CFRRetriever.search()` for each sub-question (top_k=10, reranker=True), deduplicates by `chunk_id`
- **Cross-reference expansion** (replaces Neo4j): collects unique `cross_references_internal` section numbers from retrieved chunks → Qdrant search filtered by `section_number` → stored in `cross_ref_chunks`

#### Definition Resolver (`definition_resolver.py`)
- Input: `query`, `retrieved_chunks`
- Output: `definitions_resolved`, `definition_chunks`
- Phase 1 (LLM): identify regulated terms needing formal definitions → JSON `{"terms": [...]}`
- Phase 2 (Qdrant): search with `chunk_type="definition"` filter and query `"definition of {term}"` → match on `defines` payload field

#### Synthesizer (`synthesizer.py`)
- Input: `query`, `retrieved_chunks`, `cross_ref_chunks`, `definitions_resolved`
- Output: `draft_answer`, `citations`, `confidence_score`
- Builds context: definitions block + top-8 primary chunks + top-4 cross-ref chunks
- LLM returns JSON with inline `[21 CFR X.Y]` citations, citations list, confidence score (0–1); `max_tokens=2048`

#### Verifier (`verifier.py`)
- Input: `draft_answer`, `retrieved_chunks`, `cross_ref_chunks`, `retry_count`
- Output: `verification_passed`, `verification_issues`, `retry_count` (incremented on issues)
- LLM checks every factual claim against source chunks in strict mode (numbers, thresholds, section refs must appear verbatim)

#### Conflict Detector (`conflict_detector.py`)
- Input: full state
- Output: `conflicts_detected`, `conflict_flags`, `final_response`
- LLM checks for contradictions (general rule vs. exception, conditional applicability)
- Also assembles the final `final_response` dict returned by `POST /api/query`:
  ```json
  {
    "answer": "...",
    "citations": [...],
    "confidence_score": 0.87,
    "conflicts_detected": false,
    "conflict_details": [],
    "disclaimer": "This is for informational purposes only...",
    "retrieved_sections": ["101.62", "101.13"],
    "verification_passed": true
  }
  ```

---

## 11. Flow A — Query Flow

**Trigger**: User submits a natural language compliance question via API or UI.

```
POST /api/query
{
  "question": "Can I label my granola bar as 'excellent source of fiber'?"
}
```

**Pipeline**:
```
1. Planner agent decomposes question into sub-questions
2. For each sub-question:
   a. HyDE generates hypothetical regulatory passage
   b. Embed with BGE-M3 (dense + sparse) and Kanon 2 (dense)
   c. Qdrant hybrid search → top-20 per model
   d. RRF fusion → top-30 merged
   e. Cross-encoder rerank → top-10
   f. Neo4j graph expand → pull referenced sections
3. Definition resolver resolves any regulated terms
4. Synthesizer assembles draft answer with inline citations
5. Verifier cross-checks all claims
6. Conflict detector checks for exceptions/ambiguity
7. Return final response
```

**Response Schema**:
```json
{
  "answer": "To label a product as 'excellent source of fiber', the product must contain at least 20% of the Daily Value (DV) for dietary fiber per reference amount customarily consumed (RACC)...",
  "citations": [
    {"section": "101.54(b)", "title": "Nutrient content claims for fiber", "url": "https://..."},
    {"section": "101.9(c)(6)", "title": "Fiber DV reference", "url": "https://..."}
  ],
  "confidence_score": 0.94,
  "conflicts_detected": false,
  "disclaimer": "This is for informational purposes only and does not constitute legal advice.",
  "retrieved_sections": ["101.54", "101.9", "101.13"]
}
```

---

## 12. Flow B — Document Violation Analysis Flow

**Trigger**: User uploads a food label document (PDF, DOCX, PNG/JPG).

```
POST /api/analyze-document
Content-Type: multipart/form-data
file: [food_label.pdf]
```

This flow is processed asynchronously via Celery + Redis (returns a `job_id` immediately, result fetched via polling or webhook).

### Step 1 — Document Parser Agent

Accepts: PDF, DOCX, PNG, JPG  
Extracts text using:
- PDF: `PyMuPDF` (fitz) — preserves layout structure
- DOCX: `python-docx`
- Image (label photo): GPT-4o vision or Tesseract OCR

```python
def parse_document(file: UploadFile) -> ParsedDocument:
    if file.content_type == "application/pdf":
        return parse_pdf(file)
    elif file.content_type in ["image/png", "image/jpeg"]:
        return parse_image_via_vision(file)  # sends to GPT-4o
    elif file.content_type == "application/vnd.openxmlformats...":
        return parse_docx(file)
```

### Step 2 — Claim Extractor Agent

An LLM with structured output (Pydantic) reads the raw document text and extracts every distinct claim or statement as a structured object:

```python
EXTRACTION_PROMPT = """
You are a food labeling regulatory expert. Given the text of a food label or 
product specification document, extract every distinct claim, statement, or 
piece of information that would appear on or in the label.

For each item, return:
- claim_text: The exact text from the document
- claim_type: One of ["nutrient_content_claim", "health_claim", "ingredient_statement",
  "nutrition_facts_panel", "allergen_statement", "net_weight", "manufacturer_info",
  "serving_size", "front_of_pack_claim", "other"]
- location_hint: Where on the label this appears (front panel, back, side, etc.)
"""

class ExtractedClaim(BaseModel):
    claim_text: str
    claim_type: str
    location_hint: str
```

### Step 3 — Regulation Mapper Agent

For each extracted claim, run a hybrid search against the CFR corpus to find the most relevant regulatory section:

```python
def map_claim_to_regulation(claim: ExtractedClaim) -> ClaimMapping:
    # Embed the claim text
    results = hybrid_search(claim.claim_text, top_k=10)
    reranked = reranker.rerank(claim.claim_text, results)
    best_match = reranked[0]
    
    # Graph expand to get related sections
    related = neo4j.get_related_sections(best_match.section_id)
    
    return ClaimMapping(
        claim=claim,
        primary_regulation=best_match,
        related_regulations=related,
        mapping_confidence=best_match.score
    )
```

### Step 4 — Gap Analyzer Agent (Reverse Pass)

This is the **most important step for violation detection**. Instead of only checking each claim, we iterate over **every section in the CFR corpus** and check whether the uploaded document satisfies it.

```python
def gap_analysis(doc_text: str, all_cfr_sections: list[dict]) -> list[GapResult]:
    results = []
    for section in all_cfr_sections:
        # Search the uploaded document for content matching this regulation
        doc_chunks = chunk_text(doc_text)  # temp chunking of uploaded doc
        matches = search_within_doc(section['text'], doc_chunks)
        best_match = reranker.rerank(section['text'], matches)[0]
        
        if best_match.score >= 0.85:
            coverage = "full"
        elif best_match.score >= 0.55:
            coverage = "partial"
        else:
            coverage = "missing"
        
        results.append(GapResult(
            section_id=section['section_id'],
            coverage=coverage,
            best_match_text=best_match.text,
            match_score=best_match.score
        ))
    
    return results
```

### Step 5 — Violation Classifier Agent

For every `partial` or `missing` gap result, an LLM performs a verification pass to eliminate false positives and classify severity:

```python
VERIFICATION_PROMPT = """
You are an FDA food labeling compliance expert.

Regulation requirement:
<regulation>
{regulation_text}
</regulation>

Company label content:
<label_content>
{matched_label_text}
</label_content>

Based on 21 CFR Part 101, determine:
1. Does the label content satisfy this regulation? (yes / partial / no)
2. If partial or no: what specific element is missing or incorrect?
3. Severity: critical (recall risk) | high (warning letter risk) | medium (technical violation) | low (best practice)
4. Suggested correction

Return as JSON only.
"""
```

### Step 6 — Structured Violation Report

```python
class Violation(BaseModel):
    violation_id: str
    regulation_section: str           # "101.62(b)(1)"
    regulation_title: str             # "Low fat definition"
    violation_type: str               # "missing" | "incorrect" | "prohibited"
    severity: str                     # "critical" | "high" | "medium" | "low"
    offending_text: str               # Exact quote from uploaded document
    requirement_summary: str          # What the regulation requires
    suggested_correction: str         # How to fix it
    cfr_url: str                      # Link to the regulation
    confidence: float                 # 0-1

class ViolationReport(BaseModel):
    document_name: str
    analyzed_at: datetime
    overall_compliance_score: float   # 0-100%
    total_requirements_checked: int
    fully_compliant: int
    partial_compliance: int
    violations: list[Violation]
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
```

**Example violation output**:
```json
{
  "violation_id": "v-003",
  "regulation_section": "101.62(b)(1)",
  "regulation_title": "Nutrient content claims — Low fat",
  "violation_type": "incorrect",
  "severity": "high",
  "offending_text": "Contains only 4g of fat per serving",
  "requirement_summary": "A 'low fat' claim requires ≤3g total fat per RACC and per labeled serving. 4g exceeds this threshold.",
  "suggested_correction": "Either reformulate to ≤3g fat per serving, or remove the 'low fat' claim. Consider 'reduced fat' if 25%+ less fat than reference food.",
  "cfr_url": "https://www.ecfr.gov/current/title-21/chapter-I/part-101/section-101.62",
  "confidence": 0.97
}
```

---

## 13. Layer 6 — Response Synthesis

### LiteLLM Gateway *(implemented)*

All agent LLM calls route through **LiteLLM** via `backend/agents/llm.py`:
- Model fallback chain: `Ollama (llama3.1:8b)` → `Groq (llama-3.1-70b-versatile)` → `OpenAI (gpt-4o)`
- Configured via environment variables: `PRIMARY_MODEL`, `FALLBACK_MODEL_1`, `FALLBACK_MODEL_2`, `OLLAMA_API_BASE`
- Single API interface — swap models without changing agent code

```python
import litellm

response = litellm.completion(
    model=PRIMARY_MODEL,
    messages=messages,
    fallbacks=[FALLBACK_MODEL_1, FALLBACK_MODEL_2],
    max_tokens=max_tokens,
    temperature=temperature,
)
```

### Async Processing for Document Analysis *(not yet implemented)*

Celery + Redis async task queue for `POST /api/analyze-document` is **not implemented**. The query endpoint (`POST /api/query`) runs synchronously via the LangGraph pipeline. Document violation analysis (Flow B) is a future phase.

---

## 14. Observability & Evaluation *(not yet implemented)*

Langfuse tracing, RAGAS evaluation suite, and the user feedback endpoint (`POST /api/feedback`) are **not yet implemented**. These are planned for Phase 5.

Target metrics when implemented:
- Faithfulness > 0.90 (answers grounded in retrieved text)
- Context Recall > 0.85 (all relevant sections retrieved)
- Answer Relevance > 0.88

Current observability: standard Python `logging` at INFO level across all modules (`logging.basicConfig` in `main.py`).

---

## 15. Technology Stack Summary

### Core Stack *(actual implemented dependencies)*

| Layer | Technology | Status | Purpose |
|-------|-----------|--------|---------|
| **Ingestion** | `xml.etree.ElementTree` (stdlib) | ✅ | XML parsing (replaced lxml) |
| **Chunking** | Custom paragraph-level chunker | ✅ | CFR-aware chunking in `extractor.py` |
| **Embedding** | `BGE-M3` (BAAI/bge-m3) via `FlagEmbedding` | ✅ | Dense + sparse, local |
| **Reranker** | `bge-reranker-v2-m3` via `FlagEmbedding` | ✅ | Cross-encoder, local |
| **Vector store** | **Qdrant** | ✅ | Single collection, named vectors |
| **Graph database** | **Neo4j** | ⬜ optional | Off by default; `graph_builder.py` exists |
| **Agent framework** | **LangGraph** | ✅ | Stateful multi-agent |
| **LLM routing** | **LiteLLM** | ✅ | Ollama → Groq → OpenAI |
| **Local LLM** | Ollama + llama3.1:8b | ✅ | Local inference |
| **API** | **FastAPI** + **uvicorn** | ✅ | Async REST API |
| **Task queue** | BackgroundTasks + in-memory job dict | ✅ | Celery/Redis dropped; sufficient for current scale |
| **Document parsing** | PyMuPDF, python-docx | ✅ | PDF + DOCX + plain text (Tesseract/image dropped) |
| **Observability** | Langfuse | ❌ not implemented | Planned for Phase 5 |
| **Evaluation** | RAGAS | ❌ not implemented | Planned for Phase 5 |

### Python Dependencies (`backend/requirements.txt`) *(actual)*

```
# Web framework
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0

# Embedding / reranking (BGE-M3 hybrid dense+sparse)
FlagEmbedding>=1.2.10
transformers<4.45.0

# Vector database
qdrant-client>=1.9.0

# Graph database (optional — not required for hybrid search pipeline)
# neo4j>=5.0.0

# Multi-agent reasoning (LangGraph + LiteLLM)
langgraph>=0.2.0
langchain-core>=0.3.0
litellm>=1.40.0

# Document parsing (Phase 4: Document Violation Analysis)
pymupdf>=1.24.0
python-docx>=1.1.0
python-multipart>=0.0.9

# Utilities
tqdm>=4.0.0
python-dotenv>=1.0.0
```

**Runtime**: Python 3.13, virtual environment at `backend/venv/`

---

## 16. Data Models & Schemas

### Qdrant Point Structure

```python
from qdrant_client.models import PointStruct, SparseVector

# BGE-M3 dense point
PointStruct(
    id=chunk_hash_id,
    vector=bge_dense_vector,          # list[float], 1024 dims
    payload={
        "chunk_id": "101.62_c0",
        "section_id": "101.62",
        "section_title": "...",
        "subpart": "B",
        "part": "101",
        "text": "...",
        "prev_chunk_id": "101.61_c0",
        "next_chunk_id": "101.62_c1",
        "cross_refs": ["101.13"],
        "source_url": "https://...",
        "token_count": 487
    }
)

# BGE-M3 sparse point
PointStruct(
    id=chunk_hash_id,
    vector={"sparse": SparseVector(
        indices=[token_id_1, token_id_2, ...],
        values=[weight_1, weight_2, ...]
    )},
    payload={...}  # same payload
)
```

### Neo4j Cypher Schema

```cypher
// Constraints (run once at setup)
CREATE CONSTRAINT section_id IF NOT EXISTS
  FOR (s:Section) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT term_name IF NOT EXISTS
  FOR (d:DefinedTerm) REQUIRE d.term IS UNIQUE;

// Indexes for fast lookup
CREATE INDEX section_subpart IF NOT EXISTS
  FOR (s:Section) ON (s.subpart);

CREATE INDEX section_part IF NOT EXISTS
  FOR (s:Section) ON (s.part);
```

---

## 17. Implementation Phasing

### Phase 1 — Core RAG ✅ Complete
*Goal: working end-to-end query flow with basic retrieval*

- [x] Write CFR XML parser + paragraph-level chunker (`ingestion/extractor.py`)
- [x] Set up BGE-M3 embedding with dense + sparse named vectors in a single Qdrant collection
- [x] Index all 59,105 chunks into Qdrant `FDAComplianceAI` collection
- [x] Build FastAPI server with `/api/ingest`, `/api/search`, `/api/chunks/{id}`, `/health`
- [x] Implement single-model dense+sparse retrieval (`retrieval/retriever.py`)
- [x] Wire up LiteLLM with Ollama → Groq → OpenAI fallback chain

### Phase 2 — Hybrid Retrieval + Reranking ✅ Complete
*Goal: significantly improve retrieval quality*

- [x] BGE-M3 sparse + dense in single Qdrant collection with named vectors
- [x] RRF rank fusion (k=60) across dense and sparse result lists
- [x] `bge-reranker-v2-m3` cross-encoder reranking (top-10 final)
- [x] Overflow chunk expansion (follow `next_chunk_id` chain, max 5 hops)
- [x] Qdrant-based cross-reference expansion (replaces Neo4j graph)
- [ ] HyDE query expansion — *dropped (replaced by Planner query decomposition)*
- [ ] Kanon 2 Embedder — *dropped (BGE-M3 sufficient for current scope)*
- [ ] Neo4j graph — *optional, off by default*
- [ ] RAGAS baseline evaluation — *pending Phase 5*

### Phase 3 — Multi-Agent System ✅ Complete
*Goal: production-quality reasoning and citation*

- [x] LangGraph `StateGraph` with `ComplianceState` TypedDict
- [x] Planner agent — query decomposition + intent classification + search filter extraction
- [x] Retriever node — sub-question search + Qdrant cross-reference expansion (no LLM)
- [x] Definition Resolver agent — term identification (LLM) + Qdrant definition lookup
- [x] Synthesizer agent — answer generation with inline CFR citations (max_tokens=2048)
- [x] Verifier agent — hallucination detection + retry signal (max 2 retries)
- [x] Conflict Detector agent — cross-section conflict detection + final response assembly
- [x] `POST /api/query` endpoint wired to `query_graph.invoke()`

### Phase 4 — Document Violation Analysis Flow ✅ Complete
*Goal: full document upload and analysis pipeline*

- [x] Build Document Parser (PDF via PyMuPDF + DOCX via python-docx + plain text; Tesseract/image dropped)
- [x] Build Claim Extractor node (LLM extracts claim_text, claim_type, location_hint)
- [x] Build Regulation Mapper node (per-claim CFRRetriever hybrid search, top-5 chunks with reranking)
- [x] Build Violation Classifier node (per-claim LLM compliance check + single-pass gap analysis for missing required elements — Gap Analyzer folded in)
- [x] Build Report Builder node (assembles ViolationReport: overall_status, severity_summary, sorted violations[])
- [x] Build LangGraph `DocumentAnalysisState` + StateGraph with error short-circuit routing
- [x] Async job store via BackgroundTasks + in-memory `dict[job_id → status/result]` (Celery/Redis dropped)
- [x] `POST /api/analyze-document` (file upload → job_id) and `GET /api/jobs/{job_id}` (poll) endpoints
- [x] Frontend Next.js route handler polls job until completion and transforms ViolationReport → AnalysisResponse schema

### Phase 5 — Observability, Evaluation & Fine-Tuning ⬜ Pending
*Goal: measure quality, improve, and make the system trustworthy*

- [ ] Deploy self-hosted Langfuse (Docker)
- [ ] Add Langfuse tracing to all pipeline steps
- [ ] Build golden test set of 100 Q&A pairs
- [ ] Run full RAGAS evaluation suite
- [ ] Collect 500 query-passage pairs for BGE-M3 fine-tuning
- [ ] Fine-tune BGE-M3 with `sentence-transformers` MultipleNegativesRankingLoss
- [ ] Add `POST /api/feedback` endpoint
- [ ] Performance optimization (async retrieval, batch embedding)

---

## 18. Local Hardware Requirements

### Minimum (development, CPU-only)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| CPU | 8 cores | 12+ cores |
| Disk | 20 GB free | 50 GB free |
| GPU | None | Optional (speeds BGE-M3 ~10x) |

### Memory Breakdown at Runtime

| Component | RAM Usage |
|-----------|-----------|
| BGE-M3 (fp16) | ~1.1 GB |
| bge-reranker-v2-m3 (fp16) | ~0.6 GB |
| Qdrant (3 collections, ~5K chunks) | ~0.5 GB |
| Neo4j | ~0.3 GB |
| Redis | ~0.1 GB |
| Ollama llama3.1:8b | ~5 GB (via quantization) |
| FastAPI + Celery | ~0.3 GB |
| **Total** | **~8 GB active** |

16 GB RAM is sufficient. 32 GB allows comfortable headroom for Kanon 2 API calls and larger batch operations.

### GPU Note

BGE-M3 runs on CPU but indexing 300 sections takes ~2–5 minutes on CPU vs ~15 seconds on GPU. For development this is fine since ingestion is a one-time batch operation. Query-time embedding (single chunk) takes <1 second on CPU.

---

## 19. Directory Structure *(actual)*

```
FDAComplianceAI/
├── implementation_plan.md      # This document
├── multiagent_plan.txt         # Multi-agent layer design notes (implemented)
├── retrieval_plan.txt          # Retrieval layer design notes (implemented)
│
├── backend/
│   ├── main.py                 # FastAPI app — all endpoints
│   ├── config.py               # Settings (env vars, Qdrant URL, CORS origins)
│   ├── requirements.txt        # Python dependencies
│   ├── venv/                   # Python 3.13 virtual environment
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── extractor.py        # CFR XML → paragraph-level JSON chunks (59,105 chunks)
│   │   ├── embedder.py         # BGE-M3 dense+sparse → Qdrant "FDAComplianceAI" collection
│   │   ├── pipeline.py         # Orchestrates extract → embed (graph optional)
│   │   └── graph_builder.py    # Neo4j ingestion (optional, off by default)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py        # CFRRetriever: hybrid search + RRF + reranker + overflow
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py            # ComplianceState TypedDict
│   │   ├── llm.py              # LiteLLM wrapper (Ollama → Groq → Gemini)
│   │   ├── planner.py          # Intent classification + query decomposition
│   │   ├── retriever_node.py   # Qdrant search + cross-ref expansion (no LLM)
│   │   ├── definition_resolver.py  # Term identification + Qdrant definition lookup
│   │   ├── synthesizer.py      # Answer generation with inline CFR citations
│   │   ├── verifier.py         # Hallucination detection + retry signal
│   │   ├── conflict_detector.py    # Conflict detection + final response assembly
│   │   └── graph.py            # LangGraph StateGraph wiring + compile
│   │
│   ├── document_analysis/
│   │   ├── __init__.py
│   │   ├── state.py            # DocumentAnalysisState TypedDict
│   │   ├── parser.py           # PDF (PyMuPDF) + DOCX + plain text extraction
│   │   ├── claim_extractor.py  # LangGraph node: LLM claim extraction
│   │   ├── regulation_mapper.py # LangGraph node: per-claim CFR hybrid search
│   │   ├── violation_classifier.py # LangGraph node: compliance check + gap analysis
│   │   ├── report_builder.py   # LangGraph node: assemble ViolationReport
│   │   └── graph.py            # LangGraph StateGraph + error short-circuit routing
│   │
│   └── data/
│       ├── cfr_xml/            # 10 CFR XML files (Title 21, vols 1–9)
│       └── chunks/
│           └── cfr_chunks.json # Extractor output (generated; ~59,105 chunks)
│
└── frontend/                   # Next.js frontend (Vercel deployment)
    └── src/
        ├── app/
        │   ├── api/
        │   │   ├── query/route.ts           # Proxy → POST /api/query
        │   │   └── analyze-document/route.ts # Proxy → POST /api/analyze-document + job polling
        │   ├── chat/page.tsx               # Compliance Q&A chat UI
        │   └── analyzer/page.tsx           # Document upload + violation report UI
        ├── lib/api.ts                      # Client API helpers
        └── types/index.ts                  # Shared TypeScript interfaces
```

**Not yet created** (planned for Phase 5):
- `evaluation/` — RAGAS golden test set + evaluation runner
- `docker-compose.yml` — full local stack

---

## Appendix A — Embedding Model Decision Summary

| Model | Legal Accuracy | Biomedical Accuracy | Local? | Sparse? | Context | Decision |
|-------|---------------|--------------------|----|------|---------|----------|
| E5-base-v2 | Low | Low | ✅ | ❌ | 512 | ❌ Ruled out |
| E5-large-v2 | Low-Medium | Low-Medium | ✅ | ❌ | 512 | ❌ Ruled out |
| PubMedBERT | Low | Very High | ✅ | ❌ | 512 | ❌ Wrong domain fit |
| BiomedBERT | Low | Very High | ✅ | ❌ | 512 | ❌ Wrong domain fit |
| BGE-M3 | Medium-High | Medium | ✅ | ✅ | 8,192 | ✅ **Implemented — sole model** |
| Kanon 2 Embedder | Very High (91.48 NDCG@10 regulatory) | Low | API | ❌ | 16,384 | ⬜ Not implemented (future upgrade) |
| Qwen3-Embedding-8B | High | Medium | ✅ (GPU) | ❌ | 8,192 | ⚠️ Upgrade path (GPU required) |

**Why BGE-M3 alone**: The dual-model approach with Kanon 2 was deferred — it adds an external API dependency and cost. BGE-M3 + `bge-reranker-v2-m3` cross-encoder reranking provides sufficient retrieval quality for the current scope. The cross-encoder reranker at the end of the pipeline compensates significantly for single-model retrieval gaps by doing a full (query, chunk) relevance score pass.

**Kanon 2 upgrade path**: When the system needs higher regulatory retrieval precision, adding Kanon 2 vectors as a third named vector in the existing `FDAComplianceAI` collection (with RRF fusion across 3 lists) is straightforward — no schema rebuild needed.

---

## Appendix B — Key API References

- eCFR Structure API: `https://www.ecfr.gov/api/versioner/v1/structure/current/title-21.json`
- eCFR Full Text API: `https://www.ecfr.gov/api/versioner/v1/full/current/title-21.xml?part=101`
- Kanon 2 Embedder API: `https://api.isaacus.com/v1/embeddings`
- Qdrant REST API: `http://localhost:6333`
- Neo4j Bolt: `bolt://localhost:7687`
- Langfuse (self-hosted): `http://localhost:3000`
- RAGAS docs: `https://docs.ragas.io`
- LangGraph docs: `https://langchain-ai.github.io/langgraph/`

---

*End of Implementation Plan*
