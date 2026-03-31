# FDA Food Labeling Compliance RAG — Implementation Plan

> **Version**: 1.0  
> **Last Updated**: March 2026  
> **Status**: Architecture & Design Phase  

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

### Decision: Dual-Model Hybrid Strategy

The fundamental insight is that **no single model covers both legal regulatory structure AND biomedical chemical terminology equally well**. The solution is a **dual-model ensemble with late fusion**:

#### Primary Dense Embedding: BGE-M3
- Runs fully locally (CPU or GPU); ~570MB model size
- Simultaneously produces: dense vector (1024-dim), sparse SPLADE vector, and ColBERT multi-vector
- Qdrant natively stores and searches all three in a single collection
- Context window of 8,192 tokens — handles full CFR sections without truncation
- MIT license; fine-tunable on domain data
- Handles the semantic structure of regulatory language

#### Secondary Dense Embedding: Kanon 2 Embedder (via Isaacus API)
- NDCG@10 = 91.48 specifically on **regulatory** retrieval tasks (highest in class)
- Trained on millions of laws, regulations, cases from 38 jurisdictions
- 16,384 token context window
- Available via Isaacus API (with free tier for development)
- Handles legal regulatory language with highest known accuracy

#### Why Not PubMedBERT?
PubMedBERT achieves 95.62% Pearson correlation on biomedical benchmarks and excels at clinical/pharma terminology. However, it has a 512-token context window (too small for full CFR sections), produces no sparse vectors, and is entirely blind to regulatory legal structure. The biomedical terminology in 21 CFR Part 101 (fatty acids, vitamins, minerals, nutrient thresholds) is sufficiently represented by BGE-M3 after fine-tuning on the CFR corpus — PubMedBERT's advantage is in clinical notes and PubMed abstracts, not regulatory text.

#### Fine-Tuning Plan (after initial deployment)
Collect 500–1000 query-passage pairs from the CFR corpus using:
- Real compliance questions + their authoritative section answers
- Negative pairs (plausible but incorrect section matches)
Fine-tune BGE-M3 using `sentence-transformers` with MultipleNegativesRankingLoss. Expected gain: 15–25% improvement on domain-specific recall.

### Retrieval Strategy: Triple-Vector Fusion

At query time, each query is embedded with both models and searched in parallel:

```
User query
    │
    ├── BGE-M3 dense vector    → Qdrant dense collection → top-20 results
    ├── BGE-M3 SPLADE sparse   → Qdrant sparse collection → top-20 results  
    └── Kanon 2 dense vector   → Qdrant kanon collection → top-20 results

    All three lists → RRF rank fusion → top-30 merged candidates
    → bge-reranker-v2-m3 cross-encoder → final top-10
    → Neo4j 1-hop graph expansion → final context set
```

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

### 7.1 XML Parsing

The eCFR XML uses a hierarchical `<DIV>` structure:

```
DIV1 TYPE="TITLE"        → Title 21
  DIV3 TYPE="CHAPTER"    → Chapter I (FDA)
    DIV5 TYPE="PART"     → Part 101
      DIV6 TYPE="SUBPART"→ Subpart A, B, C...
        SECTION          → §101.1, §101.2, §101.3...
          P              → Paragraph text
```

The parser walks this tree using `lxml`, extracts each `SECTION` as a document, and records the full ancestry path as metadata.

**Implementation** (`ingestion/xml_parser.py`):

```python
from lxml import etree

def parse_cfr_xml(xml_path: str) -> list[dict]:
    tree = etree.parse(xml_path)
    sections = []
    
    for section in tree.findall('.//SECTION'):
        section_num = section.findtext('SECTNO', '').strip()
        subject = section.findtext('SUBJECT', '').strip()
        
        # Collect all paragraph text
        paragraphs = []
        for p in section.findall('.//P'):
            text = ''.join(p.itertext()).strip()
            if text:
                paragraphs.append(text)
        
        full_text = '\n\n'.join(paragraphs)
        
        # Extract ancestry metadata
        ancestors = get_ancestors(section)  # walk .getparent() chain
        
        # Extract cross-references (§ patterns)
        cross_refs = extract_cross_refs(full_text)
        
        sections.append({
            'section_id': section_num,          # e.g. "101.62"
            'title': subject,                   # e.g. "Nutrient content claims..."
            'text': full_text,
            'part': ancestors.get('part'),      # "101"
            'subpart': ancestors.get('subpart'),# "B"
            'chapter': 'I',
            'cfr_title': '21',
            'cross_refs': cross_refs,           # ["101.13", "101.14"]
            'char_count': len(full_text),
            'token_estimate': len(full_text) // 4
        })
    
    return sections

def extract_cross_refs(text: str) -> list[str]:
    import re
    # Match patterns like §101.62, § 101.13(b), §§101.13 and 101.14
    pattern = r'§+\s*(\d+\.\d+)'
    return list(set(re.findall(pattern, text)))
```

### 7.2 Contextual Chunking

Each section is chunked with a **contextual prefix** prepended before embedding. This technique (pioneered by Anthropic's research) dramatically improves retrieval by ensuring no chunk loses its regulatory context.

**Contextual prefix format**:
```
[CONTEXT] This passage is from 21 CFR Title 21, Chapter I (FDA), Part 101 
(Food Labeling), Subpart B (Specific Food Labeling Requirements), Section 101.62 
(Nutrient content claims for fat, fatty acid, and cholesterol content). 
Part 101 governs the labeling of human food products including nutrition 
facts, health claims, and nutrient content claims.
[/CONTEXT]

{chunk_text}
```

**Chunking rules**:
- Minimum chunk: 100 tokens → merge with adjacent section
- Target chunk: 300–512 tokens  
- Maximum chunk: 1,024 tokens → split at sentence boundary with 20% overlap
- Sections under 1,024 tokens are kept whole (most CFR sections are)
- For long sections, child chunks retain the parent section's header as a prefix

**prev/next linking**: Each chunk stores `prev_chunk_id` and `next_chunk_id` so the retriever can expand context windows at query time.

### 7.3 Metadata Schema per Chunk

```python
class CFRChunk:
    chunk_id: str          # "101.62_c0" (section + chunk index)
    section_id: str        # "101.62"
    section_title: str     # "Nutrient content claims for fat..."
    subpart: str           # "B"
    part: str              # "101"
    cfr_title: str         # "21"
    text: str              # raw chunk text (for storage in Qdrant payload)
    context_text: str      # contextual prefix + chunk text (for embedding)
    token_count: int
    prev_chunk_id: str | None
    next_chunk_id: str | None
    cross_refs: list[str]  # ["101.13", "101.14"]
    source_url: str        # eCFR URL for this section
    indexed_at: datetime
```

### 7.4 Dual-Model Embedding

```python
from FlagEmbedding import BGEM3FlagModel
import isaacus  # Kanon 2 API client

bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def embed_chunk(chunk: CFRChunk) -> EmbeddedChunk:
    text = chunk.context_text  # prefixed text
    
    # BGE-M3: dense + sparse in one call
    bge_output = bge_model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False  # optional, adds ColBERT multi-vec
    )
    
    # Kanon 2: dense only, regulatory-optimized
    kanon_output = isaacus_client.embed(
        model="kanon-2-embedder",
        texts=[text],
        task="retrieval/document"
    )
    
    return EmbeddedChunk(
        chunk=chunk,
        bge_dense=bge_output['dense_vecs'][0],       # 1024-dim
        bge_sparse=bge_output['lexical_weights'][0],  # sparse dict {token_id: weight}
        kanon_dense=kanon_output.embeddings[0]        # 1792-dim
    )
```

### 7.5 Graph Extraction

After parsing, a separate graph builder processes every section's `cross_refs` list and writes edges to Neo4j:

```python
def build_graph(sections: list[dict], neo4j_driver):
    with neo4j_driver.session() as session:
        # Create section nodes
        for s in sections:
            session.run("""
                MERGE (sec:Section {id: $id})
                SET sec.title = $title,
                    sec.subpart = $subpart,
                    sec.part = $part,
                    sec.url = $url
            """, **s)
        
        # Create cross-reference edges
        for s in sections:
            for ref in s['cross_refs']:
                session.run("""
                    MATCH (a:Section {id: $from_id})
                    MATCH (b:Section {id: $to_id})
                    MERGE (a)-[:REFERENCES]->(b)
                """, from_id=s['section_id'], to_id=ref)
```

**Neo4j node types**:
- `Section` — each CFR section
- `Subpart` — container grouping sections
- `DefinedTerm` — legal definitions (extracted from §101.2, §101.9, etc.)

**Neo4j edge types**:
- `REFERENCES` — explicit §-reference in text
- `DEFINED_IN` — term defined in a section
- `EXEMPTED_BY` — exception relationship
- `CONTAINED_IN` — section → subpart hierarchy

---

## 8. Layer 3 — Hybrid Storage (Qdrant + Neo4j)

### 8.1 Qdrant Collections

Three named collections store the three vector types:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

client = QdrantClient("localhost", port=6333)

# BGE-M3 dense collection
client.create_collection("cfr101_bge_dense", vectors_config=VectorParams(
    size=1024, distance=Distance.COSINE
))

# BGE-M3 sparse collection (SPLADE)
client.create_collection("cfr101_bge_sparse", sparse_vectors_config={
    "sparse": SparseVectorParams()
})

# Kanon 2 dense collection
client.create_collection("cfr101_kanon_dense", vectors_config=VectorParams(
    size=1792, distance=Distance.COSINE
))
```

**Payload stored in each point** (all three collections share same payload):
```json
{
  "chunk_id": "101.62_c0",
  "section_id": "101.62",
  "section_title": "Nutrient content claims for fat, fatty acid, and cholesterol content",
  "subpart": "B",
  "part": "101",
  "text": "...(raw chunk text)...",
  "prev_chunk_id": "101.61_c0",
  "next_chunk_id": "101.62_c1",
  "cross_refs": ["101.13", "101.14"],
  "source_url": "https://www.ecfr.gov/current/title-21/chapter-I/part-101/section-101.62",
  "token_count": 487
}
```

**Metadata filters available at search time** — filter by `subpart`, `part`, `section_id` before vector search to scope queries.

### 8.2 Neo4j Schema

```cypher
// Node: Section
CREATE (s:Section {
  id: "101.62",
  title: "Nutrient content claims for fat...",
  subpart: "B",
  part: "101",
  url: "https://..."
})

// Node: DefinedTerm
CREATE (d:DefinedTerm {
  term: "nutrient content claim",
  defined_in_section: "101.13"
})

// Edge: cross-reference
MATCH (a:Section {id:"101.62"}), (b:Section {id:"101.13"})
CREATE (a)-[:REFERENCES {context: "as defined in"}]->(b)

// Edge: definition
MATCH (s:Section {id:"101.13"}), (d:DefinedTerm {term:"nutrient content claim"})
CREATE (s)-[:DEFINES]->(d)
```

---

## 9. Layer 4 — Retrieval Engine

The retrieval engine runs in 5 sequential steps for every query:

### Step 1 — Query Decomposition

For complex queries, an LLM breaks the question into atomic sub-questions that can each be answered by a single CFR section:

```
User: "Can my product be labeled 'low sodium' and include a heart disease health claim?"

Sub-questions:
  1. What is the regulatory definition of 'low sodium'?
  2. What are the threshold requirements to qualify for a low sodium nutrient content claim?
  3. What are the approved health claims related to heart disease and cardiovascular disease?
  4. What are the general conditions for using a health claim on a food label?
  5. Are there any restrictions on combining nutrient content claims and health claims?
```

Simple queries (single-section answers) skip decomposition.

### Step 2 — HyDE (Hypothetical Document Embedding)

For each sub-question, instead of embedding the question directly, an LLM generates a **hypothetical regulatory passage** that would answer the question. This hypothetical text is then embedded — its embedding lives in the same semantic space as real CFR text, improving recall.

```python
HYDE_PROMPT = """
You are a food labeling regulatory expert. Write a concise passage 
(2-4 sentences) from the Code of Federal Regulations that would 
directly answer the following question. Write it in regulatory language.
Respond with only the passage, no preamble.

Question: {question}
"""

hypothetical_doc = llm.generate(HYDE_PROMPT.format(question=sub_question))
hyde_embedding = bge_model.encode([hypothetical_doc])['dense_vecs'][0]
```

### Step 3 — Qdrant Hybrid Search

For each sub-question, run **parallel searches** across all three vector collections:

```python
def hybrid_search(query: str, top_k: int = 20) -> list[SearchResult]:
    # Embed query with both models
    bge_dense_vec = bge_model.encode([query])['dense_vecs'][0]
    bge_sparse_vec = bge_model.encode([query])['lexical_weights'][0]
    kanon_dense_vec = kanon_client.embed(query, task="retrieval/query")
    
    # Parallel search
    results_bge_dense = qdrant.search("cfr101_bge_dense", bge_dense_vec, limit=top_k)
    results_bge_sparse = qdrant.search("cfr101_bge_sparse", bge_sparse_vec, limit=top_k)
    results_kanon = qdrant.search("cfr101_kanon_dense", kanon_dense_vec, limit=top_k)
    
    return results_bge_dense, results_bge_sparse, results_kanon
```

### Step 4 — RRF Rank Fusion + Cross-Encoder Reranking

**Reciprocal Rank Fusion** merges the three ranked lists:

```python
def rrf_fusion(ranked_lists: list[list], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            scores[doc.chunk_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:30]
```

**Cross-encoder reranking** — `bge-reranker-v2-m3` reads the query and each candidate chunk together and produces a true relevance score:

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def rerank(query: str, candidates: list[str]) -> list[tuple]:
    pairs = [(query, chunk_text) for chunk_text in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:10]
```

### Step 5 — Neo4j Cross-Reference Expansion

For the top-10 reranked results, run a graph expansion to pull in directly referenced sections:

```cypher
MATCH (s:Section {id: $section_id})-[:REFERENCES]->(related:Section)
RETURN related.id as ref_id, related.title as ref_title
LIMIT 5
```

These related sections are fetched from Qdrant (by `section_id` filter) and added to the context set. This is the step that ensures §101.62 always brings in §101.13 and §101.14.

---

## 10. Layer 5 — Multi-Agent Reasoning Layer

All agents run inside a **LangGraph** state machine. The state object flows through nodes and conditional edges.

### LangGraph State

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

class ComplianceState(TypedDict):
    # Input
    query: str
    uploaded_doc_text: str | None
    flow: str  # "query" or "document_analysis"
    
    # Intermediate
    sub_questions: list[str]
    retrieved_chunks: list[dict]
    graph_expanded_chunks: list[dict]
    extracted_claims: list[dict]     # for document analysis flow
    
    # Agent outputs
    definitions_resolved: dict[str, str]
    compliance_verdict: dict | None
    conflicts_detected: list[str]
    draft_answer: str
    verified_answer: str
    
    # Output
    final_response: dict
    citations: list[str]
    confidence_score: float
```

### Agent Roles

#### Planner Agent
- Receives the user query
- Classifies intent: `compliance_question` | `definition_lookup` | `comparison` | `document_analysis`
- Decomposes complex queries into sub-questions
- Routes to appropriate specialist agents via LangGraph conditional edges

#### Definition Resolver Agent
- Scans retrieved chunks for regulated terms (e.g., "nutrient content claim", "characterizing flavor", "RACC")
- Performs a targeted Neo4j + Qdrant lookup for the definition of each term
- Recursively resolves nested definitions (a definition that itself uses a defined term)
- Injects resolved definitions into the context before synthesis

#### Synthesizer Agent
- Receives: user query + all retrieved + expanded chunks + resolved definitions
- Generates a structured draft answer
- Extracts and formats CFR citations inline
- Produces a `confidence_score` (0–1) based on how well the retrieved chunks support the answer

#### Verifier Agent
- Cross-checks every factual claim in the draft answer against the source chunks
- Flags any claim that is not directly supported by retrieved text
- If a hallucination is detected, sends the state back to the Planner agent (LangGraph loop)
- Maximum 2 retry loops before outputting with a low-confidence flag

#### Conflict Detector Agent
- Checks whether two or more retrieved sections appear to contradict each other
- Uses a Neo4j query to detect known conflict edges
- Flags ambiguity in the final response (e.g., "Note: §101.14(e) provides an exception to the above that may apply depending on your specific claim")

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

### LiteLLM Gateway

All agent LLM calls route through **LiteLLM**, which provides:
- Model fallback chain: `Ollama (llama3.1:8b)` → `Groq (llama-3.1-70b)` → `OpenAI (gpt-4o)`
- Single API interface — swap models without changing agent code
- Automatic retry on rate limits

```python
import litellm

# LiteLLM automatically routes based on availability
response = litellm.completion(
    model="ollama/llama3.1:8b",  # tries Ollama first
    messages=[{"role": "user", "content": prompt}],
    fallbacks=["groq/llama-3.1-70b-versatile", "gpt-4o"],
    max_tokens=1000
)
```

### Async Processing for Document Analysis

Document analysis is queued via **Celery + Redis** so the API returns immediately:

```
POST /api/analyze-document → 202 Accepted → {"job_id": "job_abc123"}
GET /api/job/job_abc123    → {"status": "processing" | "complete", "result": {...}}
```

---

## 14. Observability & Evaluation

### Langfuse Tracing

Every pipeline step is traced:
```python
from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(name="compliance_query")
span = trace.span(name="retrieval_engine")
# ... retrieval code ...
span.end(output={"chunks_retrieved": 10, "latency_ms": 340})
```

Tracked metrics:
- Query → embedding latency
- Qdrant search latency + result scores
- Reranker latency + score distributions
- Agent execution time per step
- LLM call latency + token count
- End-to-end latency

### RAGAS Evaluation Suite

Build a **golden test set** of 100 Q&A pairs:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset=golden_test_set,
    metrics=[faithfulness, answer_relevancy, context_recall],
    llm=evaluation_llm,
    embeddings=bge_model
)
```

Target metrics:
- Faithfulness > 0.90 (answers grounded in retrieved text)
- Context Recall > 0.85 (all relevant sections retrieved)
- Answer Relevance > 0.88

### User Feedback Loop

```
POST /api/feedback
{
  "query_id": "q_abc123",
  "rating": 1-5,
  "correct_answer": "optional correction",
  "missing_citations": ["101.65"]
}
```

Feedback is logged to Langfuse and periodically used to:
- Extend the golden test set
- Identify weak retrieval spots
- Fine-tune the BGE-M3 embedding model

---

## 15. Technology Stack Summary

### Core Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Ingestion** | `lxml` | 5.x | XML parsing |
| **Ingestion** | `PyMuPDF` (fitz) | 1.24.x | PDF text extraction |
| **Ingestion** | `python-docx` | 1.x | DOCX text extraction |
| **Ingestion** | Tesseract OCR | 5.x | Image OCR |
| **Chunking** | Custom hierarchical splitter | — | CFR-aware chunking |
| **Embedding (primary)** | `BGE-M3` (BAAI/bge-m3) | — | Dense + sparse, local |
| **Embedding (secondary)** | Kanon 2 Embedder | — | Legal-specialized dense, API |
| **Reranker** | `bge-reranker-v2-m3` | — | Cross-encoder, local |
| **Vector store** | **Qdrant** | 1.9.x | Dense + sparse vectors |
| **Graph database** | **Neo4j** | 5.x | Cross-reference graph |
| **Agent framework** | **LangGraph** | 0.2.x | Stateful multi-agent |
| **LLM routing** | **LiteLLM** | 1.x | Ollama → Groq → OpenAI |
| **Local LLM** | Ollama + llama3.1:8b | — | Local inference |
| **API** | **FastAPI** | 0.115.x | Async REST API |
| **Task queue** | Celery + Redis | — | Async doc processing |
| **Observability** | **Langfuse** (self-hosted) | 2.x | Full pipeline tracing |
| **Evaluation** | **RAGAS** | 0.2.x | RAG quality metrics |
| **Containerization** | Docker Compose | — | Full local stack |

### Python Dependencies (`requirements.txt`)

```
# Ingestion
lxml==5.2.2
pymupdf==1.24.5
python-docx==1.1.2
pytesseract==0.3.13

# ML / Embedding
FlagEmbedding==1.3.0        # BGE-M3 + reranker
isaacus==0.1.x              # Kanon 2 API client
torch==2.3.0
sentence-transformers==3.1.0

# Databases
qdrant-client==1.9.1
neo4j==5.21.0

# Agent framework
langgraph==0.2.x
langchain==0.3.x
litellm==1.43.x

# API
fastapi==0.115.0
uvicorn==0.30.1
celery==5.4.0
redis==5.0.8

# Evaluation
ragas==0.2.x
langfuse==2.x

# Utilities
pydantic==2.8.x
python-multipart==0.0.9     # file uploads
httpx==0.27.0
```

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

### Phase 1 — Core RAG (Weeks 1–2)
*Goal: working end-to-end query flow with basic retrieval*

- [ ] Set up Docker Compose (Qdrant + Neo4j + Redis)
- [ ] Write eCFR XML downloader and parser (`ingestion/xml_parser.py`)
- [ ] Implement contextual chunker with overlap
- [ ] Set up BGE-M3 embedding (local, dense only first)
- [ ] Index CFR Part 101 into Qdrant dense collection
- [ ] Build basic FastAPI `/api/query` endpoint
- [ ] Implement single-model dense retrieval
- [ ] Wire up LiteLLM with Ollama + Groq fallback
- [ ] Build simple synthesizer agent (no LangGraph yet)
- [ ] Manual testing of 20 questions

### Phase 2 — Hybrid Retrieval + Reranking (Week 3)
*Goal: significantly improve retrieval quality*

- [ ] Add BGE-M3 sparse (SPLADE) collection to Qdrant
- [ ] Add Kanon 2 Embedder collection (via API)
- [ ] Implement RRF rank fusion across 3 result lists
- [ ] Integrate `bge-reranker-v2-m3` cross-encoder
- [ ] Add HyDE query expansion
- [ ] Build Neo4j graph (section nodes + cross-reference edges)
- [ ] Implement graph expansion step in retrieval pipeline
- [ ] Run RAGAS baseline evaluation (golden test set v1)

### Phase 3 — Multi-Agent System (Week 4)
*Goal: production-quality reasoning and citation*

- [ ] Set up LangGraph state machine
- [ ] Implement Planner agent (query decomposition + intent routing)
- [ ] Implement Definition Resolver agent (recursive term lookup)
- [ ] Implement Synthesizer agent (structured output + citations)
- [ ] Implement Verifier agent (hallucination detection + retry loop)
- [ ] Implement Conflict Detector agent (Neo4j conflict query)
- [ ] Wire all agents into LangGraph graph with conditional edges
- [ ] End-to-end testing with 50 compliance questions

### Phase 4 — Document Violation Analysis Flow (Weeks 5–6)
*Goal: full document upload and analysis pipeline*

- [ ] Build Document Parser Agent (PDF + DOCX + image)
- [ ] Build Claim Extractor Agent (LLM structured output)
- [ ] Build Regulation Mapper Agent (per-claim hybrid search)
- [ ] Build Gap Analyzer Agent (reverse pass over all CFR sections)
- [ ] Build Violation Classifier Agent (LLM verification + severity scoring)
- [ ] Build structured `ViolationReport` output schema
- [ ] Set up Celery async task queue for document processing
- [ ] Build `POST /api/analyze-document` and `GET /api/job/{id}` endpoints
- [ ] Test with 10 real food label documents (manually verify outputs)

### Phase 5 — Observability, Evaluation & Fine-Tuning (Weeks 7–8)
*Goal: measure quality, improve, and make the system trustworthy*

- [ ] Deploy self-hosted Langfuse (Docker)
- [ ] Add Langfuse tracing to all pipeline steps
- [ ] Build golden test set of 100 Q&A pairs
- [ ] Run full RAGAS evaluation suite
- [ ] Collect 500 query-passage pairs for BGE-M3 fine-tuning
- [ ] Fine-tune BGE-M3 with `sentence-transformers` MultipleNegativesRankingLoss
- [ ] Add user feedback endpoint
- [ ] Performance optimization (async retrieval, batch embedding)
- [ ] Final load testing

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

## 19. Directory Structure

```
fda-compliance-rag/
├── docker-compose.yml          # Qdrant + Neo4j + Redis + Langfuse + API
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── xml_parser.py           # eCFR XML → section dicts
│   ├── chunker.py              # Contextual hierarchical chunker
│   ├── embedder.py             # BGE-M3 + Kanon 2 embedding
│   ├── graph_builder.py        # Neo4j ingestion
│   ├── qdrant_indexer.py       # Qdrant upsert (3 collections)
│   └── run_ingestion.py        # Orchestrates full ingestion pipeline
│
├── retrieval/
│   ├── hybrid_search.py        # Multi-collection Qdrant search
│   ├── rrf_fusion.py           # Reciprocal Rank Fusion
│   ├── reranker.py             # bge-reranker-v2-m3 cross-encoder
│   ├── hyde.py                 # Hypothetical Document Embedding
│   ├── graph_expander.py       # Neo4j 1-hop expansion
│   └── retrieval_pipeline.py   # Orchestrates all retrieval steps
│
├── agents/
│   ├── state.py                # LangGraph ComplianceState TypedDict
│   ├── planner.py              # Query decomposition + intent routing
│   ├── definition_resolver.py  # Recursive term definition lookup
│   ├── synthesizer.py          # Draft answer generation + citations
│   ├── verifier.py             # Hallucination detection + retry
│   ├── conflict_detector.py    # Cross-section conflict detection
│   └── graph.py                # LangGraph graph assembly
│
├── document_analysis/
│   ├── parser.py               # PDF + DOCX + image text extraction
│   ├── claim_extractor.py      # LLM-based claim extraction
│   ├── regulation_mapper.py    # Per-claim hybrid search
│   ├── gap_analyzer.py         # Reverse pass over CFR corpus
│   ├── violation_classifier.py # LLM verification + severity scoring
│   └── report_builder.py       # ViolationReport assembly
│
├── api/
│   ├── main.py                 # FastAPI app
│   ├── routers/
│   │   ├── query.py            # POST /api/query
│   │   ├── document.py         # POST /api/analyze-document
│   │   ├── jobs.py             # GET /api/job/{job_id}
│   │   └── feedback.py         # POST /api/feedback
│   ├── schemas.py              # Pydantic request/response models
│   └── celery_app.py           # Celery task definitions
│
├── evaluation/
│   ├── golden_test_set.json    # 100 Q&A pairs with citations
│   ├── run_ragas.py            # RAGAS evaluation runner
│   └── generate_test_pairs.py  # Script to semi-automate test pair gen
│
├── scripts/
│   ├── download_ecfr.sh        # Download CFR Part 101 XML
│   ├── setup_neo4j.cypher      # Neo4j schema + constraints
│   └── setup_qdrant.py         # Create Qdrant collections
│
└── notebooks/
    ├── 01_explore_cfr_xml.ipynb
    ├── 02_chunking_experiments.ipynb
    ├── 03_embedding_comparison.ipynb
    └── 04_retrieval_quality_analysis.ipynb
```

---

## Appendix A — Embedding Model Decision Summary

| Model | Legal Accuracy | Biomedical Accuracy | Local? | Sparse? | Context | Decision |
|-------|---------------|--------------------|----|------|---------|----------|
| E5-base-v2 | Low | Low | ✅ | ❌ | 512 | ❌ Ruled out |
| E5-large-v2 | Low-Medium | Low-Medium | ✅ | ❌ | 512 | ❌ Ruled out |
| PubMedBERT | Low | Very High | ✅ | ❌ | 512 | ❌ Wrong domain fit |
| BiomedBERT | Low | Very High | ✅ | ❌ | 512 | ❌ Wrong domain fit |
| BGE-M3 | Medium-High | Medium | ✅ | ✅ | 8,192 | ✅ **Primary local model** |
| Kanon 2 Embedder | Very High (91.48 NDCG@10 regulatory) | Low | API | ❌ | 16,384 | ✅ **Secondary legal model** |
| Qwen3-Embedding-8B | High | Medium | ✅ (GPU) | ❌ | 8,192 | ⚠️ Upgrade path (GPU required) |

**Why not PubMedBERT/BiomedBERT**: These models excel at clinical notes and PubMed abstracts but are completely blind to regulatory legal structure. The biomedical terminology in CFR Part 101 (vitamin/mineral thresholds, fatty acid definitions) is representable by BGE-M3 after domain fine-tuning. Using PubMedBERT would sacrifice all legal structural understanding for marginal gains on nutrient terminology.

**The dual-model strategy rationale**: BGE-M3 covers the structural regulatory language and provides BM25-equivalent sparse retrieval. Kanon 2 provides state-of-the-art regulatory semantic understanding. Their combination via RRF fusion, followed by a cross-encoder reranker that reads full (query, chunk) pairs, gives the best of all worlds — without requiring a GPU for inference on the primary model.

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
