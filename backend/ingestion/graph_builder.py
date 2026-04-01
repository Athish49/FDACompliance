"""
CFR Chunk → Neo4j Lexical Graph Builder
=========================================
Reads the chunks JSON produced by extractor.py and builds a labeled property
graph in Neo4j that mirrors the lexical graph schema defined in
ChunksStructureExample.json.

Node labels created:
  (:Title)          number, name
  (:Chapter)        number, name
  (:Part)           number, name
  (:Chunk)          chunk_id, chunk_type, cfr_citation, text,
                    token_count_approx
  (:DefinedTerm)    term, defined_in_section
  (:ExternalCitation) citation_text

Relationships created:
  (Title)-[:CONTAINS]->(Chapter)
  (Chapter)-[:CONTAINS]->(Subchapter | Part)
  (Subchapter)-[:CONTAINS]->(Part)
  (Part)-[:CONTAINS]->(Subpart | Section)
  (Subpart)-[:CONTAINS]->(Section)
  (Section)-[:CONTAINS]->(Chunk)
  (Chunk)-[:CROSS_REFERENCES]->(Section)   for each cross_references_internal
  (Chunk)-[:DEFINES_TERM]->(DefinedTerm)   for definition chunks
  (Chunk)-[:USES_TERM]->(DefinedTerm)      where term appears in text
  (Chunk)-[:CITES]->(ExternalCitation)
  (Chunk)-[:OVERFLOW_CONTINUES]->(Chunk)   for overflow chunk chains

Dependencies (add to requirements.txt):
  neo4j>=5.0.0
  tqdm>=4.0.0
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphConfig:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    database: str = "neo4j"

    # Number of chunks to process per transaction
    batch_size: int = 200

    # When True, wipe the database before ingesting (use only in dev)
    clear_on_start: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Cypher queries
# ─────────────────────────────────────────────────────────────────────────────

# ── Hierarchy nodes ──────────────────────────────────────────────────────────

_MERGE_TITLE = """
MERGE (n:Title {number: $number})
  ON CREATE SET n.name = $name
"""

_MERGE_CHAPTER = """
MATCH (t:Title {number: $title_number})
MERGE (n:Chapter {number: $number, title_number: $title_number})
  ON CREATE SET n.name = $name
MERGE (t)-[:CONTAINS]->(n)
"""

_MERGE_SUBCHAPTER = """
MATCH (c:Chapter {number: $chapter_number, title_number: $title_number})
MERGE (n:Subchapter {letter: $letter, chapter_number: $chapter_number, title_number: $title_number})
  ON CREATE SET n.name = $name
MERGE (c)-[:CONTAINS]->(n)
"""

_MERGE_PART = """
MATCH (parent {number: $parent_number, title_number: $title_number})
WHERE $parent_label IN labels(parent)
MERGE (n:Part {number: $part_number, title_number: $title_number})
  ON CREATE SET n.name = $part_name
MERGE (parent)-[:CONTAINS]->(n)
"""

_MERGE_SUBPART = """
MATCH (p:Part {number: $part_number, title_number: $title_number})
MERGE (n:Subpart {letter: $letter, part_number: $part_number, title_number: $title_number})
  ON CREATE SET n.name = $name
MERGE (p)-[:CONTAINS]->(n)
"""

_MERGE_SECTION = """
MERGE (n:Section {number: $number, title_number: $title_number})
  ON CREATE SET n.name = $name, n.cfr_citation = $cfr_citation
WITH n
MATCH (parent)
WHERE (parent:Subpart AND parent.letter = $subpart_letter
       AND parent.part_number = $part_number
       AND parent.title_number = $title_number)
   OR (parent:Part AND parent.number = $part_number
       AND parent.title_number = $title_number
       AND $subpart_letter IS NULL)
MERGE (parent)-[:CONTAINS]->(n)
"""

# ── Chunk node ───────────────────────────────────────────────────────────────

_MERGE_CHUNK = """
MERGE (c:Chunk {chunk_id: $chunk_id})
  ON CREATE SET
    c.chunk_type            = $chunk_type,
    c.cfr_citation          = $cfr_citation,
    c.text                  = $text,
    c.section_preamble      = $section_preamble,
    c.defines               = $defines,
    c.token_count_approx    = $token_count_approx,
    c.is_overflow_chunk     = $is_overflow_chunk,
    c.metrics               = $metrics,
    c.cross_refs_internal   = $cross_refs_internal
WITH c
MATCH (s:Section {number: $section_number, title_number: $title_number})
MERGE (s)-[:CONTAINS]->(c)
"""

# ── Defined term ─────────────────────────────────────────────────────────────

_MERGE_DEFINED_TERM = """
MERGE (t:DefinedTerm {term: $term, defined_in_section: $defined_in_section})
WITH t
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (c)-[:DEFINES_TERM]->(t)
"""

_LINK_USES_TERM = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (t:DefinedTerm {term: $term})
MERGE (c)-[:USES_TERM]->(t)
"""

# ── External citation ─────────────────────────────────────────────────────────

_MERGE_EXTERNAL_CITATION = """
MERGE (e:ExternalCitation {citation_text: $citation_text})
WITH e
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (c)-[:CITES]->(e)
"""

# ── Cross-reference to section ────────────────────────────────────────────────

_LINK_CROSS_REF = """
MATCH (c:Chunk {chunk_id: $chunk_id})
MATCH (s:Section {number: $section_number, title_number: $title_number})
MERGE (c)-[:CROSS_REFERENCES]->(s)
"""

# ── Overflow chain ───────────────────────────────────────────────────────────

_LINK_OVERFLOW = """
MATCH (a:Chunk {chunk_id: $from_chunk_id})
MATCH (b:Chunk {chunk_id: $to_chunk_id})
MERGE (a)-[:OVERFLOW_CONTINUES]->(b)
"""

# ── Constraints / indexes ────────────────────────────────────────────────────

_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk)            REQUIRE n.chunk_id       IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Section)          REQUIRE (n.number, n.title_number) IS NODE KEY",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DefinedTerm)      REQUIRE (n.term, n.defined_in_section) IS NODE KEY",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ExternalCitation) REQUIRE n.citation_text  IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Title)            REQUIRE n.number         IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (n:Chunk)    ON (n.chunk_type)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Section)  ON (n.cfr_citation)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Part)     ON (n.number)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_chunks(chunks_path: str | Path) -> list[dict]:
    with open(chunks_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["chunks"]


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _safe(value, fallback=None):
    return value if value is not None else fallback


# ─────────────────────────────────────────────────────────────────────────────
# Graph building
# ─────────────────────────────────────────────────────────────────────────────

class CFRGraphBuilder:
    """Orchestrates Neo4j writes for the CFR lexical graph."""

    def __init__(self, config: GraphConfig):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("neo4j is required. Install with: pip install neo4j") from exc

        self._driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        self._db = config.database
        self._config = config

    def close(self):
        self._driver.close()

    def _run(self, session, query: str, **params):
        session.run(query, **params)

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup_schema(self):
        """Create constraints and indexes (idempotent)."""
        with self._driver.session(database=self._db) as session:
            for stmt in _CONSTRAINTS:
                try:
                    session.run(stmt)
                except Exception as exc:
                    logger.debug("Constraint/index skipped: %s — %s", stmt[:60], exc)
        logger.info("Schema constraints and indexes applied.")

    def clear_database(self):
        """Remove all nodes and relationships. USE WITH CAUTION."""
        with self._driver.session(database=self._db) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Database cleared.")

    # ── Hierarchy ────────────────────────────────────────────────────────────

    def _upsert_hierarchy(self, session, chunk: dict):
        h = chunk.get("hierarchy", {})
        title   = h.get("title") or {}
        chapter = h.get("chapter") or {}
        subchap = h.get("subchapter")
        part    = h.get("part") or {}
        subpart = h.get("subpart")

        title_num   = _safe(title.get("number"), "21")
        chapter_num = _safe(chapter.get("number"), "?")
        part_num    = _safe(part.get("number"), "?")

        # Title
        session.run(_MERGE_TITLE, number=title_num, name=_safe(title.get("name"), ""))

        # Chapter
        session.run(
            _MERGE_CHAPTER,
            title_number=title_num,
            number=chapter_num,
            name=_safe(chapter.get("name"), ""),
        )

        # Subchapter (chapI files)
        if subchap:
            session.run(
                _MERGE_SUBCHAPTER,
                title_number=title_num,
                chapter_number=chapter_num,
                letter=subchap.get("letter", "?"),
                name=_safe(subchap.get("name"), ""),
            )

        # Part — parent is Subchapter if present, else Chapter
        parent_label = "Subchapter" if subchap else "Chapter"
        parent_number = subchap.get("letter", "?") if subchap else chapter_num
        session.run(
            _MERGE_PART,
            title_number=title_num,
            parent_label=parent_label,
            parent_number=parent_number,
            part_number=part_num,
            part_name=_safe(part.get("name"), ""),
        )

        # Subpart
        if subpart:
            session.run(
                _MERGE_SUBPART,
                title_number=title_num,
                part_number=part_num,
                letter=_safe(subpart.get("letter"), "?"),
                name=_safe(subpart.get("name"), ""),
            )

        # Section
        sec = h.get("section") or {}
        sec_num = _safe(sec.get("number"), "?")
        session.run(
            _MERGE_SECTION,
            title_number=title_num,
            number=sec_num,
            name=_safe(sec.get("name"), ""),
            cfr_citation=chunk.get("cfr_citation", ""),
            part_number=part_num,
            subpart_letter=subpart.get("letter") if subpart else None,
        )

    # ── Chunk node ───────────────────────────────────────────────────────────

    def _upsert_chunk(self, session, chunk: dict):
        h = chunk.get("hierarchy", {})
        sec = h.get("section") or {}
        title_num = (h.get("title") or {}).get("number", "21")

        session.run(
            _MERGE_CHUNK,
            chunk_id=chunk["chunk_id"],
            chunk_type=chunk.get("chunk_type", ""),
            cfr_citation=chunk.get("cfr_citation", ""),
            text=chunk.get("text", ""),
            section_preamble=chunk.get("section_preamble") or "",
            defines=chunk.get("defines") or "",
            token_count_approx=chunk.get("token_count_approx", 0),
            is_overflow_chunk=chunk.get("is_overflow_chunk", False),
            metrics=chunk.get("metrics", []),
            cross_refs_internal=chunk.get("cross_references_internal", []),
            section_number=_safe(sec.get("number"), "?"),
            title_number=title_num,
        )

    # ── Semantic edges ───────────────────────────────────────────────────────

    def _upsert_chunk_edges(self, session, chunk: dict):
        chunk_id = chunk["chunk_id"]
        title_num = (chunk.get("hierarchy", {}).get("title") or {}).get("number", "21")

        # DEFINES_TERM / defined term node
        if chunk.get("chunk_type") == "definition" and chunk.get("defines"):
            sec_num = (chunk.get("hierarchy", {}).get("section") or {}).get("number", "?")
            session.run(
                _MERGE_DEFINED_TERM,
                term=chunk["defines"],
                defined_in_section=sec_num,
                chunk_id=chunk_id,
            )

        # External citations
        for cit in chunk.get("citations", []):
            session.run(
                _MERGE_EXTERNAL_CITATION,
                citation_text=cit,
                chunk_id=chunk_id,
            )

        # Internal cross-references (§ X.Y)
        for ref_sec_num in chunk.get("cross_references_internal", []):
            session.run(
                _LINK_CROSS_REF,
                chunk_id=chunk_id,
                section_number=ref_sec_num,
                title_number=title_num,
            )

        # Overflow chain
        overflow_seq = chunk.get("overflow_sequence") or {}
        next_chunk_id = overflow_seq.get("next_chunk_id")
        if next_chunk_id:
            session.run(
                _LINK_OVERFLOW,
                from_chunk_id=chunk_id,
                to_chunk_id=next_chunk_id,
            )

    # ── USES_TERM pass (second pass after all terms are defined) ─────────────

    def _link_uses_terms(self, chunks: list[dict]):
        """
        For every chunk whose text contains a defined term, add a USES_TERM edge.
        This is a second pass so DefinedTerm nodes exist before we link them.
        """
        # Build a lookup: term string → defined_in_section
        defined_terms: dict[str, str] = {}
        for chunk in chunks:
            if chunk.get("chunk_type") == "definition" and chunk.get("defines"):
                sec_num = (chunk.get("hierarchy", {}).get("section") or {}).get("number", "?")
                defined_terms[chunk["defines"]] = sec_num

        if not defined_terms:
            return

        logger.info("Linking USES_TERM edges for %d defined terms …", len(defined_terms))
        with self._driver.session(database=self._db) as session:
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                text = chunk.get("text", "")
                for term, sec_num in defined_terms.items():
                    if term.lower() in text.lower():
                        session.run(
                            _LINK_USES_TERM,
                            chunk_id=chunk_id,
                            term=term,
                        )

    # ── Orchestration ────────────────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> dict:
        """
        Write all hierarchy nodes, chunk nodes, and edges to Neo4j.
        Returns a summary dict.
        """
        total = len(chunks)
        processed = 0
        start = time.time()

        for batch in _batched(chunks, self._config.batch_size):
            with self._driver.session(database=self._db) as session:
                with session.begin_transaction() as tx:
                    for chunk in batch:
                        self._upsert_hierarchy(tx, chunk)
                        self._upsert_chunk(tx, chunk)
                        self._upsert_chunk_edges(tx, chunk)
                    tx.commit()
            processed += len(batch)
            logger.debug("Graph: %d / %d chunks processed", processed, total)

        # Second pass for USES_TERM (needs all DefinedTerm nodes to exist)
        self._link_uses_terms(chunks)

        duration = round(time.time() - start, 2)
        logger.info(
            "Graph build complete: %d chunks → Neo4j in %.1fs",
            total,
            duration,
        )
        return {"total_chunks_written": total, "duration_seconds": duration}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    chunks_path: str | Path = "data/chunks/cfr_chunks.json",
    config: Optional[GraphConfig] = None,
) -> dict:
    """
    Load chunks from *chunks_path* and build the Neo4j lexical graph.
    Returns a summary dict.
    """
    if config is None:
        config = GraphConfig()

    chunks = _load_chunks(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    builder = CFRGraphBuilder(config)
    try:
        builder.setup_schema()
        if config.clear_on_start:
            builder.clear_database()
        return builder.build(chunks)
    finally:
        builder.close()
