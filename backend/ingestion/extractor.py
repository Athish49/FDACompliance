"""
CFR XML → Chunk JSON Extractor
================================
Parses all CFR XML files from the data directory, applies paragraph-level
chunking, and writes a structured JSON file ready for embedding + graph
ingestion.

Chunking strategy (matches ChunksStructureExample.json):
  - SECTION with 1 P element           → single "section" chunk
  - SECTION with labeled (a)(b)..      → one "paragraph" chunk per letter label
  - Definitions SECTION                → one "definition" chunk per defined term
  - Paragraph > 400 tokens             → split at sentence boundary, keep overlap tail
  - Numbered sub-points (1)(2)..       → embedded in parent paragraph chunk
  - Roman-numeral sub-points (i)(ii).. → embedded in parent paragraph chunk
  - GPOTABLE                           → serialized as markdown table in the chunk text

Tags skipped (non-regulatory):
  TOC, TOCHD, CONTENTS, CHAPTI, PTHD, PGHD, SECHD, EAR, PRTPAGE,
  LRH, RRH, FAIDS, ALPHLIST, CITE, EXPLA, IPAR, STUB, SIDEHED,
  SIG, NAME, POSITION, OFFICE, DATE, EDNOTE

XML hierarchy understood:
  CFRGRANULE
    FDSYS                      → document-level metadata
    CHAPTER
      TOC                      → SKIP
      [SUBCHAP TYPE="N"]*      → subchapter (chapI files only)
        HD                     → "SUBCHAPTER A—GENERAL"
        PART+
      [PART]*                  → parts directly under CHAPTER (chapIII)
        HD                     → "PART 1401—PUBLIC AVAILABILITY..."
        CONTENTS               → SKIP (TOC for the part)
        SOURCE                 → part source FR citation
        [SUBPART]*
          HD                   → "Subpart A—Freedom of Information..."
          SECTION+
        [SECTION]*             → sections directly under PART (no subpart)
          SECTNO / SUBJECT / P / GPOTABLE / EXTRACT / CITA
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SKIP_TAGS = frozenset({
    "TOC", "TOCHD", "CONTENTS", "CHAPTI", "PTHD", "PGHD", "SECHD",
    "EAR", "PRTPAGE", "LRH", "RRH", "FAIDS", "ALPHLIST",
    "EXPLA", "IPAR", "STUB", "SIDEHED", "SIG", "NAME",
    "POSITION", "OFFICE", "EDNOTE",
})

OVERFLOW_TOKEN_THRESHOLD = 400   # tokens; paragraph chunks above this get split

# Paragraph label regexes (matched at start of text)
# CFR four-level paragraph hierarchy:
#   Level 1 → letter     (a), (b), (c) …
#   Level 2 → number     (1), (2), (3) …
#   Level 3 → roman      (i), (ii), (iii) …
#   Level 4 → cap-letter (A), (B), (C) …
_LETTER_LABEL    = re.compile(r"^\s*\(\s*([a-z])\s*\)\s+")
_NUMBER_LABEL    = re.compile(r"^\s*\(\s*(\d+)\s*\)\s+")
_ROMAN_LABEL     = re.compile(r"^\s*\(\s*([ivxlcdm]{1,8})\s*\)\s+", re.IGNORECASE)
_CAPLETTER_LABEL = re.compile(r"^\s*\(\s*([A-Z])\s*\)\s+")

# Citation patterns for metadata extraction
_INTERNAL_SECTION_RE = re.compile(r"§\s*(\d{3,5}\.\d+(?:\([a-z0-9]+\))*)", re.IGNORECASE)
_USC_RE    = re.compile(r"\d+\s+U\.S\.C\.?\s+[\d§\s]+[\d][a-z]?(?:\([a-z0-9]+\))?", re.IGNORECASE)
_CFR_RE    = re.compile(r"\d+\s+CFR\s+(?:parts?\s+)?\d+(?:\.\d+)?", re.IGNORECASE)
_FR_RE     = re.compile(r"\d+\s+FR\s+\d+", re.IGNORECASE)
_EO_RE     = re.compile(r"E\.O\.\s*\d+|Executive\s+Order\s+\d+", re.IGNORECASE)



# Heading parsers
_SUBCHAP_HD_RE = re.compile(r"SUBCHAPTER\s+([A-Z]+)\s*[—\-–]\s*(.+)", re.IGNORECASE)
_PART_HD_RE    = re.compile(r"PART\s+(\d+[A-Z]?)\s*[—\-–]\s*(.+)", re.IGNORECASE)
_SUBPART_HD_RE = re.compile(r"[Ss]ubpart\s+([A-Z]+)\s*[—\-–]\s*(.+)")

# Sentence split (conservative — keeps sentence boundaries)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(])")


# ─────────────────────────────────────────────────────────────────────────────
# Low-level XML text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(element: ET.Element, skip: tuple[str, ...] = ("PRTPAGE",)) -> str:
    """
    Recursively collect text from *element*, skipping subtrees whose tag is in *skip*.
    Normalises internal whitespace to single spaces.
    """
    if element is None:
        return ""
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if child.tag not in skip:
            parts.append(_extract_text(child, skip))
        if child.tail:
            parts.append(child.tail)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _approx_tokens(text: str) -> int:
    """Rough word-count-based token estimate (words × 1.35)."""
    return max(1, int(len(text.split()) * 1.35))


def _get_text(element: Optional[ET.Element]) -> str:
    return _extract_text(element) if element is not None else ""


# ─────────────────────────────────────────────────────────────────────────────
# Paragraph label detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_label(text: str, has_level2: bool) -> Optional[tuple[int, str]]:
    """
    Detect the CFR paragraph label at the start of *text*.
    Returns (level, label_str) or None.

    Level hierarchy:
      1 = letter     (a), (b) …  [lowercase, not roman-in-context]
      2 = number     (1), (2) …
      3 = roman      (i), (ii), (iii) …
      4 = cap_letter (A), (B) …

    Disambiguation: single-char lowercase roman letters (i, v, x, l, c, d, m)
    are treated as level-3 roman only when a level-2 number has already appeared
    in the current stack (has_level2=True).
    """
    m = _NUMBER_LABEL.match(text)
    if m:
        return (2, f"({m.group(1)})")
    m = _ROMAN_LABEL.match(text)
    if m:
        rv = m.group(1).lower()
        if len(rv) > 1:
            return (3, f"({rv})")
    m = _CAPLETTER_LABEL.match(text)
    if m:
        return (4, f"({m.group(1)})")
    m = _LETTER_LABEL.match(text)
    if m:
        ch = m.group(1)
        # Single-char roman letters after a number label → level 3 roman
        if has_level2 and ch in "ivxlcdm":
            return (3, f"({ch})")
        return (1, f"({ch})")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metric / citation extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_external_citations(text: str) -> list[str]:
    found: set[str] = set()
    for pattern in (_USC_RE, _CFR_RE, _FR_RE, _EO_RE):
        for m in pattern.finditer(text):
            found.add(re.sub(r"\s+", " ", m.group(0)).strip())
    return sorted(found)


def _extract_internal_refs(text: str) -> list[str]:
    return sorted({m.group(1) for m in _INTERNAL_SECTION_RE.finditer(text)})


# ─────────────────────────────────────────────────────────────────────────────
# GPOTABLE serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_table(table_el: ET.Element) -> str:
    """Convert a <GPOTABLE> element to a markdown-style plain-text table."""
    lines: list[str] = []
    boxhd = table_el.find("BOXHD")
    headers: list[str] = []
    if boxhd is not None:
        headers = [_extract_text(c) for c in boxhd.findall("CHED")]
    if headers:
        lines.append(" | ".join(headers))
        lines.append(" | ".join(["---"] * len(headers)))
    for row in table_el.findall("ROW"):
        cells = [_extract_text(e) for e in row.findall("ENT")]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk ID generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk_id(
    title: str,
    chapter: str,
    subchap: Optional[str],
    part: str,
    subpart: Optional[str],
    section: str,
    suffix: str,
) -> str:
    """
    Builds a deterministic, URL-safe chunk identifier.
    Format: {title}-{chap}-{part}-{subpart}-{section}-{suffix}
    """
    def _clean(s: Optional[str], fallback: str = "X") -> str:
        if not s:
            return fallback
        return re.sub(r"[^A-Za-z0-9]", "", s).upper()

    chap_part = _clean(chapter)
    subchap_part = f"SC{_clean(subchap)}" if subchap else ""
    part_part = part.replace(" ", "").replace(".", "-")
    subpart_part = _clean(subpart) if subpart else "nosub"
    sec_part = re.sub(r"[^A-Za-z0-9]", "-", section).strip("-")
    suf_part = re.sub(r"[^A-Za-z0-9]", "-", suffix).strip("-")

    components = filter(None, [title, chap_part, subchap_part, part_part, subpart_part, sec_part, suf_part])
    return "-".join(components)


# ─────────────────────────────────────────────────────────────────────────────
# Overflow splitting
# ─────────────────────────────────────────────────────────────────────────────

def _split_overflow(text: str, threshold: int = OVERFLOW_TOKEN_THRESHOLD) -> list[str]:
    """
    Split *text* at sentence boundaries when it exceeds *threshold* tokens.
    Returns a list of text parts (length 1 if no split needed).
    """
    if _approx_tokens(text) <= threshold:
        return [text]

    sentences = _SENTENCE_END.split(text)
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _approx_tokens(sentence)
        if current_tokens + s_tokens > threshold and current:
            parts.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += s_tokens

    if current:
        parts.append(" ".join(current))

    return parts if len(parts) > 1 else [text]


# ─────────────────────────────────────────────────────────────────────────────
# Chunk factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_chunk(
    chunk_id: str,
    chunk_type: str,
    cfr_citation: str,
    hierarchy: dict,
    text: str,
    section_preamble: Optional[str],
    doc_meta: dict,
    *,
    defines: Optional[str] = None,
    overflow_sequence: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Assembles the standard chunk dict."""
    chunk = {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "cfr_citation": cfr_citation,
        "hierarchy": hierarchy,
        "section_preamble": section_preamble,
        "text": text,
        "citations": _extract_external_citations(text),
        "cross_references_internal": _extract_internal_refs(text),
        "token_count_approx": _approx_tokens(text),
        "is_overflow_chunk": overflow_sequence is not None,
        "overflow_sequence": overflow_sequence,
    }
    if defines is not None:
        chunk["defines"] = defines
    if extra:
        chunk.update(extra)
    return chunk


# ─────────────────────────────────────────────────────────────────────────────
# Definitions section handler
# ─────────────────────────────────────────────────────────────────────────────

def _is_definitions_section(subject: str) -> bool:
    return "definition" in subject.lower()


def _chunk_definitions_section(
    content_els: list[ET.Element],
    hierarchy: dict,
    doc_meta: dict,
    id_parts: tuple,
) -> list[dict]:
    """
    Each <P> that has an <E> as its first child (a defined term) becomes
    a separate "definition" chunk.  Unlabelled preamble P elements are kept
    as the section_preamble shared by all definition chunks.
    """
    preamble: Optional[str] = None
    chunks: list[dict] = []

    for el in content_els:
        if el.tag not in ("P", "FP"):
            continue
        text = _extract_text(el)
        if not text:
            continue

        # Check whether the FIRST child is an <E> element (a defined term)
        first_child = next(iter(el), None)
        if first_child is not None and first_child.tag == "E" and first_child.text:
            term_name = first_child.text.strip()
            term_slug = re.sub(r"[^a-z0-9]+", "-", term_name.lower()).strip("-")

            chunk_id = _make_chunk_id(
                *id_parts,
                suffix=f"def-{term_slug}",
            )
            cfr_citation = (
                f"21 CFR § {hierarchy['section']['number']} "
                f"(definition: {term_name})"
            )
            chunk = _build_chunk(
                chunk_id=chunk_id,
                chunk_type="definition",
                cfr_citation=cfr_citation,
                hierarchy={**hierarchy, "paragraph": None},
                text=text,
                section_preamble=preamble,
                doc_meta=doc_meta,
                defines=term_name,
            )
            chunks.append(chunk)
        else:
            # Treat as preamble / general text for the definitions section
            if preamble is None:
                preamble = text

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Regular section handler — paragraph-level chunking
# ─────────────────────────────────────────────────────────────────────────────

def _collect_content_texts(content_els: list[ET.Element]) -> list[tuple[str, str]]:
    """
    Returns [(kind, text), ...] where kind is 'p' or 'table'.
    EXTRACT elements are treated as 'p'.
    """
    items: list[tuple[str, str]] = []
    for el in content_els:
        if el.tag in ("P", "FP", "EXTRACT"):
            t = _extract_text(el)
            if t:
                items.append(("p", t))
        elif el.tag == "GPOTABLE":
            t = _serialize_table(el)
            if t:
                items.append(("table", t))
    return items


def _chunk_regular_section(
    content_els: list[ET.Element],
    hierarchy: dict,
    doc_meta: dict,
    id_parts: tuple,
) -> list[dict]:
    """
    Paragraph-level chunking for non-definition sections.

    Every labeled P element at every CFR hierarchy level becomes its own chunk.
    The compound label path (e.g. "(a)(1)(i)") is built via a stack machine and
    stored in hierarchy.paragraph.label.

    Algorithm:
      1. First unlabeled item → section_preamble (shared by all chunks)
      2. Walk all items maintaining a compound-path stack
         - Each labeled P → new group with compound label
         - Unlabeled P / table → folded into the most recent group
      3. No labels found → single "section" chunk
      4. Each group → one or more "paragraph" chunks (split on overflow)
    """
    items = _collect_content_texts(content_els)
    if not items:
        return []

    sec_num   = hierarchy["section"]["number"]
    base_cite = f"21 CFR § {sec_num}"

    # ── Step 1: preamble detection ────────────────────────────────────────
    preamble: Optional[str] = None
    start_idx = 0
    first_label = _detect_label(items[0][1], False) if items and items[0][0] == "p" else None
    if items and first_label is None:
        preamble = items[0][1]
        start_idx = 1

    # ── Step 2: build groups via compound-path stack ──────────────────────
    # stack: [(level, label_str), ...]
    # groups: [(compound_label, [text, ...]), ...]

    stack: list[tuple[int, str]] = []
    groups: list[tuple[str, list[str]]] = []

    for kind, text in items[start_idx:]:
        if kind == "p":
            has_level2 = any(lvl == 2 for lvl, _ in stack)
            label_info = _detect_label(text, has_level2)
        else:
            label_info = None

        if label_info is not None:
            level, leaf_label = label_info
            # Pop stack entries at the same or deeper level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, leaf_label))
            compound = "".join(lbl for _, lbl in stack)
            groups.append((compound, [text]))
        else:
            # Unlabeled content: fold into current group or extend preamble
            if groups:
                groups[-1][1].append(text)
            elif preamble is None:
                preamble = text
            else:
                preamble += " " + text

    # ── Step 3: no labels → single section chunk ─────────────────────────
    if not groups:
        all_text = " ".join(t for _, t in items)
        chunk_id = _make_chunk_id(*id_parts, suffix="sec")
        return [
            _build_chunk(
                chunk_id=chunk_id,
                chunk_type="section",
                cfr_citation=base_cite,
                hierarchy={**hierarchy, "paragraph": None},
                text=all_text,
                section_preamble=None,
                doc_meta=doc_meta,
            )
        ]

    # ── Step 4: one (or more overflow) chunk per labeled group ───────────
    chunks: list[dict] = []
    for compound_label, texts in groups:
        para_text = " ".join(texts)

        para_hierarchy = {
            **hierarchy,
            "paragraph": {"label": compound_label},
        }
        cfr_citation = f"{base_cite}{compound_label}"

        # Build an ID-safe suffix from the compound label: (a)(1)(i) → a-1-i
        label_suffix = re.sub(r"\(|\)", lambda m: "" if m.group() == "(" else "-", compound_label).strip("-")

        overflow_parts = _split_overflow(para_text)
        total = len(overflow_parts)

        for idx, part_text in enumerate(overflow_parts):
            part_num = idx + 1
            suffix = f"para-{label_suffix}" if total == 1 else f"para-{label_suffix}-part{part_num}"
            chunk_id = _make_chunk_id(*id_parts, suffix=suffix)

            overflow_seq: Optional[dict] = None

            if total > 1:
                overflow_seq = {"part": part_num, "total_parts": total}
                if part_num < total:
                    next_id = _make_chunk_id(*id_parts, suffix=f"para-{label_suffix}-part{part_num + 1}")
                    overflow_seq["next_chunk_id"] = next_id
                if part_num > 1:
                    prev_id = _make_chunk_id(*id_parts, suffix=f"para-{label_suffix}-part{part_num - 1}")
                    overflow_seq["prev_chunk_id"] = prev_id

            chunk = _build_chunk(
                chunk_id=chunk_id,
                chunk_type="paragraph",
                cfr_citation=cfr_citation if total == 1 else f"{cfr_citation} [part {part_num} of {total}]",
                hierarchy=para_hierarchy,
                text=part_text,
                section_preamble=preamble,
                doc_meta=doc_meta,
                overflow_sequence=overflow_seq,
            )
            chunks.append(chunk)

    return chunks


def _infer_topic(text: str) -> str:
    """
    Try to extract the inline topic from a labelled paragraph.
    CFR pattern: "(a) TopicName. The rest of the paragraph..."
    """
    # Strip leading label of any type: (a) (1) (ii) (A)
    stripped = re.sub(r"^\s*\([a-zA-Z0-9]+\)\s*", "", text)
    # Take up to the first sentence-ending period followed by whitespace, max 80 chars
    m = re.match(r"^([^.\n—–]{3,80}?)\.\s", stripped)
    if m:
        return m.group(1).strip()
    return stripped[:80].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Section dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _process_section(
    section_el: ET.Element,
    hierarchy: dict,
    doc_meta: dict,
) -> list[dict]:
    """
    Parse a <SECTION> element and return a list of chunk dicts.
    Skips RESERVED sections.
    """
    # Check for RESERVED marker
    if section_el.find("RESERVED") is not None:
        return []

    sectno_el = section_el.find("SECTNO")
    subject_el = section_el.find("SUBJECT")
    cita_el = section_el.find("CITA")

    if sectno_el is None:
        return []

    sectno = re.sub(r"§\s*", "", _get_text(sectno_el)).strip()
    subject = _get_text(subject_el)

    sec_hierarchy = {
        **hierarchy,
        "section": {"number": sectno, "name": subject},
        "paragraph": None,
    }

    # Collect content elements (skip structural/nav tags and already-handled tags)
    content_els = [
        child for child in section_el
        if child.tag not in (
            "SECTNO", "SUBJECT", "CITA", "AUTH", "SOURCE",
            "PRTPAGE", "EAR", "RESERVED",
        )
        and child.tag not in SKIP_TAGS
    ]

    if not content_els:
        return []

    # Build id_parts tuple used by all chunk factories
    h = sec_hierarchy
    id_parts = (
        h["title"]["number"],
        h["chapter"]["number"],
        h.get("subchapter", {}).get("letter") if h.get("subchapter") else None,
        h["part"]["number"],
        h.get("subpart", {}).get("letter") if h.get("subpart") else None,
        sectno.replace(".", "-"),
    )

    if _is_definitions_section(subject):
        return _chunk_definitions_section(content_els, sec_hierarchy, doc_meta, id_parts)

    return _chunk_regular_section(content_els, sec_hierarchy, doc_meta, id_parts)


# ─────────────────────────────────────────────────────────────────────────────
# PART / SUBPART / SUBCHAP traversal
# ─────────────────────────────────────────────────────────────────────────────

def _parse_subpart_heading(hd_text: str) -> tuple[Optional[str], str]:
    """Returns (letter, name) from a Subpart heading like 'Subpart A—General'."""
    m = _SUBPART_HD_RE.match(hd_text)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return None, hd_text.strip()


def _parse_part_heading(hd_text: str) -> tuple[Optional[str], str]:
    """Returns (number, name) from a Part heading like 'PART 1401—PUBLIC AVAILABILITY'."""
    m = _PART_HD_RE.search(hd_text)
    if m:
        return m.group(1), m.group(2).strip().title()
    return None, hd_text.strip()


def _parse_subchap_heading(hd_text: str) -> tuple[Optional[str], str]:
    """Returns (letter, name) from 'SUBCHAPTER A—GENERAL'."""
    m = _SUBCHAP_HD_RE.match(hd_text)
    if m:
        return m.group(1).upper(), m.group(2).strip().title()
    return None, hd_text.strip()


def _process_part(
    part_el: ET.Element,
    base_hierarchy: dict,
    doc_meta: dict,
) -> list[dict]:
    """Process a <PART> element, handling both SUBPART>SECTION and direct SECTION children."""
    # Skip RESERVED parts
    if part_el.find("RESERVED") is not None:
        return []

    hd_el = part_el.find("HD")
    hd_text = _get_text(hd_el)
    part_num, part_name = _parse_part_heading(hd_text)
    if part_num is None:
        # Part with no recognisable number heading — skip
        return []

    # Part-level source citation
    source_el = part_el.find("SOURCE")
    part_source = _extract_text(source_el.find("P")) if source_el is not None and source_el.find("P") is not None else ""

    part_doc_meta = {
        **doc_meta,
    }

    part_hierarchy = {
        **base_hierarchy,
        "part": {"number": part_num, "name": part_name},
        "subpart": None,
        "section": None,
        "paragraph": None,
    }

    chunks: list[dict] = []

    for child in part_el:
        if child.tag in SKIP_TAGS or child.tag in ("HD", "AUTH", "SOURCE", "EAR", "RESERVED", "PRTPAGE"):
            continue

        if child.tag == "SUBPART":
            # SUBPART > SECTION
            subpart_hd = _get_text(child.find("HD"))
            subpart_letter, subpart_name = _parse_subpart_heading(subpart_hd)
            subpart_hierarchy = {
                **part_hierarchy,
                "subpart": {"letter": subpart_letter, "name": subpart_name},
            }
            for section_el in child.iter("SECTION"):
                chunks.extend(_process_section(section_el, subpart_hierarchy, part_doc_meta))

        elif child.tag == "SECTION":
            # Direct SECTION under PART (no subpart)
            chunks.extend(_process_section(child, part_hierarchy, part_doc_meta))

    return chunks


def _process_subchap(
    subchap_el: ET.Element,
    base_hierarchy: dict,
    doc_meta: dict,
) -> list[dict]:
    """Process a <SUBCHAP> element (chapI-style files only)."""
    hd_el = subchap_el.find("HD")
    hd_text = _get_text(hd_el)
    subchap_letter, subchap_name = _parse_subchap_heading(hd_text)

    subchap_hierarchy = {
        **base_hierarchy,
        "subchapter": {"letter": subchap_letter, "name": subchap_name} if subchap_letter else None,
    }

    chunks: list[dict] = []
    for part_el in subchap_el.findall("PART"):
        chunks.extend(_process_part(part_el, subchap_hierarchy, doc_meta))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# FDSYS metadata parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fdsys(root: ET.Element) -> dict:
    """Extract document-level metadata from the <FDSYS> element."""
    fdsys = root.find("FDSYS")
    if fdsys is None:
        return {}
    return {
        "title_number": _get_text(fdsys.find("CFRTITLE")),
        "title_name": _get_text(fdsys.find("CFRTITLETEXT")),
        "volume": _get_text(fdsys.find("VOL")),
        "chapter_number": _get_text(fdsys.find("GRANULENUM")),
        "chapter_name": _get_text(fdsys.find("TITLE")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# File-level processor
# ─────────────────────────────────────────────────────────────────────────────

def process_file(xml_path: str | Path) -> list[dict]:
    """
    Parse a single CFR XML file and return all extracted chunks.
    Handles both chapI (SUBCHAP) and chapIII (no SUBCHAP) structures.
    """
    xml_path = Path(xml_path)
    logger.info("Processing %s", xml_path.name)

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        logger.error("XML parse error in %s: %s", xml_path.name, exc)
        return []

    root = tree.getroot()
    doc_meta = _parse_fdsys(root)
    doc_meta["source_file"] = xml_path.name

    title_num   = doc_meta.get("title_number", "21")
    chapter_num = doc_meta.get("chapter_number", "?")
    chapter_name = doc_meta.get("chapter_name", "")

    base_hierarchy = {
        "title":      {"number": title_num, "name": doc_meta.get("title_name", "")},
        "chapter":    {"number": chapter_num, "name": chapter_name},
        "subchapter": None,
        "part":       None,
        "subpart":    None,
        "section":    None,
        "paragraph":  None,
    }

    chapter_el = root.find("CHAPTER")
    if chapter_el is None:
        logger.warning("No <CHAPTER> found in %s", xml_path.name)
        return []

    chunks: list[dict] = []

    for child in chapter_el:
        if child.tag in SKIP_TAGS or child.tag == "TOC":
            continue
        if child.tag == "SUBCHAP":
            chunks.extend(_process_subchap(child, base_hierarchy, doc_meta))
        elif child.tag == "PART":
            chunks.extend(_process_part(child, base_hierarchy, doc_meta))

    logger.info("  → %d chunks extracted from %s", len(chunks), xml_path.name)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(
    xml_dir: str | Path = "data/cfr_xml",
    output_path: str | Path = "data/chunks/cfr_chunks.json",
    *,
    file_pattern: str = "*.xml",
) -> dict:
    """
    Process every XML file matching *file_pattern* in *xml_dir* and write
    the merged chunk list to *output_path*.

    Returns a summary dict with keys: total_chunks, source_files, output_path.
    """
    xml_dir = Path(xml_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob(file_pattern))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {xml_dir} matching '{file_pattern}'")

    all_chunks: list[dict] = []
    source_files: list[str] = []

    for xml_file in xml_files:
        file_chunks = process_file(xml_file)
        all_chunks.extend(file_chunks)
        source_files.append(xml_file.name)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(all_chunks),
        "source_files": source_files,
        "chunks": all_chunks,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Extraction complete: %d chunks from %d files → %s",
        len(all_chunks),
        len(source_files),
        output_path,
    )
    return {
        "total_chunks": len(all_chunks),
        "source_files": source_files,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    result = extract_all()
    print(f"\nDone — {result['total_chunks']:,} chunks written to {result['output_path']}")
