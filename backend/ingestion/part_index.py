"""CFR Part Index Generator.

Reads the chunks JSON produced by the extractor and writes a compact
cfr_part_index.json to backend/data/. This file is used by the CFR
Part classifier at query time to give the LLM accurate knowledge of
which Part numbers exist and what topic each covers — without relying
on the LLM's pretrained weights, which may be wrong or outdated.

Called automatically by the ingestion pipeline after extraction.
Safe to call multiple times (idempotent — overwrites in place).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PART_INDEX_PATH = Path(__file__).parent.parent / "data" / "cfr_part_index.json"


def generate_part_index(chunks_path: str | Path) -> dict:
    """
    Extract unique CFR parts from the chunks JSON and write the index file.

    Returns the index dict (same structure as what is written to disk):
    {
      "generated_at": "<ISO timestamp>",
      "total_parts": <int>,
      "parts": {
        "<part_number>": {
          "title": "<Part title>",
          "subchapter": "<letter>",
          "subchapter_name": "<subchapter title>"
        },
        ...
      }
    }
    """
    chunks_path = Path(chunks_path)
    logger.info("[part_index] Reading chunks from %s", chunks_path)

    with open(chunks_path, encoding="utf-8") as fh:
        data = json.load(fh)

    chunks = data.get("chunks", [])
    parts: dict[str, dict] = {}

    for chunk in chunks:
        h = chunk.get("hierarchy", {}) or {}
        part_node = h.get("part", {}) or {}
        num = str(part_node.get("number", "")).strip()
        title = str(part_node.get("name", "")).strip()
        subchap_node = h.get("subchapter", {}) or {}
        subchap_letter = str(subchap_node.get("letter", "")).strip()
        subchap_name = str(subchap_node.get("name", "")).strip()

        if num and num not in parts:
            parts[num] = {
                "title": title,
                "subchapter": subchap_letter,
                "subchapter_name": subchap_name,
            }

    sorted_parts = dict(
        sorted(parts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 9999)
    )

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_parts": len(sorted_parts),
        "parts": sorted_parts,
    }

    PART_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PART_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    logger.info(
        "[part_index] Wrote %d CFR parts → %s",
        len(sorted_parts),
        PART_INDEX_PATH,
    )
    return index
