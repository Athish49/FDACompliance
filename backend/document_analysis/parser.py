"""
Document parser — extracts plain text from PDF, DOCX, or plain-text uploads.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

# Maximum characters to process (avoids runaway LLM costs on huge files)
MAX_CHARS = 50_000


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded file bytes.

    Supports:
      - PDF  (.pdf)  via PyMuPDF
      - DOCX (.docx) via python-docx
      - Plain text   (everything else — UTF-8 with latin-1 fallback)

    Returns at most MAX_CHARS characters (truncates with a notice).
    Raises ValueError if the file appears empty after extraction.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        text = _extract_pdf(file_bytes)
    elif ext == "docx":
        text = _extract_docx(file_bytes)
    else:
        text = _extract_plain(file_bytes)

    text = text.strip()
    if not text:
        raise ValueError(f"No text could be extracted from '{filename}'")

    if len(text) > MAX_CHARS:
        logger.warning(
            "Document '%s' truncated from %d to %d chars", filename, len(text), MAX_CHARS
        )
        text = text[:MAX_CHARS] + "\n\n[Document truncated for analysis]"

    return text


def _extract_pdf(file_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF parsing: pip install pymupdf") from exc

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX parsing: pip install python-docx"
        ) from exc

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _extract_plain(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")
