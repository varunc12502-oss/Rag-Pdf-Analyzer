"""
ingest.py — PDF Extraction and Text Chunking
============================================
Step 1 of the RAG pipeline: raw PDF → text chunks with metadata.

We use pdfplumber as the primary extractor (it handles complex layouts,
tables, and multi-column PDFs better than most alternatives). If
pdfplumber fails for a particular file, we fall back to PyPDF2.

Each chunk carries metadata (source filename + page number) so we can
cite the exact location when answering.
"""

import io
import logging
from typing import List, Dict, Any

import pdfplumber
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def extract_pages(uploaded_file) -> List[Dict[str, Any]]:
    """
    Extract raw text from every page of a PDF file object (Streamlit UploadedFile).

    Returns a list of dicts, one per page:
        {
            "text":   str,   # raw page text
            "page":   int,   # 1-indexed page number
            "source": str,   # original filename
        }

    Raises:
        ValueError  — if the file is empty or completely un-parseable.
    """
    filename = uploaded_file.name
    raw_bytes = uploaded_file.read()

    if not raw_bytes:
        raise ValueError(f"The uploaded file '{filename}' appears to be empty.")

    pages: List[Dict[str, Any]] = []

    # ── Primary extractor: pdfplumber ──────────────────────────────────────
    # pdfplumber wraps pdfminer.six and produces cleaner text for documents
    # with complex layouts (two-column academic papers, styled reports, etc.)
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({
                    "text":   text.strip(),
                    "page":   page_num,
                    "source": filename,
                })
        logger.info("pdfplumber extracted %d pages from '%s'.", len(pages), filename)

    except Exception as primary_err:
        # ── Fallback extractor: PyPDF2 ────────────────────────────────────
        # PyPDF2 handles some edge-case PDFs that pdfplumber cannot open.
        logger.warning(
            "pdfplumber failed for '%s' (%s). Falling back to PyPDF2.",
            filename, primary_err,
        )
        pages = []
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({
                    "text":   text.strip(),
                    "page":   page_num,
                    "source": filename,
                })
            logger.info("PyPDF2 extracted %d pages from '%s'.", len(pages), filename)
        except Exception as fallback_err:
            raise ValueError(
                f"Could not parse '{filename}'. "
                f"pdfplumber error: {primary_err}. "
                f"PyPDF2 error: {fallback_err}."
            ) from fallback_err

    # Guard: at least one page must have extractable text.
    # Scanned/image-only PDFs will pass parsing but have no text content.
    non_empty_pages = [p for p in pages if p["text"]]
    if not non_empty_pages:
        raise ValueError(
            f"'{filename}' was parsed successfully but contained no extractable text. "
            "It may be a scanned or image-only PDF (OCR not supported)."
        )

    return pages


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Any]:
    """
    Split each page's text into small, overlapping chunks.

    Why chunk at all?
    -----------------
    LLMs have a limited context window. Rather than feeding the entire
    document, we split it into small segments and retrieve only the most
    relevant ones at query time. This keeps prompts focused and short.

    Why RecursiveCharacterTextSplitter?
    ------------------------------------
    It tries to break on natural boundaries in order of preference:
        paragraph break → newline → sentence end → space → character
    This preserves semantic coherence better than a hard character split.

    Why overlap?
    ------------
    Overlap ensures that context at a chunk boundary is not lost — a
    sentence split across two chunks will still appear fully in one of them.

    Args:
        pages        — list of page dicts from extract_pages().
        chunk_size   — target size in characters. ~700 chars ≈ ~175 tokens,
                       well within all-MiniLM-L6-v2's 256-token limit.
        chunk_overlap — characters shared between consecutive chunks.

    Returns:
        List of LangChain Document objects, each with:
            .page_content — the chunk text
            .metadata     — {"source": filename, "page": page_number}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for page in pages:
        if not page["text"]:
            continue  # skip pages with no text content

        chunks = splitter.create_documents(
            texts=[page["text"]],
            metadatas=[{"source": page["source"], "page": page["page"]}],
        )
        all_chunks.extend(chunks)

    logger.info(
        "Created %d chunks from %d non-empty pages.", len(all_chunks), len(pages)
    )
    return all_chunks
