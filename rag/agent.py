"""
agent.py — Agentic Query Retry Logic
======================================
This module adds a simple "reflect and retry" loop on top of the
standard retrieve → answer pipeline.

Why do we need this?
---------------------
A user's raw question is not always optimal for vector search. Short
questions ("What are the main findings?"), unusual terminology, or
queries that use different phrasing than the document may return zero
relevant chunks even when the answer exists in the PDF.

The agentic loop here:
    1. Try retrieval with the original user query.
    2. If no relevant chunks are found, ask the LLM to reformulate
       the query — expanding it with synonyms, making the intent more
       explicit.
    3. Retry retrieval once with the new query.
    4. If the second attempt also fails, return an empty list so the
       caller can respond with "I don't know" rather than hallucinating.

This is a minimal but instructive implementation of an agentic pattern:
the system observes a failure, reasons about how to improve its next
action, and retries — rather than blindly handing off bad context to
the LLM.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI

from rag.retriever import retrieve_relevant_chunks
from rag.llm import reformulate_query_with_llm

logger = logging.getLogger(__name__)


def retrieve_with_fallback(
    collection: Any,
    client: OpenAI,
    query: str,
    top_k: int = 5,
    distance_threshold: float = 0.6,
    model: str = "meta-llama/llama-3.1-8b-instruct:free",
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """
    Retrieve relevant chunks for a query, with one automatic retry on failure.

    This wraps retrieve_relevant_chunks with a query-reformulation fallback:
      - Attempt 1: search with the user's original query.
      - If attempt 1 returns no relevant chunks:
          - Ask the LLM to rewrite the query.
          - Attempt 2: search with the reformulated query.

    Args:
        collection         — ChromaDB collection to search.
        client             — initialised OpenAI/OpenRouter client.
        query              — original user question.
        top_k              — max number of chunks to retrieve.
        distance_threshold — relevance cutoff (cosine distance).
        model              — model to use for query reformulation.

    Returns:
        A 3-tuple:
            chunks             (list)          — relevant chunk dicts (may be empty).
            was_reformulated   (bool)          — True if the retry path was used.
            reformulated_query (str | None)    — the rewritten query, or None.
    """
    # ── Attempt 1: original query ──────────────────────────────────────────
    logger.info("[Agent] Attempt 1 — original query: '%s'", query)
    chunks = retrieve_relevant_chunks(
        collection=collection,
        query=query,
        top_k=top_k,
        distance_threshold=distance_threshold,
    )

    if chunks:
        # Retrieval succeeded on the first try — nothing more to do.
        return chunks, False, None

    # ── Attempt 1 failed: no chunks passed the relevance threshold ─────────
    logger.info(
        "[Agent] No relevant chunks found for original query. "
        "Reformulating and retrying..."
    )

    reformulated_query = reformulate_query_with_llm(
        client=client,
        original_query=query,
        model=model,
    )

    if reformulated_query.strip().lower() == query.strip().lower():
        # The LLM produced the same query — retrying would be pointless.
        logger.info(
            "[Agent] Reformulation returned identical query. Giving up."
        )
        return [], True, reformulated_query

    # ── Attempt 2: reformulated query ─────────────────────────────────────
    logger.info("[Agent] Attempt 2 — reformulated query: '%s'", reformulated_query)
    chunks = retrieve_relevant_chunks(
        collection=collection,
        query=reformulated_query,
        top_k=top_k,
        distance_threshold=distance_threshold,
    )

    # Return whatever we found (may still be empty if the document truly
    # doesn't contain the answer — the caller handles that gracefully).
    return chunks, True, reformulated_query
