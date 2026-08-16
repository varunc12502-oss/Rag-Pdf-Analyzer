"""
retriever.py — Semantic Similarity Search and Relevance Filtering
=================================================================
Step 3 of the RAG pipeline: user question → relevant chunks.

Given a natural-language question, we embed it with the same model used
during ingestion, then search the vector store for the closest chunks.

Key design decision — relevance thresholding:
----------------------------------------------
A naive RAG system passes whatever chunks it finds to the LLM, even if
they have nothing to do with the question. The LLM then tends to
hallucinate an answer using its parametric knowledge instead of the
document.

We prevent this by checking the cosine distance of every retrieved chunk
against a configurable threshold. If all chunks are too distant from the
query vector, we return an empty list — signalling to the rest of the
pipeline that the document doesn't contain an answer.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    collection: Any,
    query: str,
    top_k: int = 5,
    distance_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Find the top-k most semantically similar chunks to the query,
    filtered by a cosine distance threshold.

    How it works:
    1. ChromaDB embeds the query using the collection's embedding function.
    2. An HNSW index search finds the approximate nearest neighbours in
       the high-dimensional embedding space.
    3. Results come back sorted by cosine distance (ascending = most similar first).
    4. We discard any chunk whose distance exceeds `distance_threshold`.

    Cosine distance interpretation:
        0.0 – 0.3  → highly relevant
        0.3 – 0.6  → somewhat relevant
        0.6 – 1.0  → likely irrelevant

    Args:
        collection         — ChromaDB collection to search.
        query              — user's question as plain text.
        top_k              — maximum number of chunks to consider.
        distance_threshold — chunks with distance > this are discarded.

    Returns:
        A filtered list of result dicts:
            {
                "text":     str,   # chunk content
                "source":   str,   # original filename
                "page":     int,   # page number in source PDF
                "distance": float, # cosine distance (lower = more relevant)
            }
        Returns [] if no chunks pass the relevance filter.
    """
    total_stored = collection.count()
    if total_stored == 0:
        logger.warning("Retrieval called on an empty collection.")
        return []

    # Clamp top_k to avoid requesting more results than are stored
    effective_top_k = min(top_k, total_stored)

    try:
        # ChromaDB handles query embedding internally using the same
        # SentenceTransformerEmbeddingFunction set up in embed_store.py
        results = collection.query(
            query_texts=[query],
            n_results=effective_top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error("ChromaDB query failed: %s", e)
        return []

    # ChromaDB returns results as nested lists (one sublist per query).
    # Since we always query one string at a time, we take index [0].
    raw_docs = results.get("documents", [[]])[0]
    raw_meta = results.get("metadatas", [[]])[0]
    raw_dist = results.get("distances", [[]])[0]

    relevant: List[Dict[str, Any]] = []
    for text, meta, distance in zip(raw_docs, raw_meta, raw_dist):
        if distance > distance_threshold:
            logger.debug(
                "Filtered out chunk from '%s' p%s (distance=%.3f > threshold=%.3f).",
                meta.get("source"), meta.get("page"), distance, distance_threshold,
            )
            continue  # too distant → not relevant enough

        relevant.append({
            "text":     text,
            "source":   meta.get("source", "unknown"),
            "page":     meta.get("page", "?"),
            "distance": round(distance, 4),
        })

    logger.info(
        "Retrieved %d relevant chunk(s) from %d candidate(s) for query: '%s...'",
        len(relevant), len(raw_docs), query[:60],
    )
    return relevant
