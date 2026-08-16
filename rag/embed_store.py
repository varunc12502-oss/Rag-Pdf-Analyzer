"""
embed_store.py — Embedding and Vector Store Management
=======================================================
Step 2 of the RAG pipeline: text chunks → embeddings → ChromaDB.

We use sentence-transformers (all-MiniLM-L6-v2) to convert text into
dense numeric vectors (embeddings) that capture semantic meaning.
ChromaDB stores these vectors on disk for fast similarity search.

Why local embeddings instead of an API?
----------------------------------------
Running the embedding model locally (via sentence-transformers) means:
  - No API calls, no cost, no rate limits for this step.
  - Embeddings are computed on the fly, in memory.
  - The model is ~90MB and downloads once, then is cached locally.
"""

import logging
import uuid
from typing import List, Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 is a 22M-parameter model optimised for sentence
# similarity tasks. It's fast, lightweight, and accurate enough for
# most document Q&A use cases.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """
    Returns a ChromaDB-compatible embedding function backed by
    SentenceTransformers. The model is downloaded once and cached
    to disk by the sentence-transformers library (~90 MB).
    """
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_chroma_collection(
    persist_dir: str,
    collection_name: str,
) -> Any:
    """
    Get (or create) a ChromaDB collection configured for cosine similarity.

    What is ChromaDB?
    -----------------
    ChromaDB is an open-source vector database. It stores text chunks
    alongside their embeddings and provides fast approximate nearest-
    neighbour (ANN) search via an HNSW index on disk.

    Why cosine distance?
    --------------------
    Cosine distance measures the angle between two vectors, not their
    magnitude. This makes it robust to text length differences — a short
    chunk and a long chunk about the same topic will still score as
    similar. Distance 0 = identical; 1 = completely unrelated.

    Args:
        persist_dir     — directory where ChromaDB stores index files.
        collection_name — unique name identifying this session's collection.

    Returns:
        A ChromaDB Collection object ready for add/query operations.
    """
    embedding_fn = get_embedding_function()

    # PersistentClient writes data to disk automatically after each write.
    # This means embeddings survive a Streamlit rerun within the same session.
    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # use cosine distance metric
    )
    logger.info(
        "ChromaDB collection '%s' ready (path: %s).", collection_name, persist_dir
    )
    return collection


def add_documents_to_store(
    collection: Any,
    documents: List[Any],
) -> int:
    """
    Embed and store a list of LangChain Document objects in ChromaDB.

    ChromaDB will automatically compute embeddings for each chunk using
    the collection's embedding function, then store both the text and
    the resulting vector.

    We generate unique IDs per chunk using a combination of filename,
    page number, and a random suffix — this prevents ID collisions when
    the same PDF is re-uploaded or when multiple files share page numbers.

    Args:
        collection — the ChromaDB Collection to write to.
        documents  — list of LangChain Document objects (from chunk_pages).

    Returns:
        Number of chunks successfully stored.
    """
    if not documents:
        logger.warning("add_documents_to_store called with an empty document list.")
        return 0

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    ids = [
        "{src}_p{pg}_{uid}".format(
            src=doc.metadata.get("source", "doc").replace(" ", "_")[:30],
            pg=doc.metadata.get("page", 0),
            uid=uuid.uuid4().hex[:8],
        )
        for doc in documents
    ]

    # ChromaDB handles the embedding computation internally here.
    # Under the hood it calls our SentenceTransformerEmbeddingFunction
    # on each text in `texts`, then indexes the resulting vectors.
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info("Stored %d chunks in ChromaDB collection.", len(documents))
    return len(documents)


def clear_collection(persist_dir: str, collection_name: str) -> None:
    """
    Delete the ChromaDB collection entirely, removing all stored chunks
    and their embeddings.

    Called when the user clicks "Clear Documents" in the sidebar.
    """
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(name=collection_name)
        logger.info("Deleted ChromaDB collection '%s'.", collection_name)
    except Exception as e:
        logger.debug("Could not delete collection '%s': %s", collection_name, e)


def get_collection_count(collection: Any) -> int:
    """Returns the total number of chunks currently stored in the collection."""
    try:
        return collection.count()
    except Exception:
        return 0
