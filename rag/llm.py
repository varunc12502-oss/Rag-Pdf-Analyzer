"""
llm.py — OpenRouter LLM API Wrapper
=====================================
Step 4 of the RAG pipeline: context + question → grounded answer.

We use the official OpenAI Python SDK, but pointed at OpenRouter's
API endpoint. OpenRouter is API-compatible with OpenAI and provides
unified access to hundreds of LLMs — including free-tier models — via
a single key and endpoint.

The prompt structure follows the standard RAG pattern:
  - System message: restricts the model to answering from the given context.
  - User message: the retrieved context chunks + the user's question.

This two-message structure is the primary mechanism that keeps answers
grounded in the document rather than the model's training data.
"""

import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI, RateLimitError, APIError, APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)

# Default free-tier model on OpenRouter.
# The ":free" suffix indicates models that don't consume OpenRouter credits.
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

MAX_RESPONSE_TOKENS = 1024


def get_llm_client(api_key: str) -> OpenAI:
    """
    Initialise and return an OpenAI client pointed at OpenRouter's endpoint.

    The OpenAI SDK is model-agnostic: by changing `base_url` we can use
    any OpenAI-API-compatible backend without touching the rest of the code.
    """
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _build_context_string(chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved chunks into a labelled context block.

    Each chunk is prefixed with a source label ([Source N]) so the LLM
    can cite it in its answer and we can display it to the user.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    client: OpenAI,
    question: str,
    context_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate an answer grounded in the retrieved document chunks.

    The system prompt is the critical anti-hallucination guard: it
    explicitly instructs the model to answer only from the context
    provided and to admit uncertainty if the context is insufficient.

    Temperature is set low (0.2) to keep answers factual and consistent
    rather than creative.

    Args:
        client         — initialised OpenAI/OpenRouter client.
        question       — the user's original question.
        context_chunks — relevant chunks from the retriever.
        model          — OpenRouter model identifier.

    Returns:
        The LLM's answer as a plain string.

    Raises:
        RuntimeError — on any API failure, with a user-friendly message.
    """
    context_str = _build_context_string(context_chunks)

    system_prompt = (
        "You are a precise, helpful assistant that answers questions based "
        "strictly on the document excerpts provided.\n\n"
        "Rules you must follow:\n"
        "1. Answer ONLY using information from the context below. "
        "Do NOT use outside knowledge or training data.\n"
        "2. If the context does not contain enough information to fully "
        "answer the question, say so honestly.\n"
        "3. When relevant, cite the source label (e.g., [Source 1]) "
        "to indicate which excerpt supports your answer.\n"
        "4. Do NOT fabricate quotes, statistics, or facts not present in the context.\n"
        "5. Keep your answer concise and well-structured."
    )

    user_message = (
        f"Document context:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("LLM response received (%d chars).", len(answer))
        return answer

    except RateLimitError:
        raise RuntimeError(
            "Rate limit reached on OpenRouter. Please wait a moment and try again."
        )
    except APIConnectionError:
        raise RuntimeError(
            "Could not reach the OpenRouter API. Please check your internet connection."
        )
    except APIStatusError as e:
        raise RuntimeError(
            f"OpenRouter returned an error (HTTP {e.status_code}): {e.message}"
        )
    except APIError as e:
        raise RuntimeError(f"OpenRouter API error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error calling the LLM: {e}")


def reformulate_query_with_llm(
    client: OpenAI,
    original_query: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Ask the LLM to rewrite the user's query to improve retrieval.

    When the first retrieval attempt returns no relevant chunks, we call
    this function to expand/clarify the query before retrying. The LLM
    can resolve abbreviations, add synonyms, and make the intent more
    explicit — all of which can help surface relevant chunks that a
    short or ambiguous query missed.

    This technique is sometimes called query expansion or HyDE-lite in
    the RAG literature.

    Returns:
        A reformulated query string, or the original if reformulation fails.
    """
    prompt = (
        "You are a search query optimizer. Rewrite the following question to "
        "make it more explicit and retrieval-friendly for a vector database search.\n\n"
        "Guidelines:\n"
        "- Expand abbreviations and acronyms.\n"
        "- Add relevant synonyms or related terms.\n"
        "- Make the intent of the question more specific.\n"
        "- Output ONLY the rewritten question — no explanation, no preamble.\n\n"
        f"Original question: {original_query}\n\n"
        "Rewritten question:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.3,
        )
        reformulated = response.choices[0].message.content.strip()
        logger.info(
            "Query reformulated: '%s' → '%s'", original_query, reformulated
        )
        return reformulated
    except Exception as e:
        logger.warning("Query reformulation failed (%s). Using original query.", e)
        return original_query
