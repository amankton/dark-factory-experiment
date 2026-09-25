"""
Embeddings service — wraps the provider's (OpenRouter or Gemini) embeddings API
via the openai SDK.

Exposes:
  embed_text(text: str) -> list[float]
  embed_batch(texts: list[str]) -> list[list[float]]

Uses config.EMBEDDING_MODEL (dimensionality: 1536)
"""

from __future__ import annotations

import logging

from openai import Omit, OpenAI, omit

from backend.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_PROVIDER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client (module-level singleton — re-used across calls)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
    return _client


def _dimensions() -> int | Omit:
    """Gemini's embedding model defaults to 3072 dims; request 1536 to match."""
    if LLM_PROVIDER == "gemini":
        return EMBEDDING_DIMENSIONS
    return omit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string via the configured provider.

    Args:
        text: A non-empty string to embed.

    Returns:
        A list of 1536 floats representing the embedding vector.

    Raises:
        ValueError: If *text* is empty or whitespace-only.
        Exception: If the embeddings API call fails.
    """
    if not text or not text.strip():
        raise ValueError(
            "embed_text() requires a non-empty string; got an empty or whitespace-only value."
        )

    client = _get_client()
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=_dimensions(),
        )
    except Exception as exc:
        logger.error("Embeddings API call failed: %s", exc)
        raise RuntimeError(f"Embeddings API request failed: {exc}") from exc

    embedding = response.data[0].embedding
    return list(embedding)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings via the configured provider in a single batched API call.

    Args:
        texts: A list of strings. May be empty (returns [] immediately).

    Returns:
        A list of embedding vectors, one per input text, in the same order.
        Returns [] if *texts* is empty without making any API call.

    Raises:
        ValueError: If any text in the list is empty or whitespace-only.
        Exception: If the embeddings API call fails.
    """
    if not texts:
        return []

    # Validate all texts before making the API call
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(
                f"embed_batch() requires all texts to be non-empty; got empty string at index {i}."
            )

    client = _get_client()
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=_dimensions(),
        )
    except Exception as exc:
        logger.error("Embeddings batch API call failed: %s", exc)
        raise RuntimeError(f"Embeddings batch API request failed: {exc}") from exc

    # The API guarantees results in the same order as inputs
    # but we sort by index just to be safe. Gemini omits index (None), in
    # which case the input order is kept.
    data = response.data
    if all(d.index is not None for d in data):
        data = sorted(data, key=lambda d: d.index)
    return [list(d.embedding) for d in data]
