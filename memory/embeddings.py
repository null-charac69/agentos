"""
memory/embeddings.py
────────────────────
Local embedding provider using sentence-transformers.

Why local instead of OpenAI embeddings?
  - Zero API cost — model runs on CPU (or GPU if available)
  - No network latency for each embed call
  - `all-MiniLM-L6-v2` gives excellent quality at 384 dimensions
  - Downloads once, cached in ~/.cache/torch/sentence_transformers

The model is lazy-loaded and cached so the first embed call pays the
load cost; subsequent calls are ~50ms on CPU.
"""

from __future__ import annotations

from functools import lru_cache

import structlog

logger = structlog.get_logger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the SentenceTransformer model (cached singleton)."""
    from sentence_transformers import SentenceTransformer

    logger.info("loading_embedding_model", model=_MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME)
    logger.info("embedding_model_loaded", dimension=model.get_sentence_embedding_dimension())
    return model


def embed_text(text: str) -> list[float]:
    """
    Embed a single string. Returns a list[float] of length 384.
    """
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch embed multiple strings. More efficient than calling embed_text
    in a loop because sentence-transformers batches internally.
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


@property
def embedding_dimension() -> int:
    """Returns 384 for all-MiniLM-L6-v2."""
    return _get_model().get_sentence_embedding_dimension()
