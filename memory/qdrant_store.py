"""
memory/qdrant_store.py
──────────────────────
Qdrant vector store client for episodic memory.

Two collections:
  - episodic_memory : stores summaries of past research sessions
  - knowledge_cache : stores chunked content from scraped pages

The EpisodicMemoryStore class wraps raw qdrant-client operations to give
agents a clean, high-level interface:
  - store_episode(session_id, query, report_summary)
  - recall_episodes(query, top_k) → list[str]
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import get_settings
from memory.embeddings import embed_text, embed_texts

logger = structlog.get_logger(__name__)

# ─── Collection configuration ─────────────────────────────────────────────────

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension
DISTANCE = qmodels.Distance.COSINE


# ─── Client factory ──────────────────────────────────────────────────────────

def _get_sync_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _get_async_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


# ─── Collection initialisation ────────────────────────────────────────────────

def ensure_collections_exist(client: Optional[QdrantClient] = None) -> None:
    """
    Idempotently create Qdrant collections if they don't exist.
    Called from scripts/init_qdrant.py and on API startup.
    """
    settings = get_settings()
    c = client or _get_sync_client()

    for collection_name in [
        settings.qdrant_collection_episodic,
        settings.qdrant_collection_knowledge,
    ]:
        try:
            c.get_collection(collection_name)
            logger.debug("qdrant_collection_exists", name=collection_name)
        except (UnexpectedResponse, Exception):
            c.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_DIM,
                    distance=DISTANCE,
                    on_disk=False,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=100,  # Build HNSW index after 100 vectors
                ),
            )
            logger.info("qdrant_collection_created", name=collection_name)


# ─── Episodic Memory Store ────────────────────────────────────────────────────

class EpisodicMemoryStore:
    """
    High-level interface for storing and recalling research episodes.

    An "episode" is a structured snapshot of one research session:
        - The original user query
        - A summary of the final report
        - Metadata: session_id, timestamp, report length
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._collection = self._settings.qdrant_collection_episodic

    # ── Synchronous API ──────────────────────────────────────────────────────

    def store_episode(
        self,
        session_id: str,
        query: str,
        report_summary: str,
    ) -> str:
        """
        Persist a research episode to Qdrant.

        We embed a concatenation of the query + summary so that future
        similarity searches match on topic, not just surface keywords.
        """
        text_to_embed = f"Query: {query}\n\nSummary: {report_summary}"
        vector = embed_text(text_to_embed)

        point_id = str(uuid.uuid4())
        client = _get_sync_client()

        client.upsert(
            collection_name=self._collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "query": query,
                        "report_summary": report_summary,
                        "stored_at": datetime.now(timezone.utc).isoformat(),
                        "report_length": len(report_summary),
                    },
                )
            ],
        )
        logger.info("episode_stored", session_id=session_id, point_id=point_id)
        return point_id

    def recall_episodes(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.55,
    ) -> list[str]:
        """
        Retrieve the most semantically similar past episodes.

        Returns a list of strings (episode summaries) ready to be injected
        into an agent's context window.

        Args:
            query: The current research question.
            top_k: Maximum number of episodes to return.
            score_threshold: Minimum cosine similarity (0-1) to qualify.
                             0.55 avoids returning completely unrelated episodes.
        """
        vector = embed_text(query)
        client = _get_sync_client()

        try:
            hits = client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("qdrant_recall_failed", error=str(exc))
            return []

        memories: list[str] = []
        for hit in hits:
            payload = hit.payload or {}
            memory_text = (
                f"[Past Research — score: {hit.score:.2f}]\n"
                f"Query: {payload.get('query', 'unknown')}\n"
                f"Summary: {payload.get('report_summary', '')}"
            )
            memories.append(memory_text)

        logger.debug("episodes_recalled", query=query, count=len(memories))
        return memories

    # ── Async API ────────────────────────────────────────────────────────────

    async def astore_episode(
        self,
        session_id: str,
        query: str,
        report_summary: str,
    ) -> str:
        """Async version of store_episode."""
        text_to_embed = f"Query: {query}\n\nSummary: {report_summary}"
        vector = embed_text(text_to_embed)  # embed_text is fast enough synchronously

        point_id = str(uuid.uuid4())
        client = _get_async_client()

        await client.upsert(
            collection_name=self._collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "query": query,
                        "report_summary": report_summary,
                        "stored_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )
        logger.info("episode_stored_async", session_id=session_id, point_id=point_id)
        return point_id

    async def arecall_episodes(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.55,
    ) -> list[str]:
        """Async version of recall_episodes."""
        vector = embed_text(query)
        client = _get_async_client()

        try:
            hits = await client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("qdrant_arecall_failed", error=str(exc))
            return []

        memories: list[str] = []
        for hit in hits:
            payload = hit.payload or {}
            memory_text = (
                f"[Past Research — score: {hit.score:.2f}]\n"
                f"Query: {payload.get('query', 'unknown')}\n"
                f"Summary: {payload.get('report_summary', '')}"
            )
            memories.append(memory_text)

        return memories
