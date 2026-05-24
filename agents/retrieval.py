"""
agents/retrieval.py
────────────────────
Retrieval agent — two responsibilities in one node:

1. RECALL: At the start of a session, retrieve semantically similar past
   research episodes from Qdrant. This gives the agent "memory" of what
   it has researched before.

2. STORE: After the final report is produced (post-critic approval),
   store a summary of this session so future queries can recall it.

Why combine recall + store in one node?
  - They share the same Qdrant client and embedding model
  - The graph calls this node twice: once before synthesis (RECALL mode)
    and once after critic approval (STORE mode)
  - A `mode` field in state disambiguates the two calls
"""

from __future__ import annotations

from typing import Any

import structlog

from config.settings import get_settings
from core.state import AgentState
from memory.qdrant_store import EpisodicMemoryStore

logger = structlog.get_logger(__name__)

_store = EpisodicMemoryStore()


async def retrieval_recall_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Retrieval (Recall phase).

    Searches Qdrant for relevant past research sessions and injects
    them into state as `retrieved_memories`.

    Reads:  state["query"]
    Writes: state["retrieved_memories"]
    """
    query = state["query"]
    settings = get_settings()
    top_k = settings.retrieval_top_k

    logger.info("retrieval_recall_started", query=query[:80])

    memories = await _store.arecall_episodes(query=query, top_k=top_k)

    if memories:
        logger.info("memories_recalled", count=len(memories))
    else:
        logger.debug("no_relevant_memories_found")

    return {"retrieved_memories": memories}


async def retrieval_store_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Retrieval (Store phase).

    Persists the current session's report to Qdrant so it can be recalled
    in future research sessions.

    Called only when the critic has approved the final report.

    Reads:  state["query"], state["session_id"], state["final_report"]
    Writes: nothing (side-effect only; state unchanged)
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "unknown")
    final_report = state.get("final_report", "")

    if not final_report:
        logger.warning("retrieval_store_skipped_no_report")
        return {}

    # Store a condensed summary (first 1000 chars) to keep memory vectors compact
    report_summary = final_report[:1000].strip()
    if len(final_report) > 1000:
        report_summary += "…"

    logger.info("retrieval_store_started", session_id=session_id)

    try:
        await _store.astore_episode(
            session_id=session_id,
            query=query,
            report_summary=report_summary,
        )
        logger.info("episode_stored_successfully", session_id=session_id)
    except Exception as exc:
        # Storage failure should never block the final response
        logger.error("retrieval_store_failed", error=str(exc))

    return {}
