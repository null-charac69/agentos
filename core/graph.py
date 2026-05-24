r"""
core/graph.py
─────────────
LangGraph StateGraph builder — wires all five agent nodes into the
autonomous research pipeline.

Graph topology:
                    ┌─────────────────────────────────────────────┐
                    │              LangGraph StateGraph             │
                    │                                              │
  START ──► retrieval_recall ──► planner ──► web_search           │
                                                   │               │
                                           code_executor           │
                                                   │               │
                                            synthesis              │
                                                   │               │
                                              critic               │
                                            /         \            │
                                    [approved]    [rejected &       │
                                        │          retries < MAX]  │
                                        ▼               │          │
                              retrieval_store         planner ◄────┘
                                        │            (retry)
                                       END

Conditional routing:
  - critic → END             : critique.approved == True
  - critic → planner (retry) : not approved AND retry_count < MAX_RETRIES
  - critic → END (forced)    : not approved AND retry_count >= MAX_RETRIES
                               (uses the last synthesis as final_report)
"""

from __future__ import annotations

from typing import Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.code_executor import code_executor_node
from agents.critic import critic_node, synthesis_node
from agents.planner import planner_node
from agents.retrieval import retrieval_recall_node, retrieval_store_node
from agents.web_search import web_search_node
from config.settings import get_settings
from core.state import AgentState

logger = structlog.get_logger(__name__)


# ─── Conditional edge logic ───────────────────────────────────────────────────

def _route_after_critic(state: AgentState) -> Literal["planner", "retrieval_store", "__end__"]:
    """
    Determines the next node after the critic evaluates the synthesis.

    Returns:
        "retrieval_store" → critique passed; store the episode and finish
        "planner"         → critique failed; retry with feedback
        "__end__"         → max retries exhausted; deliver best effort
    """
    settings = get_settings()
    max_retries = settings.max_critic_retries

    critique = state.get("critique", {})
    retry_count = state.get("retry_count", 0)

    if critique.get("approved", False):
        logger.info("critic_approved_routing_to_store")
        return "retrieval_store"

    if retry_count < max_retries:
        logger.info(
            "critic_rejected_routing_to_retry",
            retry=retry_count,
            max=max_retries,
        )
        return "planner"

    # Max retries hit — force-finish with whatever synthesis we have
    logger.warning(
        "max_retries_exhausted_force_finishing",
        retry_count=retry_count,
    )
    return "__end__"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the AgentOS research graph.

    Uses MemorySaver for in-process checkpointing. To scale horizontally,
    replace MemorySaver with langgraph_checkpoint_redis.RedisSaver:

        from langgraph_checkpoint_redis import RedisSaver
        checkpointer = RedisSaver.from_conn_string(settings.redis_url)

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("retrieval_recall", retrieval_recall_node)
    builder.add_node("planner", planner_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("critic", critic_node)
    builder.add_node("retrieval_store", retrieval_store_node)

    # ── Define edges (linear flow) ────────────────────────────────────────────
    builder.add_edge(START, "retrieval_recall")
    builder.add_edge("retrieval_recall", "planner")
    builder.add_edge("planner", "web_search")
    builder.add_edge("web_search", "code_executor")
    builder.add_edge("code_executor", "synthesis")
    builder.add_edge("synthesis", "critic")

    # ── Conditional edge: critic routes to store, retry, or forced end ────────
    builder.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "retrieval_store": "retrieval_store",
            "planner": "planner",          # retry loop
            "__end__": END,
        },
    )

    # ── After storing, we're done ─────────────────────────────────────────────
    builder.add_edge("retrieval_store", END)

    # ── Compile with in-process checkpointer ──────────────────────────────────
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("graph_compiled")
    return graph


# ── Singleton graph instance ──────────────────────────────────────────────────
# Imported by the API routes — one graph object shared across all requests.
research_graph = build_graph()
