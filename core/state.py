"""
core/state.py
─────────────
Defines the canonical AgentState TypedDict shared across all agent nodes.

Design principle: State is the single source of truth. Every agent node
receives the full state and returns a *partial* dict with only the keys
it mutates. LangGraph merges these partials into the next state snapshot.

Using TypedDict (not Pydantic) because:
  - LangGraph's Annotated reducers work natively with TypedDict
  - Zero serialization overhead between nodes
  - Pydantic is reserved for the API boundary (api/schemas.py)
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ─── Sub-models (plain dicts for zero-cost serialisation) ───────────────────


class PlannerTask(TypedDict):
    """A single research sub-task produced by the Planner agent."""

    id: str                 # Unique within this session, e.g. "task-1"
    description: str        # What needs to be researched/computed
    task_type: str          # "search" | "code" | "both"


class SearchResult(TypedDict):
    """A single search result returned by the Web Search agent."""

    title: str
    url: str
    snippet: str
    source: str             # "duckduckgo" | "tavily"
    task_id: str            # Which PlannerTask this result addresses


class CodeResult(TypedDict):
    """Outcome of a sandboxed code execution."""

    task_id: str
    code: str               # The snippet that was executed
    stdout: str
    stderr: str
    success: bool
    execution_time_ms: float


class CritiqueResult(TypedDict):
    """Verdict produced by the Critic agent."""

    approved: bool
    score: float            # 0.0 – 1.0
    feedback: str           # Actionable feedback if not approved
    checked_criteria: list[str]


# ─── Root State ──────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all nodes in the LangGraph StateGraph.

    `total=False` means every key is optional at construction time, which is
    intentional — early nodes won't have values for keys set by later nodes.
    The graph is responsible for ensuring keys exist before a node reads them.
    """

    # ── Inputs ───────────────────────────────────────────────────────────────
    query: str                              # Original user research question
    session_id: str                         # UUID identifying this run

    # ── Planner output ───────────────────────────────────────────────────────
    plan: list[PlannerTask]

    # ── Web Search output ────────────────────────────────────────────────────
    search_results: list[SearchResult]

    # ── Retrieval output ─────────────────────────────────────────────────────
    retrieved_memories: list[str]           # Relevant past episode summaries

    # ── Code Executor output ─────────────────────────────────────────────────
    code_outputs: list[CodeResult]

    # ── Synthesis (produced just before the Critic) ───────────────────────────
    synthesis: str                          # Draft research report (Markdown)

    # ── Critic output ────────────────────────────────────────────────────────
    critique: CritiqueResult
    retry_count: int                        # Incremented on each critic rejection

    # ── Final output ─────────────────────────────────────────────────────────
    final_report: str                       # Approved, polished Markdown report

    # ── Message history (uses add_messages reducer) ───────────────────────────
    # Annotated with add_messages so LangGraph appends rather than overwrites
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Agent scratchpad ─────────────────────────────────────────────────────
    metadata: dict[str, Any]               # Timing, token counts, etc.
