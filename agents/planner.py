"""
agents/planner.py
─────────────────
Planner agent — the first node in the research graph.

Responsibility:
  - Receive the raw user query
  - Decompose it into 3–5 concrete, independently researchable sub-tasks
  - Classify each sub-task as "search", "code", or "both"
  - Return the plan as a structured list[PlannerTask]

Why structured output (JSON mode)?
  - Downstream agents parse the plan programmatically
  - Free-text plans cause brittle string parsing and subtle bugs
  - Groq/OpenAI both support `with_structured_output` reliably
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm import get_llm
from core.state import AgentState, PlannerTask

logger = structlog.get_logger(__name__)

# ─── Structured output schema ─────────────────────────────────────────────────


class TaskItem(BaseModel):
    """Pydantic model used only for LLM structured output parsing."""

    description: str = Field(description="What needs to be researched or computed")
    task_type: str = Field(
        description="One of: 'search' (needs web search), 'code' (needs computation), 'both'"
    )


class ResearchPlan(BaseModel):
    """Top-level structured output from the Planner."""

    tasks: list[TaskItem] = Field(
        description="3 to 5 concrete sub-tasks that together answer the research question",
        min_length=2,
        max_length=6,
    )
    plan_rationale: str = Field(
        description="One sentence explaining the overall research approach"
    )


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Planner agent in an autonomous research system.

Your job is to decompose a research question into a structured research plan.

Guidelines:
1. Break the question into 3–5 focused sub-tasks. Each sub-task should be answerable independently.
2. Classify each sub-task:
   - "search": Requires finding information from the web (facts, news, papers, opinions)
   - "code": Requires computation, data analysis, or mathematical reasoning
   - "both": Requires both web evidence and computational verification
3. Be specific — avoid vague tasks like "research the topic". Instead: "Find the top 3 studies published in 2023-2024 on X".
4. Order tasks logically: foundational context first, then specifics, then synthesis.
5. Do NOT include tasks about writing the final report — a separate synthesis step handles that.
"""


# ─── Agent node ───────────────────────────────────────────────────────────────


async def planner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Planner.

    Reads:  state["query"], state.get("critique") (on retry)
    Writes: state["plan"]
    """
    query = state["query"]
    critique = state.get("critique")
    retry_count = state.get("retry_count", 0)

    logger.info("planner_started", query=query[:80], retry=retry_count)

    # On retry, incorporate critic feedback into the planning prompt
    human_content = f"Research question: {query}"
    if critique and not critique.get("approved", True):
        human_content += (
            f"\n\n[REVISION CONTEXT]\n"
            f"Previous attempt scored {critique.get('score', 0):.0%}.\n"
            f"Critic feedback: {critique.get('feedback', '')}\n"
            f"Please adjust the research plan to address these gaps."
        )

    llm = get_llm()
    structured_llm = llm.with_structured_output(ResearchPlan)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    try:
        result: ResearchPlan = await structured_llm.ainvoke(messages)
    except Exception as exc:
        logger.error("planner_failed", error=str(exc))
        # Graceful degradation: create a single catch-all search task
        fallback_plan: list[PlannerTask] = [
            {
                "id": "task-1",
                "description": query,
                "task_type": "search",
            }
        ]
        return {"plan": fallback_plan}

    # Convert Pydantic models to plain TypedDicts
    plan: list[PlannerTask] = [
        {
            "id": f"task-{i + 1}",
            "description": task.description,
            "task_type": task.task_type,
        }
        for i, task in enumerate(result.tasks)
    ]

    logger.info(
        "planner_completed",
        task_count=len(plan),
        rationale=result.plan_rationale[:100],
    )

    return {"plan": plan}
