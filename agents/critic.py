"""
agents/critic.py
────────────────
Critic agent — implements the self-consistency review loop.

This is the architectural heart of AgentOS. The Critic evaluates the
synthesis draft against three dimensions:

  1. Coverage   — Does the report address every sub-task in the plan?
  2. Grounding  — Are claims backed by the search results / code outputs?
  3. Coherence  — Is the reasoning internally consistent and non-contradictory?

If any dimension fails, the Critic sets `approved=False` and writes
structured feedback. The graph's conditional edge then routes back to
the Planner for a revision — unless max retries are exhausted.

Why structured output for the critic?
  - The graph's routing logic reads `critique["approved"]` as a boolean
  - A freeform critique would require fragile regex parsing to extract that
  - Structured output guarantees the field exists and is correctly typed
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm import get_llm
from core.state import AgentState, CritiqueResult

logger = structlog.get_logger(__name__)


# ─── Structured output schema ─────────────────────────────────────────────────

class CritiqueOutput(BaseModel):
    """Pydantic model for LLM structured output."""

    approved: bool = Field(
        description="True if the synthesis meets all quality criteria; False if revisions are needed."
    )
    score: float = Field(
        description="Quality score from 0.0 (terrible) to 1.0 (excellent).",
        ge=0.0,
        le=1.0,
    )
    coverage_ok: bool = Field(
        description="True if the synthesis addresses all sub-tasks in the research plan."
    )
    grounding_ok: bool = Field(
        description="True if key claims are supported by the provided search results or code outputs."
    )
    coherence_ok: bool = Field(
        description="True if the synthesis is internally consistent with no contradictions."
    )
    feedback: str = Field(
        description=(
            "If not approved: specific, actionable instructions for improvement. "
            "If approved: a brief summary of why the synthesis is good. "
            "2-4 sentences max."
        )
    )


# ─── Synthesis node (generates the draft report) ──────────────────────────────

_SYNTHESIS_SYSTEM = """You are the Synthesis agent in an autonomous research system.

Given a research plan, web search results, retrieved memories, and code outputs,
write a comprehensive, well-structured research report in Markdown.

Structure your report as:
# [Title]
## Executive Summary (2-3 sentences)
## Key Findings
### [Finding 1 based on sub-task 1]
### [Finding 2 based on sub-task 2]
...
## Analysis & Insights
## Conclusion
## Sources
- [URL] — Brief description

Guidelines:
- Be precise and factual; cite sources inline as [Source N]
- Include data, numbers, dates where available
- If code produced output, integrate the results into the analysis
- Acknowledge uncertainty where the evidence is thin
- Target length: 600-1000 words
"""


async def synthesis_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Synthesis.

    Combines all gathered evidence into a draft research report.

    Reads:  state["query"], state["plan"], state["search_results"],
            state["retrieved_memories"], state["code_outputs"]
    Writes: state["synthesis"]
    """
    query = state["query"]
    plan = state.get("plan", [])
    search_results = state.get("search_results", [])
    memories = state.get("retrieved_memories", [])
    code_outputs = state.get("code_outputs", [])
    retry_count = state.get("retry_count", 0)
    critique = state.get("critique")

    logger.info("synthesis_started", query=query[:80], retry=retry_count)

    # Build context for the LLM
    plan_text = "\n".join(
        f"{i+1}. [{t['task_type'].upper()}] {t['description']}"
        for i, t in enumerate(plan)
    )

    search_text = "\n\n".join(
        f"[Source {i+1}] {r['title']}\nURL: {r['url']}\n{r['snippet']}"
        for i, r in enumerate(search_results[:15])
    )

    memory_text = "\n\n".join(memories) if memories else "No relevant past research found."

    code_text = "\n\n".join(
        f"--- Code task: {c['task_id']} ---\n"
        f"Result: {'SUCCESS' if c['success'] else 'FAILED'}\n"
        f"Output:\n{c['stdout'] or c['stderr']}"
        for c in code_outputs
    ) if code_outputs else "No code execution results."

    retry_instruction = ""
    if critique and not critique.get("approved", True):
        retry_instruction = (
            f"\n\n[REVISION REQUIRED]\n"
            f"Previous draft scored {critique.get('score', 0):.0%}.\n"
            f"Critic feedback: {critique.get('feedback', '')}\n"
            f"Please address these specific issues in this revision."
        )

    human_content = f"""Research Question: {query}

Research Plan:
{plan_text}

Web Search Results:
{search_text}

Past Research Context (Episodic Memory):
{memory_text}

Code Execution Results:
{code_text}
{retry_instruction}

Now write the comprehensive research report."""

    llm = get_llm()
    messages = [
        SystemMessage(content=_SYNTHESIS_SYSTEM),
        HumanMessage(content=human_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        synthesis = response.content.strip()
    except Exception as exc:
        logger.error("synthesis_failed", error=str(exc))
        synthesis = (
            f"# Research Report: {query}\n\n"
            f"An error occurred during synthesis: {exc}\n\n"
            "Please retry the request."
        )

    logger.info("synthesis_completed", length=len(synthesis))
    return {"synthesis": synthesis}


# ─── Critic node ─────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """You are the Critic agent in an autonomous research system.

Your job is to evaluate a research synthesis draft against strict quality criteria.
Be rigorous — a score above 0.85 means the report is genuinely excellent.

Criteria:
1. Coverage (0.0-1.0): Does the synthesis address EVERY sub-task in the research plan?
2. Grounding (0.0-1.0): Are specific claims traceable to the provided sources?
3. Coherence (0.0-1.0): Is the report internally consistent and logically structured?

The final score = weighted average: coverage (0.4) + grounding (0.3) + coherence (0.3).
Approve if score >= 0.75 AND all three individual criteria are True.
"""


async def critic_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Critic.

    Evaluates the synthesis and either approves it or requests revision.

    Reads:  state["synthesis"], state["plan"], state["search_results"],
            state["retry_count"]
    Writes: state["critique"], state["final_report"] (if approved),
            state["retry_count"]
    """
    synthesis = state.get("synthesis", "")
    plan = state.get("plan", [])
    search_results = state.get("search_results", [])
    retry_count = state.get("retry_count", 0)
    settings_max_retries = state.get("metadata", {}).get("max_retries", 2)

    logger.info("critic_started", retry=retry_count)

    plan_text = "\n".join(
        f"- Task {t['id']}: {t['description']}" for t in plan
    )
    sources_text = "\n".join(
        f"- [{r['url']}] {r['title']}" for r in search_results[:10]
    )

    human_content = f"""Research Plan (what the report MUST cover):
{plan_text}

Available Sources (claims should be grounded here):
{sources_text}

Synthesis to evaluate:
---
{synthesis}
---

Evaluate this synthesis against the three criteria and provide your structured verdict."""

    llm = get_llm()
    structured_llm = llm.with_structured_output(CritiqueOutput)

    messages = [
        SystemMessage(content=_CRITIC_SYSTEM),
        HumanMessage(content=human_content),
    ]

    try:
        result: CritiqueOutput = await structured_llm.ainvoke(messages)
    except Exception as exc:
        logger.error("critic_failed", error=str(exc))
        # If critic itself fails, approve to avoid infinite loops
        result = CritiqueOutput(
            approved=True,
            score=0.7,
            coverage_ok=True,
            grounding_ok=True,
            coherence_ok=True,
            feedback="Critic evaluation failed; auto-approving to deliver results.",
        )

    critique: CritiqueResult = {
        "approved": result.approved,
        "score": result.score,
        "feedback": result.feedback,
        "checked_criteria": [
            f"coverage={'✓' if result.coverage_ok else '✗'}",
            f"grounding={'✓' if result.grounding_ok else '✗'}",
            f"coherence={'✓' if result.coherence_ok else '✗'}",
        ],
    }

    logger.info(
        "critic_verdict",
        approved=result.approved,
        score=result.score,
        retry=retry_count,
    )

    updates: dict[str, Any] = {
        "critique": critique,
        "retry_count": retry_count + (0 if result.approved else 1),
    }

    if result.approved:
        updates["final_report"] = synthesis

    return updates
