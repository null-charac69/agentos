"""
agents/code_executor.py
────────────────────────
Code Execution agent — executes Python snippets for computational sub-tasks.

Responsibility:
  - Scan the plan for tasks of type "code" or "both"
  - Ask the LLM to generate a Python snippet for each such task
  - Execute the snippet in the RestrictedPython sandbox
  - Collect stdout/stderr and return CodeResult dicts

Why generate code via LLM rather than pre-write it?
  - The planner sub-tasks are open-ended and query-dependent
  - The LLM generates contextually appropriate snippets (statistics,
    unit conversions, data processing) that can't be pre-defined
  - The sandbox keeps this safe
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings
from core.llm import get_llm
from core.state import AgentState, CodeResult, PlannerTask
from tools.code_tools import execute_code

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Code Execution agent in an autonomous research system.

Given a research sub-task that requires computation, generate a self-contained Python snippet.

Rules:
1. The snippet MUST be self-contained — no external files, no network calls.
2. Only use: math, statistics, json, re, datetime, collections, itertools, functools, random, decimal
3. Use print() to output results — stdout is captured and shown.
4. Keep the snippet under 50 lines.
5. Handle potential errors with try/except.
6. Output ONLY the Python code, nothing else. No markdown fences.
"""


async def _generate_and_execute(
    task: PlannerTask,
    search_context: str,
    timeout: int,
) -> CodeResult:
    """Generate a Python snippet for one task and execute it."""
    llm = get_llm()

    context_snippet = search_context[:2000] if search_context else "No search context available."

    prompt = (
        f"Research sub-task: {task['description']}\n\n"
        f"Available context from web search:\n{context_snippet}\n\n"
        f"Write a Python snippet to compute or analyse this sub-task."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        code = response.content.strip()
        # Strip accidental markdown fences if the model adds them
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = "\n".join(code.split("\n")[:-1])
        code = code.strip()
    except Exception as exc:
        logger.warning("code_generation_failed", task_id=task["id"], error=str(exc))
        return {
            "task_id": task["id"],
            "code": "",
            "stdout": "",
            "stderr": f"Code generation failed: {exc}",
            "success": False,
            "execution_time_ms": 0.0,
        }

    # Execute synchronously (execute_code uses threading internally)
    result = execute_code(code=code, task_id=task["id"], timeout=timeout)
    return result


async def code_executor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Code Executor.

    Reads:  state["plan"], state["search_results"]
    Writes: state["code_outputs"]
    """
    plan = state.get("plan", [])
    search_results = state.get("search_results", [])
    settings = get_settings()
    timeout = settings.code_exec_timeout

    # Only process tasks requiring computation
    code_tasks = [t for t in plan if t["task_type"] in ("code", "both")]

    if not code_tasks:
        logger.info("code_executor_skipped_no_code_tasks")
        return {"code_outputs": []}

    logger.info("code_executor_started", task_count=len(code_tasks))

    # Build a search context string for the LLM
    search_context = "\n\n".join(
        f"[{r['source']}] {r['title']}\n{r['snippet']}"
        for r in search_results[:10]
    )

    # Execute tasks sequentially (sandbox is CPU-bound; parallel won't help)
    code_outputs: list[CodeResult] = []
    for task in code_tasks:
        result = await _generate_and_execute(task, search_context, timeout)
        code_outputs.append(result)

    success_count = sum(1 for r in code_outputs if r["success"])
    logger.info(
        "code_executor_completed",
        total=len(code_outputs),
        success=success_count,
    )

    return {"code_outputs": code_outputs}
