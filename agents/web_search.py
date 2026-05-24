"""
agents/web_search.py
─────────────────────
Web Search agent — searches the internet for each planner sub-task.

Responsibility:
  - Iterate over plan tasks of type "search" or "both"
  - Run parallel web searches (one per task, capped at MAX_CONCURRENT)
  - Deduplicate and rank results
  - Return a flat list of SearchResult dicts

Why parallel search?
  - A 4-task plan with sequential 2s searches = 8s latency
  - Parallel with asyncio.gather = ~2s regardless of task count
  - We cap concurrency at 3 to avoid DDG rate limits
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from config.settings import get_settings
from core.state import AgentState, PlannerTask, SearchResult
from tools.search_tools import search

logger = structlog.get_logger(__name__)

_MAX_CONCURRENT_SEARCHES = 3
_SEARCH_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)


async def _search_for_task(
    task: PlannerTask,
    max_results: int,
) -> list[SearchResult]:
    """Run a search for a single planner task, respecting the concurrency limit."""
    async with _SEARCH_SEMAPHORE:
        results = await search(
            query=task["description"],
            task_id=task["id"],
            max_results=max_results,
        )
    return results


async def web_search_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Web Search.

    Reads:  state["plan"]
    Writes: state["search_results"]
    """
    plan = state.get("plan", [])
    settings = get_settings()
    max_results = settings.search_results_per_task

    # Only search for tasks that require it
    search_tasks = [t for t in plan if t["task_type"] in ("search", "both")]

    if not search_tasks:
        logger.info("web_search_skipped_no_search_tasks")
        return {"search_results": []}

    logger.info("web_search_started", task_count=len(search_tasks))

    # Fan out searches in parallel
    search_coroutines = [
        _search_for_task(task, max_results)
        for task in search_tasks
    ]
    results_per_task = await asyncio.gather(*search_coroutines, return_exceptions=True)

    all_results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for results in results_per_task:
        if isinstance(results, Exception):
            logger.warning("search_task_failed", error=str(results))
            continue
        for r in results:
            if r["url"] not in seen_urls and r["url"]:  # deduplicate
                all_results.append(r)
                seen_urls.add(r["url"])

    logger.info(
        "web_search_completed",
        total_results=len(all_results),
        unique_sources=len(seen_urls),
    )

    return {"search_results": all_results}
