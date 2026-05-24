"""
tools/search_tools.py
─────────────────────
Web search tools used by the Web Search agent.

Strategy:
  1. Try DuckDuckGo first (free, no API key, rate-limited gently)
  2. Fall back to Tavily if DDG returns fewer than 2 results or raises

Both return a consistent list[SearchResult] so the agent doesn't care
which backend was used.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from core.state import SearchResult

logger = structlog.get_logger(__name__)

# ─── DuckDuckGo ──────────────────────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=False,
)
async def search_duckduckgo(
    query: str,
    task_id: str,
    max_results: int = 5,
) -> list[SearchResult]:
    """
    Search DuckDuckGo asynchronously.

    Runs the blocking duckduckgo-search call in a thread pool so it
    doesn't block the asyncio event loop.
    """
    from duckduckgo_search import DDGS

    def _blocking_search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, _blocking_search)
    except Exception as exc:
        logger.warning("duckduckgo_search_failed", query=query, error=str(exc))
        return []

    results: list[SearchResult] = []
    for r in raw_results:
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source": "duckduckgo",
                "task_id": task_id,
            }
        )

    logger.debug("duckduckgo_results", query=query, count=len(results))
    return results


# ─── Tavily ───────────────────────────────────────────────────────────────────


async def search_tavily(
    query: str,
    task_id: str,
    max_results: int = 5,
) -> list[SearchResult]:
    """
    Search Tavily — an AI-native search API optimised for LLM consumption.
    Returns an empty list if TAVILY_API_KEY is not set (graceful degradation).
    """
    from config.settings import get_settings

    settings = get_settings()
    if not settings.tavily_api_key or settings.tavily_api_key.startswith("tvly-your"):
        logger.debug("tavily_skipped_no_key")
        return []

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )
    except Exception as exc:
        logger.warning("tavily_search_failed", query=query, error=str(exc))
        return []

    results: list[SearchResult] = []
    for r in response.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "source": "tavily",
                "task_id": task_id,
            }
        )

    logger.debug("tavily_results", query=query, count=len(results))
    return results


# ─── Unified entry point ──────────────────────────────────────────────────────


async def search(
    query: str,
    task_id: str,
    max_results: int = 5,
) -> list[SearchResult]:
    """
    Primary search function. Tries DuckDuckGo first; if it returns fewer
    than 2 results, supplements with Tavily.
    """
    ddg_results = await search_duckduckgo(query, task_id, max_results)

    if len(ddg_results) >= 2:
        return ddg_results

    logger.info("ddg_insufficient_falling_back_to_tavily", query=query)
    tavily_results = await search_tavily(query, task_id, max_results)

    # Deduplicate by URL
    seen_urls: set[str] = {r["url"] for r in ddg_results}
    for r in tavily_results:
        if r["url"] not in seen_urls:
            ddg_results.append(r)
            seen_urls.add(r["url"])

    return ddg_results
