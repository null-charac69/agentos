"""
api/routes.py
─────────────
FastAPI route definitions for AgentOS.

Endpoints:
  POST /api/v1/research         — Submit a research query (SSE stream)
  GET  /api/v1/sessions/{id}    — Fetch cached session progress/result
  GET  /api/v1/health           — Service health check

SSE streaming design:
  The `/research` endpoint returns `text/event-stream` (Server-Sent Events).
  We use LangGraph's `astream_events(version="v2")` to hook into specific
  node transitions and stream progress to the client in real time.

  Event types are defined in api/schemas.py::EventType and documented
  in the README so frontend consumers know what to expect.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from qdrant_client.http.exceptions import UnexpectedResponse

from api.schemas import (
    ErrorResponse,
    EventType,
    HealthStatus,
    ResearchRequest,
    ResearchResult,
    SessionStatus,
)
from cache.redis_cache import (
    cache_result,
    get_cached_result,
    get_session_progress,
    redis_ping,
    update_session_progress,
)
from core.graph import research_graph

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["research"])


# ─── SSE Event helper ─────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE `data:` line."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ─── Research stream generator ────────────────────────────────────────────────

async def _research_stream(
    query: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Core SSE generator. Drives the LangGraph pipeline and yields structured
    events so the client gets live progress updates.
    """
    start_time = time.monotonic()

    # ── Check cache ──────────────────────────────────────────────────────────
    cached = await get_cached_result(query)
    if cached:
        yield _sse(EventType.CACHE_HIT, {"session_id": session_id, "message": "Cache hit — returning stored result"})
        yield _sse(EventType.RESEARCH_COMPLETE, cached)
        yield _sse("done", {})
        return

    # ── Session started ──────────────────────────────────────────────────────
    start_event = {"session_id": session_id, "query": query}
    yield _sse(EventType.SESSION_STARTED, start_event)
    await update_session_progress(session_id, EventType.SESSION_STARTED, start_event)

    # ── Invoke the graph with streaming ──────────────────────────────────────
    initial_state = {
        "query": query,
        "session_id": session_id,
        "retry_count": 0,
        "metadata": {},
    }

    config = {"configurable": {"thread_id": session_id}}

    final_state: dict = {}

    try:
        async for event in research_graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            event_name = event.get("event", "")
            node_name = event.get("metadata", {}).get("langgraph_node", "")
            event_data = event.get("data", {})

            # ── Node entry → emit "agent started" ────────────────────────────
            if event_name == "on_chain_start" and node_name:
                agent_event = {
                    "session_id": session_id,
                    "agent": node_name,
                    "message": f"Agent '{node_name}' is running…",
                }
                yield _sse(EventType.AGENT_STARTED, agent_event)
                await update_session_progress(session_id, EventType.AGENT_STARTED, agent_event)

            # ── Node exit → emit specialised events ───────────────────────────
            elif event_name == "on_chain_end" and node_name:
                output = event_data.get("output", {})

                # Planner emits plan
                if node_name == "planner" and "plan" in output:
                    plan_event = {
                        "session_id": session_id,
                        "tasks": output["plan"],
                    }
                    yield _sse(EventType.PLAN_READY, plan_event)
                    await update_session_progress(session_id, EventType.PLAN_READY, plan_event)

                # Web search emits results count
                elif node_name == "web_search" and "search_results" in output:
                    sr_event = {
                        "session_id": session_id,
                        "count": len(output["search_results"]),
                        "sources": [r["url"] for r in output["search_results"][:5]],
                    }
                    yield _sse(EventType.SEARCH_RESULT, sr_event)

                # Retrieval recall emits memories
                elif node_name == "retrieval_recall" and "retrieved_memories" in output:
                    mem_event = {
                        "session_id": session_id,
                        "memories_found": len(output["retrieved_memories"]),
                    }
                    yield _sse(EventType.MEMORY_RECALLED, mem_event)

                # Code executor emits results
                elif node_name == "code_executor" and "code_outputs" in output:
                    for code_result in output["code_outputs"]:
                        yield _sse(EventType.CODE_EXECUTED, {
                            "session_id": session_id,
                            "task_id": code_result.get("task_id"),
                            "success": code_result.get("success"),
                            "stdout_preview": code_result.get("stdout", "")[:200],
                        })

                # Synthesis emits draft preview
                elif node_name == "synthesis" and "synthesis" in output:
                    synthesis_text = output["synthesis"]
                    synth_event = {
                        "session_id": session_id,
                        "preview": synthesis_text[:300] + ("…" if len(synthesis_text) > 300 else ""),
                        "length": len(synthesis_text),
                    }
                    yield _sse(EventType.SYNTHESIS_READY, synth_event)

                # Critic emits verdict
                elif node_name == "critic" and "critique" in output:
                    critique = output["critique"]
                    critique_event = {
                        "session_id": session_id,
                        "approved": critique.get("approved"),
                        "score": critique.get("score"),
                        "feedback": critique.get("feedback"),
                        "retry_count": output.get("retry_count", 0),
                    }
                    yield _sse(EventType.CRITIQUE_RESULT, critique_event)
                    await update_session_progress(session_id, EventType.CRITIQUE_RESULT, critique_event)

                # Track final state for result packaging
                if isinstance(output, dict):
                    final_state.update(output)

    except Exception as exc:
        logger.error("graph_stream_error", session_id=session_id, error=str(exc))
        error_event = {"session_id": session_id, "error": str(exc)}
        yield _sse(EventType.ERROR, error_event)
        yield _sse("done", {})
        return

    # ── Compile and emit final result ─────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    final_report = final_state.get("final_report") or final_state.get("synthesis", "")
    critique = final_state.get("critique", {})
    search_results = final_state.get("search_results", [])

    result = ResearchResult(
        session_id=session_id,
        query=query,
        final_report=final_report,
        critique_score=critique.get("score", 0.0),
        sources_count=len(search_results),
        retry_count=final_state.get("retry_count", 0),
        elapsed_seconds=round(elapsed, 2),
        cached=False,
    )

    result_dict = result.model_dump()

    # Cache the result for future identical queries
    await cache_result(query, result_dict)
    await update_session_progress(session_id, EventType.RESEARCH_COMPLETE, result_dict)

    yield _sse(EventType.RESEARCH_COMPLETE, result_dict)
    yield _sse("done", {})


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post(
    "/research",
    summary="Submit a research query",
    description=(
        "Accepts a natural language research question and streams progress events "
        "via Server-Sent Events (text/event-stream). The stream closes with a "
        "`research.complete` event containing the full report."
    ),
    response_description="SSE stream of research progress events",
)
async def research(
    body: ResearchRequest,
    request: Request,
) -> StreamingResponse:
    session_id = body.session_id or str(uuid.uuid4())
    query = body.query

    structlog.contextvars.bind_contextvars(session_id=session_id)
    logger.info("research_request_received", query=query[:80])

    return StreamingResponse(
        _research_stream(query, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",        # critical for Nginx/proxies
            "X-Session-ID": session_id,
        },
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionStatus,
    summary="Get session progress or result",
)
async def get_session(session_id: str) -> SessionStatus:
    """
    Retrieve the stored progress events for a research session.
    Useful for polling or SSE reconnect scenarios.
    """
    events = await get_session_progress(session_id)

    if not events:
        return SessionStatus(session_id=session_id, status="not_found")

    # Find the final result if it exists
    result_event = next(
        (e for e in events if e.get("event") == EventType.RESEARCH_COMPLETE),
        None,
    )

    status = "complete" if result_event else "running"
    result = None
    if result_event:
        try:
            result = ResearchResult(**result_event["data"])
        except Exception:
            pass

    return SessionStatus(
        session_id=session_id,
        status=status,
        events=events,
        result=result,
    )


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Service health check",
)
async def health() -> HealthStatus:
    """Liveness + readiness probe. Pings Redis and Qdrant."""
    from qdrant_client import QdrantClient
    from config.settings import get_settings

    settings = get_settings()

    # Check Redis
    redis_ok = await redis_ping()

    # Check Qdrant
    qdrant_ok = False
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    services = {"redis": redis_ok, "qdrant": qdrant_ok}
    all_healthy = all(services.values())

    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        services=services,
    )
