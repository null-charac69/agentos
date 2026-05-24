"""
api/schemas.py
──────────────
Pydantic v2 request/response schemas for the FastAPI API.

These schemas live at the API boundary — they're separate from core/state.py's
TypedDicts (which live inside the graph). This separation means:
  - API contracts can evolve independently of internal state structure
  - Input validation happens before touching the graph
  - Response shapes can be versioned
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ─── Request schemas ──────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """Body for POST /api/v1/research"""

    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The research question or topic to investigate.",
        examples=["What are the latest breakthroughs in quantum computing in 2024?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID. Auto-generated if not provided.",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the latest breakthroughs in quantum computing in 2024?",
            }
        }
    }


# ─── SSE event schemas ────────────────────────────────────────────────────────

class StreamEvent(BaseModel):
    """A single Server-Sent Event emitted during research."""

    event: str = Field(description="Event type identifier")
    session_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Event type constants
class EventType:
    SESSION_STARTED = "session.started"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    PLAN_READY = "plan.ready"
    SEARCH_RESULT = "search.result"
    MEMORY_RECALLED = "memory.recalled"
    CODE_EXECUTED = "code.executed"
    SYNTHESIS_READY = "synthesis.ready"
    CRITIQUE_RESULT = "critique.result"
    RESEARCH_COMPLETE = "research.complete"
    CACHE_HIT = "cache.hit"
    ERROR = "error"


# ─── Response schemas ─────────────────────────────────────────────────────────

class ResearchResult(BaseModel):
    """Final research result (returned in research.complete event and /sessions/{id})"""

    session_id: str
    query: str
    final_report: str
    critique_score: float = Field(ge=0.0, le=1.0)
    sources_count: int
    retry_count: int
    elapsed_seconds: float
    cached: bool = False


class SessionStatus(BaseModel):
    """Response for GET /api/v1/sessions/{session_id}"""

    session_id: str
    status: str  # "running" | "complete" | "not_found"
    events: list[dict[str, Any]] = Field(default_factory=list)
    result: Optional[ResearchResult] = None


class HealthStatus(BaseModel):
    """Response for GET /api/v1/health"""

    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str = "0.1.0"
    services: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-service health: {qdrant: true, redis: true}",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: Optional[str] = None
    session_id: Optional[str] = None
