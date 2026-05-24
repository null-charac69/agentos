"""
tests/test_api.py
──────────────────
API endpoint tests using FastAPI's TestClient.

Tests:
  - /api/v1/health returns 200 with correct schema
  - /api/v1/research validates bad input (short query)
  - /api/v1/research streams a valid SSE response for a real query
  - /api/v1/sessions/{id} returns 200 for known sessions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─── App fixture ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with all external services mocked.
    This allows API tests to run without Qdrant, Redis, or a real LLM.
    """
    with (
        patch("memory.qdrant_store.ensure_collections_exist"),
        patch("cache.redis_cache.get_redis", new_callable=AsyncMock),
    ):
        from api.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ─── Health endpoint ──────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        with (
            patch("api.routes.redis_ping", return_value=AsyncMock(return_value=True)()),
            patch("api.routes.QdrantClient") as mock_qdrant,
        ):
            mock_qdrant.return_value.get_collections.return_value = []
            response = client.get("/api/v1/health")

        assert response.status_code == 200

    def test_health_response_schema(self, client):
        with (
            patch("api.routes.redis_ping", return_value=AsyncMock(return_value=True)()),
            patch("api.routes.QdrantClient") as mock_qdrant,
        ):
            mock_qdrant.return_value.get_collections.return_value = []
            response = client.get("/api/v1/health")

        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data


# ─── Research endpoint validation ─────────────────────────────────────────────

class TestResearchEndpointValidation:
    def test_rejects_empty_query(self, client):
        response = client.post(
            "/api/v1/research",
            json={"query": ""},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_rejects_too_short_query(self, client):
        response = client.post(
            "/api/v1/research",
            json={"query": "CRISPR"},  # < 10 chars
        )
        assert response.status_code == 422

    def test_rejects_missing_query_field(self, client):
        response = client.post(
            "/api/v1/research",
            json={},
        )
        assert response.status_code == 422

    def test_accepts_valid_query(self, client):
        """Valid query should start streaming (200, event-stream content type)."""
        # Mock the entire graph to avoid LLM/Qdrant calls
        async def mock_stream(*args, **kwargs):
            yield 'event: session.started\ndata: {"session_id": "test"}\n\n'
            yield 'event: research.complete\ndata: {"final_report": "Test report", "session_id": "test", "query": "test", "critique_score": 0.9, "sources_count": 3, "retry_count": 0, "elapsed_seconds": 5.0, "cached": false}\n\n'
            yield 'event: done\ndata: {}\n\n'

        with (
            patch("api.routes.get_cached_result", new_callable=AsyncMock, return_value=None),
            patch("api.routes.update_session_progress", new_callable=AsyncMock),
            patch("api.routes.cache_result", new_callable=AsyncMock),
            patch("api.routes.research_graph") as mock_graph,
        ):
            mock_graph.astream_events = mock_stream
            response = client.post(
                "/api/v1/research",
                json={"query": "What are the latest breakthroughs in quantum computing?"},
                stream=True,
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


# ─── Sessions endpoint ────────────────────────────────────────────────────────

class TestSessionsEndpoint:
    def test_unknown_session_returns_not_found_status(self, client):
        with patch("api.routes.get_session_progress", new_callable=AsyncMock, return_value=[]):
            response = client.get("/api/v1/sessions/nonexistent-session-id")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"

    def test_known_session_returns_events(self, client):
        mock_events = [
            {"event": "session.started", "data": {"session_id": "abc123", "query": "test query"}},
            {"event": "plan.ready", "data": {"tasks": []}},
        ]

        with patch("api.routes.get_session_progress", new_callable=AsyncMock, return_value=mock_events):
            response = client.get("/api/v1/sessions/abc123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert len(data["events"]) == 2
