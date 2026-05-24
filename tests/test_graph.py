"""
tests/test_graph.py
────────────────────
Integration tests for the LangGraph StateGraph routing logic.

Tests the conditional edge routing without running the full LLM pipeline
by mocking individual agent nodes and verifying graph traversal behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.graph import _route_after_critic
from core.state import AgentState


# ─── Routing logic tests ──────────────────────────────────────────────────────

class TestCriticRouting:
    """Tests for the _route_after_critic conditional edge function."""

    def test_routes_to_store_on_approval(self):
        """Approved critique should route to retrieval_store."""
        state: AgentState = {
            "query": "test",
            "session_id": "s1",
            "retry_count": 0,
            "critique": {"approved": True, "score": 0.9, "feedback": "Good", "checked_criteria": []},
            "messages": [],
        }
        result = _route_after_critic(state)
        assert result == "retrieval_store"

    def test_routes_to_planner_on_rejection_within_retries(self):
        """Rejected critique within retry limit should route back to planner."""
        state: AgentState = {
            "query": "test",
            "session_id": "s1",
            "retry_count": 1,  # below default max of 2
            "critique": {"approved": False, "score": 0.4, "feedback": "Needs work", "checked_criteria": []},
            "messages": [],
        }
        with patch("core.graph.get_settings") as mock_settings:
            mock_settings.return_value.max_critic_retries = 2
            result = _route_after_critic(state)
        assert result == "planner"

    def test_routes_to_end_on_max_retries(self):
        """Exhausted retries should route to END regardless of critic verdict."""
        state: AgentState = {
            "query": "test",
            "session_id": "s1",
            "retry_count": 2,  # at the limit
            "critique": {"approved": False, "score": 0.3, "feedback": "Still bad", "checked_criteria": []},
            "messages": [],
        }
        with patch("core.graph.get_settings") as mock_settings:
            mock_settings.return_value.max_critic_retries = 2
            result = _route_after_critic(state)
        assert result == "__end__"

    def test_routes_to_store_even_with_retries_remaining_if_approved(self):
        """Approval on first retry should still route to store, not planner."""
        state: AgentState = {
            "query": "test",
            "session_id": "s1",
            "retry_count": 1,
            "critique": {"approved": True, "score": 0.82, "feedback": "Now good", "checked_criteria": []},
            "messages": [],
        }
        result = _route_after_critic(state)
        assert result == "retrieval_store"


# ─── Graph structure tests ────────────────────────────────────────────────────

class TestGraphStructure:
    def test_graph_compiles_without_error(self):
        """Graph should compile with all nodes and edges registered."""
        from core.graph import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """All five agent nodes plus retrieval variants should be present."""
        from core.graph import build_graph
        graph = build_graph()

        # Access the underlying graph structure
        node_names = set(graph.get_graph().nodes.keys())
        expected_nodes = {
            "retrieval_recall",
            "planner",
            "web_search",
            "code_executor",
            "synthesis",
            "critic",
            "retrieval_store",
        }
        for node in expected_nodes:
            assert node in node_names, f"Missing expected node: {node}"

    def test_graph_input_schema(self):
        """Graph should accept AgentState-compatible input."""
        from core.graph import research_graph
        # Verify the graph has a valid schema
        assert research_graph is not None
        schema = research_graph.get_input_schema()
        assert schema is not None
