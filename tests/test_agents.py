"""
tests/test_agents.py
─────────────────────
Unit tests for individual agent nodes.

Strategy:
  - Mock the LLM (no real API calls in tests)
  - Mock Qdrant and Redis clients
  - Test the pure state-transformation logic of each node
  - Verify each node reads the right inputs and writes the right outputs
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.state import AgentState, PlannerTask, SearchResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> AgentState:
    """Minimal valid state for testing individual agent nodes."""
    return {
        "query": "What are the latest advances in CRISPR gene editing?",
        "session_id": "test-session-001",
        "retry_count": 0,
        "metadata": {},
        "messages": [],
    }


@pytest.fixture
def state_with_plan(base_state) -> AgentState:
    return {
        **base_state,
        "plan": [
            {"id": "task-1", "description": "Find recent CRISPR research papers", "task_type": "search"},
            {"id": "task-2", "description": "Identify key researchers in the field", "task_type": "search"},
            {"id": "task-3", "description": "Calculate success rate statistics", "task_type": "code"},
        ],
    }


@pytest.fixture
def state_with_search(state_with_plan) -> AgentState:
    return {
        **state_with_plan,
        "search_results": [
            {
                "title": "CRISPR breakthrough 2024",
                "url": "https://example.com/crispr",
                "snippet": "Researchers achieved 99% efficiency in gene editing…",
                "source": "duckduckgo",
                "task_id": "task-1",
            }
        ],
        "retrieved_memories": [],
        "code_outputs": [],
    }


# ─── Planner tests ────────────────────────────────────────────────────────────

class TestPlannerNode:
    @pytest.mark.asyncio
    async def test_planner_returns_plan(self, base_state):
        """Planner should return a non-empty plan list."""
        from pydantic import BaseModel

        class MockPlan(BaseModel):
            tasks: list
            plan_rationale: str

        mock_response = MockPlan(
            tasks=[
                MagicMock(description="Find CRISPR papers", task_type="search"),
                MagicMock(description="Find key researchers", task_type="search"),
                MagicMock(description="Compute success statistics", task_type="code"),
            ],
            plan_rationale="Research the topic systematically.",
        )

        mock_structured_llm = AsyncMock(return_value=mock_response)

        with patch("agents.planner.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            from agents.planner import planner_node
            result = await planner_node(base_state)

        assert "plan" in result
        assert len(result["plan"]) == 3
        assert all("id" in task for task in result["plan"])
        assert all("task_type" in task for task in result["plan"])

    @pytest.mark.asyncio
    async def test_planner_fallback_on_llm_error(self, base_state):
        """Planner should return a single fallback task if LLM fails."""
        with patch("agents.planner.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock(
                side_effect=RuntimeError("LLM timeout")
            )
            mock_get_llm.return_value = mock_llm

            from agents.planner import planner_node
            result = await planner_node(base_state)

        assert "plan" in result
        assert len(result["plan"]) >= 1  # at least the fallback task

    @pytest.mark.asyncio
    async def test_planner_incorporates_critique_on_retry(self, base_state):
        """On retry, planner prompt should include critic feedback."""
        state_with_critique = {
            **base_state,
            "retry_count": 1,
            "critique": {
                "approved": False,
                "score": 0.5,
                "feedback": "Missing analysis of off-target effects.",
                "checked_criteria": [],
            },
        }

        captured_prompt = []

        async def capture_prompt(messages):
            captured_prompt.extend(messages)
            from pydantic import BaseModel
            class MockPlan(BaseModel):
                tasks: list
                plan_rationale: str
            return MockPlan(
                tasks=[MagicMock(description="task", task_type="search")],
                plan_rationale="retry plan",
            )

        with patch("agents.planner.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock(side_effect=capture_prompt)
            mock_get_llm.return_value = mock_llm

            from agents.planner import planner_node
            await planner_node(state_with_critique)

        # Verify critique feedback made it into the prompt
        human_message = captured_prompt[-1].content
        assert "off-target effects" in human_message


# ─── Web Search tests ─────────────────────────────────────────────────────────

class TestWebSearchNode:
    @pytest.mark.asyncio
    async def test_web_search_runs_per_task(self, state_with_plan):
        """Should call search for each 'search' or 'both' task."""
        call_count = 0

        async def mock_search(query, task_id, max_results):
            nonlocal call_count
            call_count += 1
            return [{"title": "Test", "url": f"https://test.com/{task_id}", "snippet": "...", "source": "duckduckgo", "task_id": task_id}]

        with patch("agents.web_search.search", side_effect=mock_search):
            from agents.web_search import web_search_node
            result = await web_search_node(state_with_plan)

        assert "search_results" in result
        # 2 search tasks + 1 both task = 3 searches
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_web_search_deduplicates_urls(self, state_with_plan):
        """Duplicate URLs from different tasks should appear only once."""
        async def mock_search(query, task_id, max_results):
            # Return same URL for all tasks
            return [{"title": "Dup", "url": "https://same-url.com", "snippet": "...", "source": "duckduckgo", "task_id": task_id}]

        with patch("agents.web_search.search", side_effect=mock_search):
            from agents.web_search import web_search_node
            result = await web_search_node(state_with_plan)

        urls = [r["url"] for r in result["search_results"]]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    @pytest.mark.asyncio
    async def test_web_search_skipped_for_code_only_plan(self, base_state):
        """If all tasks are 'code' type, search should be skipped."""
        code_only_state = {
            **base_state,
            "plan": [{"id": "task-1", "description": "Compute π to 100 digits", "task_type": "code"}],
        }
        from agents.web_search import web_search_node
        result = await web_search_node(code_only_state)
        assert result["search_results"] == []


# ─── Critic tests ─────────────────────────────────────────────────────────────

class TestCriticNode:
    @pytest.mark.asyncio
    async def test_critic_approves_good_synthesis(self, state_with_search):
        """Critic should approve and set final_report when score >= threshold."""
        from pydantic import BaseModel

        class MockCritique(BaseModel):
            approved: bool = True
            score: float = 0.9
            coverage_ok: bool = True
            grounding_ok: bool = True
            coherence_ok: bool = True
            feedback: str = "Excellent synthesis."

        state = {
            **state_with_search,
            "synthesis": "# CRISPR Report\n\n## Summary\nCRISPR advances in 2024...",
        }

        with patch("agents.critic.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock(return_value=MockCritique())
            mock_get_llm.return_value = mock_llm

            from agents.critic import critic_node
            result = await critic_node(state)

        assert result["critique"]["approved"] is True
        assert result["critique"]["score"] == 0.9
        assert "final_report" in result

    @pytest.mark.asyncio
    async def test_critic_rejects_poor_synthesis(self, state_with_search):
        """Critic should reject and increment retry_count when score < threshold."""
        from pydantic import BaseModel

        class MockCritique(BaseModel):
            approved: bool = False
            score: float = 0.4
            coverage_ok: bool = False
            grounding_ok: bool = True
            coherence_ok: bool = True
            feedback: str = "Missing off-target effects analysis."

        state = {
            **state_with_search,
            "synthesis": "# Brief Report\n\nCRISPR is cool.",
            "retry_count": 0,
        }

        with patch("agents.critic.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = AsyncMock(return_value=MockCritique())
            mock_get_llm.return_value = mock_llm

            from agents.critic import critic_node
            result = await critic_node(state)

        assert result["critique"]["approved"] is False
        assert result["retry_count"] == 1  # incremented
        assert "final_report" not in result
