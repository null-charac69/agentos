# AgentOS — Autonomous Multi-Agent Research System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B35?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.12+-DC143C?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Submit a research question → get a structured, cited research report in ~45 seconds.**

[Quick Start](#quick-start) · [Architecture](#architecture) · [API Reference](#api-reference) · [Configuration](#configuration) · [Development](#development)

</div>

---

## Overview

AgentOS is an autonomous research pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph) state graphs. It accepts open-ended research questions and orchestrates **five specialist sub-agents** that collaborate to produce a structured, cited Markdown report — complete with a self-consistency critic loop that improves quality before delivery.

**Key capabilities:**
- 🧠 **Stateful multi-agent graph** — typed shared state flows through every agent node
- 🔍 **Parallel web search** — DuckDuckGo + Tavily fallback, 3-concurrent, deduplicated
- 💾 **Episodic memory** — Qdrant stores past sessions; semantically similar queries get richer context
- 🐍 **Sandboxed code execution** — LLM-generated Python runs in a RestrictedPython environment
- 🔁 **Self-consistency critic loop** — 3-axis quality scoring (coverage · grounding · coherence) with automatic revision
- ⚡ **Streaming SSE API** — live progress events per agent; not just a final response after 45s
- 🗄️ **Redis result cache** — identical queries served from cache in <50ms
- 📊 **LangSmith observability** — optional; full trace visibility when key is set

---

## Architecture

```
User Query (POST /api/v1/research)
         │
         ▼
  ┌──────────────────────────────────────────┐
  │         FastAPI Streaming Service         │
  │         (SSE / text/event-stream)         │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │                  LangGraph StateGraph                     │
  │                                                          │
  │  retrieval_recall ──► planner ──► web_search             │
  │                                        │                 │
  │                                  code_executor           │
  │                                        │                 │
  │                                   synthesis              │
  │                                        │                 │
  │                                     critic               │
  │                                   /       \              │
  │                           [approved]   [rejected]        │
  │                               │            │             │
  │                       retrieval_store    planner         │
  │                               │         (retry)          │
  │                              END                         │
  └──────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
     Qdrant                           Redis
  (episodic memory)            (result cache / sessions)
```

### The Five Agents

| Agent | Role | Tools |
|---|---|---|
| **Planner** | Decomposes query into 3–5 typed sub-tasks | Structured LLM output |
| **Web Search** | Parallel search per sub-task; DDG + Tavily fallback | `duckduckgo-search`, `tavily-python` |
| **Retrieval** | Recalls similar past sessions from Qdrant | `qdrant-client`, `sentence-transformers` |
| **Code Executor** | Generates + runs Python for computational sub-tasks | RestrictedPython sandbox |
| **Critic** | Scores synthesis on 3 axes; drives the retry loop | Structured LLM output |

### Shared State

Every agent node is a pure function `(AgentState) → dict[partial updates]`. The graph merges partial updates into the next state snapshot — no node mutates state in-place.

```python
class AgentState(TypedDict, total=False):
    query: str                          # Original question
    session_id: str                     # UUID for this run
    plan: list[PlannerTask]             # Planner output
    search_results: list[SearchResult]  # Web search results
    retrieved_memories: list[str]       # Qdrant episodic recall
    code_outputs: list[CodeResult]      # Sandboxed execution output
    synthesis: str                      # Draft report
    critique: CritiqueResult            # Critic verdict
    retry_count: int                    # Critic iteration counter
    final_report: str                   # Approved report
    messages: Annotated[list, add_messages]
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- A [Groq API key](https://console.groq.com) (free tier, no credit card)

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/agentos.git
cd agentos
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
GROQ_API_KEY=gsk_your_key_here
```

### 2. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml up qdrant redis -d
```

### 3. Install and run

```bash
pip install -e ".[dev]"
make dev
```

The API is now running at `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 4. Send your first research query

```bash
curl -N -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest breakthroughs in nuclear fusion energy in 2024?"}' \
  --no-buffer
```

You'll see a live stream of SSE events as each agent completes its work.

### Full Docker deployment

```bash
docker compose -f docker/docker-compose.yml up -d
```

This starts Qdrant, Redis, the Qdrant collection init container, and the AgentOS API — all wired together with health checks.

---

## API Reference

### `POST /api/v1/research`

Submit a research query. Returns a `text/event-stream` (SSE) response.

**Request body:**
```json
{
  "query": "What are the latest advances in CRISPR gene editing?",
  "session_id": "optional-uuid-for-reconnect"
}
```

**SSE Event types (in order):**

| Event | When | Payload |
|---|---|---|
| `session.started` | Immediately | `{session_id, query}` |
| `cache.hit` | If cached | `{message}` |
| `agent.started` | Each agent begins | `{agent, message}` |
| `plan.ready` | After Planner | `{tasks: [{id, description, task_type}]}` |
| `memory.recalled` | After Retrieval | `{memories_found}` |
| `search.result` | After Web Search | `{count, sources: [url...]}` |
| `code.executed` | Each code snippet | `{task_id, success, stdout_preview}` |
| `synthesis.ready` | Draft generated | `{preview, length}` |
| `critique.result` | Critic verdict | `{approved, score, feedback, retry_count}` |
| `research.complete` | Final output | Full `ResearchResult` object |
| `done` | Stream end | `{}` |

**Example `research.complete` payload:**
```json
{
  "session_id": "a1b2c3d4-...",
  "query": "Latest advances in CRISPR...",
  "final_report": "# CRISPR Gene Editing: 2024 Advances\n\n...",
  "critique_score": 0.87,
  "sources_count": 12,
  "retry_count": 0,
  "elapsed_seconds": 38.4,
  "cached": false
}
```

---

### `GET /api/v1/sessions/{session_id}`

Retrieve stored progress events and result for a past session.

```bash
curl http://localhost:8000/api/v1/sessions/a1b2c3d4-...
```

---

### `GET /api/v1/health`

Liveness + readiness probe. Pings Redis and Qdrant.

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {"redis": true, "qdrant": true}
}
```

---

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full reference.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model name for the chosen provider |
| `GROQ_API_KEY` | — | **Required** for Groq (free at console.groq.com) |
| `TAVILY_API_KEY` | — | Optional fallback search (1000 free/month) |
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MAX_CRITIC_RETRIES` | `2` | Max revision loops before force-finishing |
| `LANGCHAIN_API_KEY` | — | Optional: enables LangSmith tracing |

---

## Development

```bash
# Install with dev dependencies
make install

# Run tests
make test

# Lint
make lint

# Type check
make typecheck

# Run with hot-reload
make dev
```

### Project Structure

```
agentos/
├── agents/          # Five specialist agent nodes
│   ├── planner.py
│   ├── web_search.py
│   ├── retrieval.py
│   ├── code_executor.py
│   └── critic.py    # also contains synthesis_node
├── core/
│   ├── state.py     # AgentState TypedDict + sub-models
│   ├── graph.py     # LangGraph StateGraph builder
│   └── llm.py       # LLM factory (Groq / OpenAI)
├── memory/
│   ├── embeddings.py    # sentence-transformers (local)
│   └── qdrant_store.py  # Episodic memory operations
├── tools/
│   ├── search_tools.py  # DuckDuckGo + Tavily
│   └── code_tools.py    # RestrictedPython sandbox
├── cache/
│   └── redis_cache.py   # Result cache + session store
├── api/
│   ├── main.py          # FastAPI app factory
│   ├── routes.py        # /research, /sessions, /health
│   ├── schemas.py       # Pydantic request/response models
│   └── middleware.py    # CORS, logging, timing
├── docker/
│   ├── Dockerfile       # Multi-stage Python image
│   └── docker-compose.yml
├── scripts/
│   └── init_qdrant.py   # One-time collection setup
└── tests/
    ├── test_agents.py
    ├── test_graph.py
    └── test_api.py
```

---

## Tech Stack Decisions

| Component | Choice | Rationale |
|---|---|---|
| **LLM** | Groq (Llama 3.3 70B) | Free tier, ~750 tok/s inference; swap in any LangChain-compatible model |
| **Agent framework** | LangGraph | Native support for cyclic state graphs, checkpointing, and streaming |
| **Vector DB** | Qdrant | Best open-source vector DB; Docker-native, rich filtering, gRPC support |
| **Embeddings** | `all-MiniLM-L6-v2` | 80MB local model, 384-dim, zero API cost |
| **Web search** | DuckDuckGo + Tavily | No-cost primary; LLM-optimised fallback |
| **Code sandbox** | RestrictedPython | No Docker-in-Docker; allowlist-based import control |
| **Streaming** | SSE over HTTP | Simple, browser-native; no WebSocket handshake overhead |
| **Cache** | Redis 7 Alpine | Lightweight; built-in TTL; hiredis for C-speed parsing |
| **Checkpointing** | `MemorySaver` | Zero-config for single-node; swap to `RedisCheckpointer` for HA |

---

## Scaling Notes

- **Horizontal scaling**: Replace `MemorySaver` with `langgraph-checkpoint-redis` for shared state across instances.
- **LLM swap**: Change `LLM_PROVIDER=openai` and `LLM_MODEL=gpt-4o` — no code changes required.
- **More search sources**: Add a new function in `tools/search_tools.py` and call it from `web_search_node`.
- **Additional agents**: Add a node function and wire it into `core/graph.py` — the typed state contract keeps everything consistent.

---

## License

MIT — see [LICENSE](LICENSE).
