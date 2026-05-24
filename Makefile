.PHONY: install dev test lint format clean docker-up docker-down docker-logs init-qdrant

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Development ───────────────────────────────────────────────────────────────
dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=html && open htmlcov/index.html

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy . --ignore-missing-imports

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f agentos

docker-build:
	docker compose -f docker/docker-compose.yml build --no-cache

# ── Initialisation ────────────────────────────────────────────────────────────
init-qdrant:
	python scripts/init_qdrant.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache dist build

# ── Demo ──────────────────────────────────────────────────────────────────────
demo:
	@echo "Sending research query to AgentOS..."
	curl -N -X POST http://localhost:8000/api/v1/research \
		-H "Content-Type: application/json" \
		-d '{"query": "What are the latest breakthroughs in quantum computing in 2024?"}' \
		--no-buffer
