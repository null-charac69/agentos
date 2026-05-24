# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
# Install all Python dependencies (includes native extensions for
# sentence-transformers, hiredis, etc.) into a venv.
FROM python:3.11-slim AS builder

# System deps required to compile native Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create a virtual environment in /build/venv so we can copy it cleanly
RUN python -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"

# Install pip tools first (faster layer caching)
RUN pip install --upgrade pip setuptools wheel

# Copy only the manifest; leverage Docker layer cache — this layer only
# rebuilds when pyproject.toml changes, not when source code changes.
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" --extra-index-url https://pypi.org/simple

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /build/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy application source
COPY . .

# Non-root user for security
RUN groupadd -r agentos && useradd -r -g agentos agentos
RUN chown -R agentos:agentos /app
USER agentos

# Pre-download the sentence-transformers model so it's baked into the image
# (avoids a 80MB download on first request)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

# Health check: hit the /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
