"""
api/middleware.py
─────────────────
FastAPI middleware stack: CORS, structured request logging, timing headers.

Why structlog over Python's stdlib logging?
  - Outputs JSON lines in production (machine-parseable for log aggregators)
  - Outputs colorised human-readable output in development
  - Context variables (session_id, request_id) auto-bind to every log line
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every incoming request and its response time.
    Attaches a unique `request_id` header to each response for traceability.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.monotonic()

        # Bind request context to all log lines in this request
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info("request_received")

        response = await call_next(request)

        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        logger.info(
            "request_completed",
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 1),
        )

        structlog.contextvars.clear_contextvars()
        return response


def setup_cors(app) -> None:
    """Add CORS middleware. In production, replace '*' with your frontend domain."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for the application."""
    import logging
    import sys
    from config.settings import get_settings

    settings = get_settings()

    # Set stdlib logging level (langchain etc. use stdlib)
    logging.basicConfig(
        level=getattr(logging, log_level),
        stream=sys.stdout,
        format="%(message)s",
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.app_env == "production":
        # JSON output for log aggregators
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty coloured output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
