"""
cache/redis_cache.py
─────────────────────
Redis-backed caching and session state for AgentOS.

Two responsibilities:

1. Result Cache (Key-Value)
   - Key:   SHA-256 hash of the normalised query string
   - Value: JSON-encoded final research report
   - TTL:   Configurable (default 1 hour)
   - Benefit: Identical queries within the TTL window skip the entire
              agent pipeline and return in <50ms

2. Session Store (progress tracking)
   - Key:   "session:{session_id}:progress"
   - Value: JSON with agent progress events
   - TTL:   30 minutes (sessions are ephemeral)
   - Benefit: Allows the SSE stream to be resumed if the client disconnects

Why Redis and not an in-memory dict?
   - Survives app restarts and horizontal scaling
   - Built-in TTL support (no manual expiry bookkeeping)
   - hiredis C extension makes it very fast (included via redis[hiredis])
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
import redis.asyncio as aioredis
from redis.asyncio import Redis

from config.settings import get_settings

logger = structlog.get_logger(__name__)

# ─── Client factory ──────────────────────────────────────────────────────────

_redis_client: Optional[Redis] = None


async def get_redis() -> Redis:
    """Return a shared async Redis connection (lazy init)."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        logger.info("redis_client_initialised", url=settings.redis_url)
    return _redis_client


async def close_redis() -> None:
    """Gracefully close the Redis connection on app shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("redis_client_closed")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _query_cache_key(query: str) -> str:
    """
    Produce a stable cache key from a query string.
    Normalise (lowercase, strip) before hashing so minor variations hit the cache.
    """
    normalised = query.strip().lower()
    digest = hashlib.sha256(normalised.encode()).hexdigest()[:16]
    return f"agentos:result:{digest}"


def _session_key(session_id: str) -> str:
    return f"agentos:session:{session_id}"


# ─── Result Cache ─────────────────────────────────────────────────────────────

async def get_cached_result(query: str) -> Optional[dict[str, Any]]:
    """
    Return a cached research result or None if cache miss.
    """
    redis = await get_redis()
    key = _query_cache_key(query)

    try:
        raw = await redis.get(key)
        if raw:
            logger.info("cache_hit", key=key)
            return json.loads(raw)
    except Exception as exc:
        logger.warning("cache_get_failed", key=key, error=str(exc))

    return None


async def cache_result(query: str, result: dict[str, Any]) -> None:
    """
    Store a research result in Redis with TTL.
    """
    settings = get_settings()
    redis = await get_redis()
    key = _query_cache_key(query)

    try:
        await redis.setex(
            name=key,
            time=settings.redis_ttl_seconds,
            value=json.dumps(result, default=str),
        )
        logger.info("result_cached", key=key, ttl=settings.redis_ttl_seconds)
    except Exception as exc:
        logger.warning("cache_set_failed", key=key, error=str(exc))


# ─── Session Store ────────────────────────────────────────────────────────────

async def update_session_progress(
    session_id: str,
    event: str,
    data: dict[str, Any],
) -> None:
    """
    Append a progress event to the session's event log.
    """
    redis = await get_redis()
    key = _session_key(session_id)

    event_entry = {
        "event": event,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Use a Redis list as an append-only event log
        await redis.rpush(key, json.dumps(event_entry))
        await redis.expire(key, 1800)  # 30-min TTL
    except Exception as exc:
        logger.warning("session_update_failed", session_id=session_id, error=str(exc))


async def get_session_progress(session_id: str) -> list[dict[str, Any]]:
    """
    Return all progress events for a session (for reconnect support).
    """
    redis = await get_redis()
    key = _session_key(session_id)

    try:
        raw_events = await redis.lrange(key, 0, -1)
        return [json.loads(e) for e in raw_events]
    except Exception as exc:
        logger.warning("session_get_failed", session_id=session_id, error=str(exc))
        return []


# ─── Health check ─────────────────────────────────────────────────────────────

async def redis_ping() -> bool:
    """Return True if Redis is reachable."""
    try:
        redis = await get_redis()
        await redis.ping()
        return True
    except Exception:
        return False
