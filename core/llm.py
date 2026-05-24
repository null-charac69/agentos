"""
core/llm.py
───────────
LLM factory — returns a LangChain BaseChatModel based on the configured provider.

Why a factory instead of instantiating directly in agents?
  - Single place to swap models (change one env var, all agents update)
  - Temperature / retry / callback config lives here, not scattered in agents
  - Easier to mock in tests

Provider choices:
  - groq  → ChatGroq (default; free tier, ~750 tok/s on Llama 3.3 70B)
  - openai → ChatOpenAI (optional; better quality, costs money)
"""

from __future__ import annotations

import os
from functools import lru_cache

import structlog
from langchain_core.language_models import BaseChatModel

from config.settings import get_settings

logger = structlog.get_logger(__name__)


def _configure_langsmith() -> None:
    """Activate LangSmith tracing if credentials are present."""
    settings = get_settings()
    if settings.is_langsmith_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("langsmith_tracing_enabled", project=settings.langchain_project)
    else:
        logger.debug("langsmith_tracing_disabled")


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """
    Return a cached LLM instance.

    Args:
        temperature: Sampling temperature. Default 0.1 keeps outputs
                     deterministic enough for structured tasks like planning
                     and critiquing, while still allowing some creativity.

    Returns:
        A LangChain-compatible chat model.
    """
    _configure_langsmith()
    settings = get_settings()

    provider = settings.llm_provider.lower()
    model = settings.llm_model

    logger.info("initialising_llm", provider=provider, model=model)

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com"
            )
        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            temperature=temperature,
            max_retries=3,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set LLM_PROVIDER=groq to use the free Groq tier instead."
            )
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_retries=3,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Valid options: 'groq', 'openai'."
    )
