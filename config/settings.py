"""
config/settings.py
──────────────────
Centralised configuration via Pydantic BaseSettings.
All values are read from environment variables (or a .env file at startup).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    secret_key: str = Field(default="insecure-dev-key-change-in-prod")

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["groq", "openai"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    openai_api_key: str = ""

    # ── Search ───────────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_episodic: str = "episodic_memory"
    qdrant_collection_knowledge: str = "knowledge_cache"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "agentos"

    # ── Agent Tuning ─────────────────────────────────────────────────────────
    max_critic_retries: int = 2
    search_results_per_task: int = 5
    retrieval_top_k: int = 3
    code_exec_timeout: int = 10

    @field_validator("groq_api_key", "openai_api_key", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if v else ""

    @property
    def is_langsmith_enabled(self) -> bool:
        return bool(self.langchain_api_key and self.langchain_tracing_v2)

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
