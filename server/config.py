"""Simplified configuration management."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


def _load_env_file() -> None:
    """Load .env from root directory if present."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                key, value = key.strip(), value.strip().strip("'\"")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


_load_env_file()


DEFAULT_APP_NAME = "OpenPoke Server"
DEFAULT_APP_VERSION = "0.3.0"


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


class Settings(BaseModel):
    """Application settings with lightweight env fallbacks."""

    # App metadata
    app_name: str = Field(default=DEFAULT_APP_NAME)
    app_version: str = Field(default=DEFAULT_APP_VERSION)

    # Server runtime
    server_host: str = Field(default=os.getenv("OPENPOKE_HOST", "0.0.0.0"))
    server_port: int = Field(default=_env_int("OPENPOKE_PORT", 8001))

    # LLM model selection
    interaction_agent_model: str = Field(default="anthropic/claude-sonnet-4")
    execution_agent_model: str = Field(default="anthropic/claude-sonnet-4")
    execution_agent_search_model: str = Field(default="anthropic/claude-sonnet-4")
    summarizer_model: str = Field(default="anthropic/claude-sonnet-4")
    email_classifier_model: str = Field(default="anthropic/claude-sonnet-4")

    # Credentials / integrations
    openrouter_api_key: Optional[str] = Field(default=os.getenv("OPENROUTER_API_KEY"))
    composio_gmail_auth_config_id: Optional[str] = Field(default=os.getenv("COMPOSIO_GMAIL_AUTH_CONFIG_ID"))
    composio_api_key: Optional[str] = Field(default=os.getenv("COMPOSIO_API_KEY"))

    # HTTP behaviour
    cors_allow_origins_raw: str = Field(default=os.getenv("OPENPOKE_CORS_ALLOW_ORIGINS", "*"))
    enable_docs: bool = Field(default=os.getenv("OPENPOKE_ENABLE_DOCS", "1") != "0")
    docs_url: Optional[str] = Field(default=os.getenv("OPENPOKE_DOCS_URL", "/docs"))

    # Summarisation controls
    conversation_summary_threshold: int = Field(default=100)
    conversation_summary_tail_size: int = Field(default=10)

    # Bounded execution-agent retrieval and routing
    agent_retrieval_top_k: int = Field(
        default=_env_int("OPENPOKE_AGENT_RETRIEVAL_TOP_K", 5),
        ge=1,
        le=20,
    )
    agent_retrieval_min_score: float = Field(
        default=_env_float("OPENPOKE_AGENT_RETRIEVAL_MIN_SCORE", 0.08),
        ge=0,
        le=1,
    )
    agent_route_reuse_threshold: float = Field(
        default=_env_float("OPENPOKE_AGENT_ROUTE_REUSE_THRESHOLD", 0.34),
        gt=0,
        le=1,
    )
    agent_route_ambiguity_margin: float = Field(
        default=_env_float("OPENPOKE_AGENT_ROUTE_AMBIGUITY_MARGIN", 0.12),
        gt=0,
        le=1,
    )

    @property
    def cors_allow_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_allow_origins_raw.strip() in {"", "*"}:
            return ["*"]
        return [origin.strip() for origin in self.cors_allow_origins_raw.split(",") if origin.strip()]

    @property
    def resolved_docs_url(self) -> Optional[str]:
        """Return documentation URL when docs are enabled."""
        return (self.docs_url or "/docs") if self.enable_docs else None

    @property
    def summarization_enabled(self) -> bool:
        """Flag indicating conversation summarisation is active."""
        return self.conversation_summary_threshold > 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
