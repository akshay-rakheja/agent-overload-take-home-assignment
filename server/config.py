"""Simplified configuration management."""

import os
import math
from enum import Enum
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
MAX_AGENT_CANDIDATES = 5
DEFAULT_MODEL_ID = "anthropic/claude-sonnet-4"


class ModelRole(str, Enum):
    """Every model-backed role whose configuration must be comparable."""

    INTERACTION = "interaction"
    EXECUTION = "execution"
    EMAIL_SEARCH = "email_search"
    SUMMARIZER = "summarizer"
    CLASSIFIER = "classifier"


class ModelCallConfig(BaseModel):
    """Validated generation and transport policy for one model call."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, protected_namespaces=()
    )

    model_id: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=1000, gt=0)
    seed: int | None = None
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=0, ge=0, le=10)

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        normalized = value.strip()
        if (
            not normalized
            or normalized.startswith("/")
            or normalized.endswith("/")
            or "/" not in normalized
            or any(character.isspace() for character in normalized)
        ):
            raise ValueError("model_id must be a provider/model identifier")
        return normalized

    @field_validator("temperature", "top_p", "timeout_seconds")
    @classmethod
    def _finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("model settings must be finite")
        return value

    def explicit_payload_fields(self) -> dict[str, int | float]:
        """Return only generation fields explicitly selected by the caller."""

        compatible = ("temperature", "top_p", "max_tokens", "seed")
        return {
            field: getattr(self, field)
            for field in compatible
            if field in self.model_fields_set and getattr(self, field) is not None
        }


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


def _env_bool(name: str, fallback: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _first_env(*names: str, fallback: str) -> str:
    for name in names:
        value = _env_optional(name)
        if value is not None:
            return value
    return fallback


def _env_optional_int(name: str) -> int | None:
    value = _env_optional(name)
    return int(value) if value is not None else None


def _env_optional_float(name: str) -> float | None:
    value = _env_optional(name)
    return float(value) if value is not None else None


def is_loopback_host(value: str) -> bool:
    """Return whether a configured bind/client host is strictly local."""

    normalized = value.strip().strip("[]")
    if normalized.casefold() == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


class Settings(BaseModel):
    """Application settings with lightweight env fallbacks."""

    model_config = ConfigDict(protected_namespaces=())

    # App metadata
    app_name: str = Field(default=DEFAULT_APP_NAME)
    app_version: str = Field(default=DEFAULT_APP_VERSION)

    # Server runtime
    server_host: str = Field(default=os.getenv("OPENPOKE_HOST", "0.0.0.0"))
    server_port: int = Field(default=_env_int("OPENPOKE_PORT", 8001))

    # LLM model selection
    interaction_agent_model: str = Field(
        default_factory=lambda: _first_env(
            "OPENPOKE_INTERACTION_MODEL",
            "OPENPOKE_INTERACTION_AGENT_MODEL",
            fallback=DEFAULT_MODEL_ID,
        )
    )
    execution_agent_model: str = Field(
        default_factory=lambda: _first_env(
            "OPENPOKE_EXECUTION_MODEL",
            "OPENPOKE_EXECUTION_AGENT_MODEL",
            fallback=DEFAULT_MODEL_ID,
        )
    )
    execution_agent_search_model: str = Field(
        default_factory=lambda: _first_env(
            "OPENPOKE_EMAIL_SEARCH_MODEL",
            "OPENPOKE_EXECUTION_AGENT_SEARCH_MODEL",
            fallback=DEFAULT_MODEL_ID,
        )
    )
    summarizer_model: str = Field(
        default_factory=lambda: _first_env(
            "OPENPOKE_SUMMARIZER_MODEL", fallback=DEFAULT_MODEL_ID
        )
    )
    email_classifier_model: str = Field(
        default_factory=lambda: _first_env(
            "OPENPOKE_CLASSIFIER_MODEL",
            "OPENPOKE_EMAIL_CLASSIFIER_MODEL",
            fallback=DEFAULT_MODEL_ID,
        )
    )

    # Optional ordinary-mode generation overrides. Their absence preserves the
    # historical request payload exactly.
    model_temperature: float | None = Field(
        default_factory=lambda: _env_optional_float("OPENPOKE_MODEL_TEMPERATURE")
    )
    model_top_p: float | None = Field(
        default_factory=lambda: _env_optional_float("OPENPOKE_MODEL_TOP_P")
    )
    model_max_tokens: int | None = Field(
        default_factory=lambda: _env_optional_int("OPENPOKE_MODEL_MAX_TOKENS")
    )
    model_seed: int | None = Field(
        default_factory=lambda: _env_optional_int("OPENPOKE_MODEL_SEED")
    )
    model_timeout_seconds: float | None = Field(
        default_factory=lambda: _env_optional_float("OPENPOKE_MODEL_TIMEOUT_SECONDS")
    )
    model_max_retries: int | None = Field(
        default_factory=lambda: _env_optional_int("OPENPOKE_MODEL_MAX_RETRIES")
    )

    # Credentials / integrations
    openrouter_api_key: Optional[str] = Field(default=os.getenv("OPENROUTER_API_KEY"))
    composio_gmail_auth_config_id: Optional[str] = Field(default=os.getenv("COMPOSIO_GMAIL_AUTH_CONFIG_ID"))
    composio_api_key: Optional[str] = Field(default=os.getenv("COMPOSIO_API_KEY"))

    # Evaluation Lab safety boundary
    lab_enabled: bool = Field(
        default_factory=lambda: _env_bool("OPENPOKE_LAB_ENABLED")
    )
    lab_composio_user_id: Optional[str] = Field(
        default_factory=lambda: _env_optional("OPENPOKE_LAB_COMPOSIO_USER_ID")
    )
    lab_trace_root: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "OPENPOKE_LAB_TRACE_ROOT",
                str(Path(__file__).resolve().parent.parent / ".lab" / "traces"),
            )
        )
    )
    lab_revision: Optional[str] = Field(
        default_factory=lambda: _env_optional("OPENPOKE_LAB_REVISION")
    )
    lab_model: Optional[str] = Field(
        default_factory=lambda: _env_optional("OPENPOKE_LAB_MODEL")
    )
    lab_temperature: float = Field(
        default_factory=lambda: _env_float("OPENPOKE_LAB_TEMPERATURE", 0.0)
    )
    lab_top_p: float = Field(
        default_factory=lambda: _env_float("OPENPOKE_LAB_TOP_P", 1.0)
    )
    lab_max_tokens: int = Field(
        default_factory=lambda: _env_int("OPENPOKE_LAB_MAX_TOKENS", 1000)
    )
    lab_seed: int | None = Field(
        default_factory=lambda: _env_optional_int("OPENPOKE_LAB_SEED")
    )
    lab_seed_compatible: bool = Field(
        default_factory=lambda: _env_bool("OPENPOKE_LAB_SEED_COMPATIBLE")
    )
    lab_timeout_seconds: float = Field(
        default_factory=lambda: _env_float("OPENPOKE_LAB_TIMEOUT_SECONDS", 60.0)
    )
    lab_max_retries: int = Field(
        default_factory=lambda: _env_int("OPENPOKE_LAB_MAX_RETRIES", 0)
    )

    # HTTP behaviour
    cors_allow_origins_raw: str = Field(default=os.getenv("OPENPOKE_CORS_ALLOW_ORIGINS", "*"))
    enable_docs: bool = Field(default=os.getenv("OPENPOKE_ENABLE_DOCS", "1") != "0")
    docs_url: Optional[str] = Field(default=os.getenv("OPENPOKE_DOCS_URL", "/docs"))

    # Summarisation controls
    conversation_summary_threshold: int = Field(default=100)
    conversation_summary_tail_size: int = Field(default=10)

    # Bounded execution-agent retrieval and routing
    agent_retrieval_top_k: int = Field(
        default=_env_int("OPENPOKE_AGENT_RETRIEVAL_TOP_K", MAX_AGENT_CANDIDATES),
        ge=1,
        le=MAX_AGENT_CANDIDATES,
        validate_default=True,
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
    agent_routing_context_max_characters: int = Field(
        default=_env_int("OPENPOKE_AGENT_ROUTING_CONTEXT_MAX_CHARACTERS", 4_000),
        ge=200,
    )
    execution_context_max_recent_episodes: int = Field(
        default=_env_int("OPENPOKE_EXECUTION_CONTEXT_MAX_RECENT_EPISODES", 8),
        ge=1,
        le=100,
    )
    execution_context_max_characters: int = Field(
        default=_env_int("OPENPOKE_EXECUTION_CONTEXT_MAX_CHARACTERS", 12_000),
        ge=200,
    )

    @model_validator(mode="after")
    def validate_lab_identity(self) -> "Settings":
        """Require one stable opaque Composio identity whenever lab mode is active."""

        normalized = (self.lab_composio_user_id or "").strip()
        self.lab_composio_user_id = normalized or None
        if self.lab_enabled and not normalized:
            raise ValueError(
                "OPENPOKE_LAB_COMPOSIO_USER_ID is required when Evaluation Lab mode is enabled"
            )
        if self.lab_enabled and not is_loopback_host(self.server_host):
            raise ValueError(
                "OPENPOKE_HOST must be a loopback host when Evaluation Lab mode is enabled"
            )
        if self.lab_enabled and self.lab_model is not None:
            self.interaction_agent_model = self.lab_model
            self.execution_agent_model = self.lab_model
            self.execution_agent_search_model = self.lab_model
            self.summarizer_model = self.lab_model
            self.email_classifier_model = self.lab_model
        for role in ModelRole:
            self.model_call_config(role)
        return self

    def model_call_config(self, role: ModelRole | str) -> ModelCallConfig:
        """Resolve one role without allowing lab measurements to drift."""

        resolved_role = role if isinstance(role, ModelRole) else ModelRole(role)
        role_models = {
            ModelRole.INTERACTION: self.interaction_agent_model,
            ModelRole.EXECUTION: self.execution_agent_model,
            ModelRole.EMAIL_SEARCH: self.execution_agent_search_model,
            ModelRole.SUMMARIZER: self.summarizer_model,
            ModelRole.CLASSIFIER: self.email_classifier_model,
        }
        if self.lab_enabled and self.lab_model is not None:
            kwargs: dict[str, Any] = {
                "model_id": self.lab_model,
                "temperature": self.lab_temperature,
                "top_p": self.lab_top_p,
                "max_tokens": self.lab_max_tokens,
                "timeout_seconds": self.lab_timeout_seconds,
                "max_retries": self.lab_max_retries,
            }
            if self.lab_seed is not None and self.lab_seed_compatible:
                kwargs["seed"] = self.lab_seed
            return ModelCallConfig(**kwargs)

        kwargs = {"model_id": role_models[resolved_role]}
        for field_name, value in (
            ("temperature", self.model_temperature),
            ("top_p", self.model_top_p),
            ("max_tokens", self.model_max_tokens),
            ("seed", self.model_seed),
            ("timeout_seconds", self.model_timeout_seconds),
            ("max_retries", self.model_max_retries),
        ):
            if value is not None:
                kwargs[field_name] = value
        return ModelCallConfig(**kwargs)

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
