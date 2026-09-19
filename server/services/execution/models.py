"""Persistent identity models for execution agents."""

from __future__ import annotations

import unicodedata
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


AGENT_SCHEMA_VERSION = 1


class AgentStatus(str, Enum):
    """Lifecycle state used to control routing visibility and ranking."""

    HOT = "hot"
    DORMANT = "dormant"
    ARCHIVED = "archived"


def normalize_agent_text(value: str) -> str:
    """Normalize human text for matching while retaining display text elsewhere."""

    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    expanded = without_marks.replace("&", " and ")

    normalized: list[str] = []
    for char in expanded:
        if char.isalnum() or char.isspace():
            normalized.append(char)
        elif char in {"'", "’"}:
            continue
        else:
            normalized.append(" ")
    return " ".join("".join(normalized).split())


class AgentRecord(BaseModel):
    """Immutable, versioned directory record for one persistent identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: UUID
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    status: AgentStatus = AgentStatus.HOT
    created_at: datetime
    last_used_at: datetime
    use_count: int = Field(default=0, ge=0)
    memory_summary: str = ""
    schema_version: int = AGENT_SCHEMA_VERSION

    @field_validator("name", "purpose")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("aliases")
    @classmethod
    def preserve_non_blank_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(alias.strip() for alias in values if alias.strip())

    @field_validator("created_at", "last_used_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @field_validator("schema_version")
    @classmethod
    def supported_schema(cls, value: int) -> int:
        if value != AGENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported agent schema version: {value}")
        return value

    @property
    def normalized_name(self) -> str:
        return normalize_agent_text(self.name)

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        return tuple(normalize_agent_text(alias) for alias in self.aliases)

