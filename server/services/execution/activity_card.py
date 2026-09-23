"""Structured, versioned execution-activity records and derived routing cards."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..evaluation_lab.redaction import redact_value


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False, extra="forbid", frozen=True, strict=True
    )


class ActivityRecord(_FrozenModel):
    """Structured record of an execution agent episode."""

    schema_version: Literal[1] = 1
    agent_id: UUID
    episode_id: UUID
    occurred_at: datetime
    request_intent: str
    enduring_responsibility: str
    entity_bindings: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    rationale: str
    disposition: Literal["completed", "failed", "paused", "pending_user"]
    open_obligations: tuple[str, ...] = ()
    provenance: str

    @model_validator(mode="after")
    def _validate_timezone(self) -> "ActivityRecord":
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return self


class ActivityRecordSummary(_FrozenModel):
    """Compact summary of a past execution episode."""

    episode_id: UUID
    occurred_at: datetime
    request_intent: str
    actions: tuple[str, ...] = ()
    disposition: str

    @model_validator(mode="after")
    def _validate_timezone(self) -> "ActivityRecordSummary":
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return self


class ActivityCard(_FrozenModel):
    """Compact, bounded representation of an execution agent for Jev scoring."""

    schema_version: Literal[1] = 1
    agent_id: UUID
    name: str
    purpose: str
    status: str
    created_at: datetime
    last_used_at: datetime
    episode_count: int = Field(ge=0)
    current_obligations: tuple[str, ...] = ()
    durable_summary: str = ""
    recent_episodes: tuple[ActivityRecordSummary, ...] = ()
    entity_claims: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_timezone(self) -> "ActivityCard":
        if self.created_at.tzinfo is None or self.last_used_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return self

    @property
    def digest(self) -> str:
        """SHA-256 fingerprint of the canonical card content."""
        payload = redact_value(
            self.model_dump(mode="json", exclude_computed_fields=True)
        )
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def render_text(self) -> str:
        """Render card as bounded text suitable for LLM context (approx 250-500 tokens)."""
        lines = [
            f"Agent ID: {self.agent_id}",
            f"Name: {self.name}",
            f"Purpose: {self.purpose}",
            f"Status: {self.status} (episodes: {self.episode_count})",
        ]
        if self.entity_claims:
            lines.append(f"Entities: {', '.join(self.entity_claims)}")
        if self.current_obligations:
            lines.append(f"Open obligations: {'; '.join(self.current_obligations)}")
        if self.durable_summary:
            lines.append(f"Durable summary: {self.durable_summary}")
        if self.recent_episodes:
            lines.append("Recent episodes:")
            for ep in self.recent_episodes[:3]:
                actions_str = f" [actions: {', '.join(ep.actions)}]" if ep.actions else ""
                lines.append(f"- {ep.request_intent} ({ep.disposition}){actions_str}")
        return "\n".join(lines)


class ActivityCardGenerator:
    """Extract and bound an ActivityCard from agent records and execution journal."""

    def __init__(
        self,
        *,
        agent_id: UUID,
        name: str,
        purpose: str,
        status: str,
        created_at: datetime,
        last_used_at: datetime,
        max_recent_episodes: int = 3,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.purpose = purpose
        self.status = status
        self.created_at = (
            created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
        )
        self.last_used_at = (
            last_used_at if last_used_at.tzinfo else last_used_at.replace(tzinfo=UTC)
        )
        self.max_recent_episodes = max_recent_episodes

    def generate_from_records(
        self,
        journal_entries: Sequence[Mapping[str, Any] | ActivityRecord],
        durable_summary: str = "",
    ) -> ActivityCard:
        """Generate a redacted and bounded ActivityCard."""
        summaries: list[ActivityRecordSummary] = []
        entity_claims: set[str] = set()
        open_obligations: list[str] = []

        # Process entries in reverse to get most recent first
        for raw in reversed(journal_entries):
            if isinstance(raw, ActivityRecord):
                entry = raw.model_dump(mode="json")
            else:
                entry = dict(raw)

            # Apply redaction to avoid leaking sensitive data
            safe_entry = redact_value(entry)

            ep_id_str = safe_entry.get("episode_id")
            try:
                ep_id = UUID(str(ep_id_str)) if ep_id_str else self.agent_id
            except (ValueError, TypeError):
                ep_id = self.agent_id

            raw_time = safe_entry.get("occurred_at")
            if isinstance(raw_time, datetime):
                occurred_at = (
                    raw_time if raw_time.tzinfo else raw_time.replace(tzinfo=UTC)
                )
            elif isinstance(raw_time, str):
                try:
                    occurred_at = datetime.fromisoformat(raw_time)
                    if occurred_at.tzinfo is None:
                        occurred_at = occurred_at.replace(tzinfo=UTC)
                except ValueError:
                    occurred_at = datetime.now(UTC)
            else:
                occurred_at = datetime.now(UTC)

            # Extract bounded fields
            intent = str(safe_entry.get("request_intent", safe_entry.get("intent", "")))[:200]
            disposition = str(safe_entry.get("disposition", "completed"))
            actions_raw = safe_entry.get("actions", ())
            if isinstance(actions_raw, (list, tuple)):
                actions = tuple(str(a)[:50] for a in actions_raw[:5])
            else:
                actions = ()

            if len(summaries) < self.max_recent_episodes and intent:
                summaries.append(
                    ActivityRecordSummary(
                        episode_id=ep_id,
                        occurred_at=occurred_at,
                        request_intent=intent,
                        actions=actions,
                        disposition=disposition,
                    )
                )

            # Collect entity claims
            bindings = safe_entry.get("entity_bindings", ())
            if isinstance(bindings, (list, tuple)):
                for b in bindings:
                    if b and isinstance(b, str) and not b.startswith("[REDACTED"):
                        entity_claims.add(b.strip()[:50])

            # Collect open obligations from the latest episode
            if not open_obligations:
                obs = safe_entry.get("open_obligations", ())
                if isinstance(obs, (list, tuple)):
                    open_obligations = [str(o)[:100] for o in obs if o]

        return ActivityCard(
            agent_id=self.agent_id,
            name=self.name,
            purpose=self.purpose,
            status=self.status,
            created_at=self.created_at,
            last_used_at=self.last_used_at,
            episode_count=len(journal_entries),
            current_obligations=tuple(open_obligations[:3]),
            durable_summary=durable_summary[:500],
            recent_episodes=tuple(summaries),
            entity_claims=tuple(sorted(entity_claims)[:10]),
        )


__all__ = [
    "ActivityCard",
    "ActivityCardGenerator",
    "ActivityRecord",
    "ActivityRecordSummary",
]
