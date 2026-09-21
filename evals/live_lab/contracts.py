"""Typed contracts shared by live-lab fixture and state tooling."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedAction(str, Enum):
    REUSE = "reuse"
    CREATE_NEW = "create_new"
    ABSTAIN = "abstain"


class JournalEntry(_FrozenModel):
    tag: Literal["agent_request", "agent_action", "tool_response", "agent_response"]
    timestamp: str
    payload: str


class LogicalAgent(_FrozenModel):
    logical_id: str
    name: str
    purpose: str
    aliases: tuple[str, ...] = ()
    status: Literal["hot", "dormant", "archived"] = "hot"
    memory_summary: str = ""
    journal_entries: tuple[JournalEntry, ...] = ()


class ScenarioExpectation(_FrozenModel):
    scenario_id: str
    expected_action: ExpectedAction
    expected_logical_identity: str | None = None
    expected_gmail_fact_ids: tuple[str, ...] = ()


class FixtureManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    seed: int
    roster_size: Literal[10, 100, 500, 1000]
    agents: tuple[LogicalAgent, ...]
    scenarios: tuple[ScenarioExpectation, ...]


class StateFingerprint(_FrozenModel):
    schema_version: Literal[1] = 1
    file_sha256: dict[str, str]
    roster_sha256: str
    roster_count: int = Field(ge=0)
    logical_identity_digest: str
    raw_journal_digest: str
    journal_bytes: int = Field(ge=0)
    journal_entries: int = Field(ge=0)
    sentinel_checks: dict[str, bool]


class StateSnapshot(_FrozenModel):
    snapshot_dir: str
    fingerprint: StateFingerprint


class EquivalenceReport(_FrozenModel):
    equivalent: bool
    checks: dict[str, bool]
    differences: tuple[str, ...] = ()
