"""Append-only, baseline-native observation records without enhanced semantics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObservedToolCall(_FrozenModel):
    name: str
    arguments: dict[str, Any]
    raw_arguments_sha256: str | None = None
    malformed: bool = False


class RawModelCall(_FrozenModel):
    component: Literal["interaction", "execution", "email_search", "summarizer", "classifier"]
    model: str | None = None
    elapsed_ms: float = Field(ge=0)
    request_sha256: str
    response_sha256: str | None = None
    message_count: int = Field(default=0, ge=0)
    tool_names: tuple[str, ...] = ()
    response_choice_count: int | None = Field(default=None, ge=0)
    response_tool_call_count: int | None = Field(default=None, ge=0)
    provider: str | None = None
    generation: dict[str, Any] = Field(default_factory=dict)
    context_limit: int | None = Field(default=None, ge=1)
    requested_seed: int | None = None
    provider_seed: int | None = None
    seed_acknowledged: bool | None = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=0, ge=0)
    failover: bool | None = None
    timeout: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0)
    rate_limit: dict[str, Any] = Field(default_factory=dict)
    malformed_tool_calls: int = Field(default=0, ge=0)
    usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None


class ObservedError(_FrozenModel):
    phase: str
    code: str
    message: str
    late: bool = False
    partial: bool = False


class StateFingerprint(_FrozenModel):
    roster: tuple[str, ...]
    journal_sha256: dict[str, str]
    journal_bytes: dict[str, int]
    journal_complete: dict[str, bool] = Field(default_factory=dict)
    journal_contents: dict[str, bytes] = Field(default_factory=dict, exclude=True, repr=False)


class BaselineInference(_FrozenModel):
    action: Literal["reuse", "create_new", "abstain", "unobservable"]
    name: str | None
    reason: str
    failed: bool = False


class BaselineObservation(_FrozenModel):
    run_id: str
    prompt_xml_sha256: str
    prompt_characters: int = Field(ge=0)
    exposed_names: tuple[str, ...]
    roster_before: tuple[str, ...]
    roster_after: tuple[str, ...]
    journal_hashes_before: dict[str, str]
    journal_hashes_after: dict[str, str]
    inferred_action: Literal["reuse", "create_new", "abstain", "unobservable"]
    inferred_name: str | None
    inference_reason: str
    final_response: str | None
    raw_model_calls: tuple[RawModelCall, ...]
    errors: tuple[ObservedError, ...]


class BaselineTurnRequest(_FrozenModel):
    run_id: str
    base_url: str = "http://127.0.0.1:8001/api/v1"
    data_dir: str
    event_path: str
    user_message: str
    timeout_seconds: float = Field(default=10.0, gt=0)
    poll_interval_seconds: float = Field(default=0.05, gt=0)
    late_grace_seconds: float = Field(default=0.0, ge=0)


__all__ = [
    "BaselineInference",
    "BaselineObservation",
    "BaselineTurnRequest",
    "ObservedError",
    "ObservedToolCall",
    "RawModelCall",
    "StateFingerprint",
]
