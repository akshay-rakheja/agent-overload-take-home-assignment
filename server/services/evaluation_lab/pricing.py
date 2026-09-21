"""Sanitized provider pricing snapshots and exact Decimal cost estimates."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import threading
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .decimal_math import (
    exact_add,
    exact_multiply,
    exact_subtract,
    validate_exact_decimal,
)
from .models import Availability
from .usage import UsageRecord


class CostSource(str, Enum):
    PROVIDER_REPORTED = "provider_reported"
    CONSERVATIVE_ESTIMATE = "conservative_estimate"
    CRASH_RECOVERY_ESTIMATE = "crash_recovery_estimate"
    UNAVAILABLE = "unavailable"


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a decimal number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return validate_exact_decimal(result, field=field)


class PriceSchedule(BaseModel):
    """One sanitized candidate schedule used for conservative routing prices."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provider_sha256: str | None = None
    prompt_price_per_token: Decimal | None = None
    completion_price_per_token: Decimal | None = None
    cached_prompt_price_per_token: Decimal | None = None

    @field_validator(
        "prompt_price_per_token",
        "completion_price_per_token",
        "cached_prompt_price_per_token",
        mode="before",
    )
    @classmethod
    def _valid_price(cls, value: object, info):
        return None if value is None else _decimal(value, field=info.field_name)

    @model_validator(mode="before")
    @classmethod
    def _sanitize_provider(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        raw_provider = normalized.pop("provider", None)
        if raw_provider is not None:
            if not isinstance(raw_provider, str):
                raise ValueError("provider must be a text label")
            label = raw_provider.strip()
            if (
                not label
                or len(label) > 128
                or any(not character.isprintable() for character in label)
            ):
                raise ValueError("provider must be a bounded printable label")
            normalized["provider_sha256"] = hashlib.sha256(
                label.encode("utf-8")
            ).hexdigest()
        return normalized

    @field_validator("provider_sha256")
    @classmethod
    def _valid_provider_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.casefold()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("provider_sha256 must be a SHA-256 digest")
        return normalized


def _sanitize_source_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("source_url must be an absolute HTTP(S) URL")
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path or "/", "", ""))


def _canonical_response(value: Mapping[str, object]) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("model metadata must be canonical JSON") from exc
    return rendered.encode("utf-8")


def _model_record(response: Mapping[str, object], model_id: str) -> Mapping[str, object]:
    data = response.get("data")
    records: list[object]
    if isinstance(data, list):
        records = data
    elif isinstance(data, Mapping):
        records = [data]
    else:
        records = [response]
    for item in records:
        if isinstance(item, Mapping) and item.get("id") == model_id:
            return item
    raise ValueError("requested model is absent from metadata response")


def _schedule_from_mapping(
    value: Mapping[str, object], *, provider: str | None = None
) -> PriceSchedule:
    pricing = value.get("pricing")
    source = pricing if isinstance(pricing, Mapping) else value
    resolved_provider = provider
    if resolved_provider is None:
        raw_provider = value.get("provider") or value.get("name")
        resolved_provider = raw_provider if isinstance(raw_provider, str) else None
    return PriceSchedule(
        provider=resolved_provider,
        prompt_price_per_token=source.get("prompt"),
        completion_price_per_token=source.get("completion"),
        cached_prompt_price_per_token=(
            source.get("input_cache_read")
            if source.get("input_cache_read") is not None
            else source.get("cached_prompt")
        ),
    )


class PricingSnapshot(BaseModel):
    """Credential-free metadata needed to price one pinned model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = Field(default=1, ge=1, le=1)
    model_id: str
    prompt_price_per_token: Decimal | None = None
    completion_price_per_token: Decimal | None = None
    cached_prompt_price_per_token: Decimal | None = None
    context_limit: int = Field(gt=0)
    source_url: str
    retrieved_at: datetime
    response_sha256: str
    alternative_prices: tuple[PriceSchedule, ...] = ()

    @field_validator("model_id")
    @classmethod
    def _valid_model_id(cls, value: str) -> str:
        normalized = value.strip()
        if (
            not normalized
            or "/" not in normalized
            or normalized.startswith("/")
            or normalized.endswith("/")
            or any(character.isspace() for character in normalized)
        ):
            raise ValueError("model_id must be a provider/model identifier")
        return normalized

    @field_validator(
        "prompt_price_per_token",
        "completion_price_per_token",
        "cached_prompt_price_per_token",
        mode="before",
    )
    @classmethod
    def _valid_price(cls, value: object, info):
        return None if value is None else _decimal(value, field=info.field_name)

    @field_validator("source_url")
    @classmethod
    def _safe_source(cls, value: str) -> str:
        return _sanitize_source_url(value)

    @field_validator("response_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        normalized = value.casefold()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("response_sha256 must be a lowercase SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def _timezone_aware(self) -> "PricingSnapshot":
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        return self

    @classmethod
    def from_model_metadata(
        cls,
        response: Mapping[str, object],
        *,
        model_id: str,
        source_url: str,
        retrieved_at: datetime,
    ) -> "PricingSnapshot":
        canonical = _canonical_response(response)
        record = _model_record(response, model_id)
        primary = _schedule_from_mapping(record)
        context = record.get("context_length")
        if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
            raise ValueError("model metadata context_length is missing or invalid")
        alternatives: list[PriceSchedule] = []
        endpoints = record.get("endpoints")
        if isinstance(endpoints, list):
            for endpoint in endpoints:
                if isinstance(endpoint, Mapping):
                    alternatives.append(_schedule_from_mapping(endpoint))
        return cls(
            model_id=model_id,
            prompt_price_per_token=primary.prompt_price_per_token,
            completion_price_per_token=primary.completion_price_per_token,
            cached_prompt_price_per_token=primary.cached_prompt_price_per_token,
            context_limit=context,
            source_url=_sanitize_source_url(source_url),
            retrieved_at=retrieved_at,
            response_sha256=hashlib.sha256(canonical).hexdigest(),
            alternative_prices=tuple(alternatives),
        )

    def schedules(self) -> tuple[PriceSchedule, ...]:
        return (
            PriceSchedule(
                prompt_price_per_token=self.prompt_price_per_token,
                completion_price_per_token=self.completion_price_per_token,
                cached_prompt_price_per_token=self.cached_prompt_price_per_token,
            ),
            *self.alternative_prices,
        )


class CostRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    amount_usd: Decimal | None
    source: CostSource
    reason: str | None = None

    @field_validator("amount_usd", mode="before")
    @classmethod
    def _valid_amount(cls, value: object):
        return None if value is None else _decimal(value, field="amount_usd")

    @model_validator(mode="after")
    def _coherent(self) -> "CostRecord":
        if self.source is CostSource.UNAVAILABLE and self.amount_usd is not None:
            raise ValueError("unavailable cost cannot carry an amount")
        if self.source is not CostSource.UNAVAILABLE and self.amount_usd is None:
            raise ValueError("available cost requires an amount")
        return self


class ReservationEstimate(CostRecord):
    prompt_token_upper_bound: int = Field(ge=0)
    completion_token_upper_bound: int = Field(ge=0)


def _available_decimal(observation: object) -> Decimal | None:
    availability = getattr(observation, "availability", None)
    value = getattr(observation, "value", None)
    if availability is not Availability.AVAILABLE or value is None:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def calculate_cost(usage: UsageRecord, pricing: PricingSnapshot) -> CostRecord:
    """Prefer provider charges, otherwise calculate the highest exact schedule."""

    provider_cost = _available_decimal(usage.provider_cost_usd)
    if provider_cost is not None:
        return CostRecord(
            amount_usd=provider_cost,
            source=CostSource.PROVIDER_REPORTED,
        )

    prompt = _available_decimal(usage.prompt_tokens)
    completion = _available_decimal(usage.completion_tokens)
    cached = _available_decimal(usage.cached_tokens)
    if prompt is None or completion is None:
        return CostRecord(
            amount_usd=None,
            source=CostSource.UNAVAILABLE,
            reason="token usage required for estimation is unavailable",
        )
    if cached is not None and cached > prompt:
        return CostRecord(
            amount_usd=None,
            source=CostSource.UNAVAILABLE,
            reason="cached tokens exceed prompt tokens",
        )

    totals: list[Decimal] = []
    for schedule in pricing.schedules():
        if (
            schedule.prompt_price_per_token is None
            or schedule.completion_price_per_token is None
        ):
            return CostRecord(
                amount_usd=None,
                source=CostSource.UNAVAILABLE,
                reason="required token pricing is unavailable",
            )
        cached_price = schedule.cached_prompt_price_per_token
        if cached is None:
            prompt_cost = exact_multiply(
                prompt,
                max(
                    schedule.prompt_price_per_token,
                    cached_price or schedule.prompt_price_per_token,
                ),
            )
        else:
            applicable_cached_price = (
                cached_price
                if cached_price is not None
                else schedule.prompt_price_per_token
            )
            prompt_cost = exact_add(
                exact_multiply(
                    exact_subtract(prompt, cached),
                    schedule.prompt_price_per_token,
                ),
                exact_multiply(
                    cached,
                    applicable_cached_price,
                ),
            )
        totals.append(
            exact_add(
                prompt_cost,
                exact_multiply(completion, schedule.completion_price_per_token),
            )
        )
    if not totals:
        return CostRecord(
            amount_usd=None,
            source=CostSource.UNAVAILABLE,
            reason="required token pricing is unavailable",
        )
    return CostRecord(
        amount_usd=max(totals),
        source=CostSource.CONSERVATIVE_ESTIMATE,
    )


def conservative_reservation(
    *, serialized_prompt: str, max_tokens: int, pricing: PricingSnapshot
) -> ReservationEstimate:
    """Price a call using prompt characters as a token upper bound."""

    if not isinstance(serialized_prompt, str):
        raise TypeError("serialized_prompt must be text")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    prompt_tokens = len(serialized_prompt)
    prompt_prices: list[Decimal] = []
    completion_prices: list[Decimal] = []
    for schedule in pricing.schedules():
        if (
            schedule.prompt_price_per_token is None
            or schedule.completion_price_per_token is None
        ):
            return ReservationEstimate(
                amount_usd=None,
                source=CostSource.UNAVAILABLE,
                reason="required normal token pricing is unavailable",
                prompt_token_upper_bound=prompt_tokens,
                completion_token_upper_bound=max_tokens,
            )
        prompt_prices.append(
            max(
                schedule.prompt_price_per_token,
                schedule.cached_prompt_price_per_token
                or schedule.prompt_price_per_token,
            )
        )
        completion_prices.append(schedule.completion_price_per_token)
    if not prompt_prices or not completion_prices:
        return ReservationEstimate(
            amount_usd=None,
            source=CostSource.UNAVAILABLE,
            reason="required token pricing is unavailable",
            prompt_token_upper_bound=prompt_tokens,
            completion_token_upper_bound=max_tokens,
        )
    return ReservationEstimate(
        amount_usd=exact_add(
            exact_multiply(Decimal(prompt_tokens), max(prompt_prices)),
            exact_multiply(Decimal(max_tokens), max(completion_prices)),
        ),
        source=CostSource.CONSERVATIVE_ESTIMATE,
        prompt_token_upper_bound=prompt_tokens,
        completion_token_upper_bound=max_tokens,
    )


_STORE_LOCKS: dict[Path, threading.RLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    with _STORE_LOCKS_GUARD:
        return _STORE_LOCKS.setdefault(path.absolute(), threading.RLock())


class PricingSnapshotStore:
    """Atomic descriptor-relative storage for sanitized pricing snapshots."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        parts = self.root.parts
        if ".." in parts:
            raise ValueError("pricing root must not contain path traversal")
        indexes = [index for index, part in enumerate(parts) if part == ".lab"]
        if not indexes or indexes[-1] == len(parts) - 1:
            raise ValueError("pricing root must be below the ignored .lab directory")
        index = indexes[-1]
        self._trusted_parent = Path(*parts[:index]) if parts[:index] else Path(".")
        self._components = parts[index:]
        self._lock = _thread_lock(self.root)

    @staticmethod
    def _name(model_id: str) -> str:
        return f"{hashlib.sha256(model_id.encode('utf-8')).hexdigest()}.json"

    def _open_root(self, *, create: bool) -> int | None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._trusted_parent, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("pricing root must not contain a symlink") from exc
            raise
        try:
            for component in self._components:
                if create:
                    try:
                        os.mkdir(component, 0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        os.close(descriptor)
                        return None
                    raise
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError("pricing root must not contain a symlink") from exc
                    raise
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @contextmanager
    def _locked_root(self, *, exclusive: bool, create: bool) -> Iterator[int | None]:
        with self._lock:
            descriptor = self._open_root(create=create)
            if descriptor is None:
                yield None
                return
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                yield descriptor
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def write(self, snapshot: PricingSnapshot) -> Path:
        payload = snapshot.model_dump_json(indent=None).encode("utf-8") + b"\n"
        name = self._name(snapshot.model_id)
        with self._locked_root(exclusive=True, create=True) as root_fd:
            assert root_fd is not None
            try:
                existing = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ValueError("pricing snapshot path must be a regular file")
            temporary = f".{name}.{secrets.token_hex(8)}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600, dir_fd=root_fd)
            try:
                view = memoryview(payload)
                while view:
                    view = view[os.write(descriptor, view) :]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.replace(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
                os.fsync(root_fd)
            except BaseException:
                try:
                    os.unlink(temporary, dir_fd=root_fd)
                except FileNotFoundError:
                    pass
                raise
        return self.root / name

    def read(self, model_id: str) -> PricingSnapshot | None:
        name = self._name(model_id)
        with self._locked_root(exclusive=False, create=False) as root_fd:
            if root_fd is None:
                return None
            try:
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("pricing snapshot path must be a regular file")
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
            try:
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 64 * 1024):
                    chunks.append(chunk)
            finally:
                os.close(descriptor)
        snapshot = PricingSnapshot.model_validate_json(b"".join(chunks))
        if snapshot.model_id != model_id:
            raise ValueError("pricing snapshot model identity mismatch")
        return snapshot


__all__ = [
    "CostRecord",
    "CostSource",
    "PriceSchedule",
    "PricingSnapshot",
    "PricingSnapshotStore",
    "ReservationEstimate",
    "calculate_cost",
    "conservative_reservation",
]
