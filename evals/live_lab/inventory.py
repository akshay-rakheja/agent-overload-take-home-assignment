"""Sanitized exploratory natural-inbox inventory."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .fixture_email import SUBJECT_PREFIX
from .scenarios import ScenarioTrack


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InventoryMarker(str, Enum):
    FIXTURE = "fixture"
    NATURAL = "natural"


class InventoryCategory(str, Enum):
    SECURITY = "security"
    ENGAGEMENT = "engagement"
    NEWSLETTER = "newsletter"
    RECEIPT = "receipt"
    INVOICE = "invoice"
    OTHER = "other"


class InventoryItem(_FrozenModel):
    category: InventoryCategory
    subject_sha256: str
    date_range: str
    marker: InventoryMarker

    @field_validator("subject_sha256")
    @classmethod
    def _subject_hash(cls, value: str) -> str:
        normalized = value.casefold()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("subject_sha256 must be a SHA-256 digest")
        return normalized

    @field_validator("date_range")
    @classmethod
    def _date_range(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(value, "%Y-%m")
        except ValueError as exc:
            raise ValueError("date_range must be YYYY-MM") from exc
        return parsed.strftime("%Y-%m")


def _aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


class InventoryWindow(_FrozenModel):
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _timezone(cls, value: datetime, info) -> datetime:
        return _aware(value, name=info.field_name)

    @model_validator(mode="after")
    def _ordered(self) -> "InventoryWindow":
        if self.start > self.end:
            raise ValueError("inventory window start must not follow end")
        return self


class InventoryDrift(_FrozenModel):
    baseline_count: int
    enhanced_count: int
    added_subject_hashes: tuple[str, ...]
    removed_subject_hashes: tuple[str, ...]

    @field_validator("baseline_count", "enhanced_count")
    @classmethod
    def _counts(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("drift counts must be non-negative integers")
        return value

    @field_validator("added_subject_hashes", "removed_subject_hashes")
    @classmethod
    def _hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.casefold() for item in value)
        if list(normalized) != sorted(set(normalized)):
            raise ValueError("drift hashes must be sorted and unique")
        if any(
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
            for item in normalized
        ):
            raise ValueError("drift values must contain only SHA-256 digests")
        return normalized


class NaturalInventory(_FrozenModel):
    schema_version: Literal[1] = 1
    track: Literal[ScenarioTrack.NATURAL] = ScenarioTrack.NATURAL
    exploratory: Literal[True] = True
    include_in_controlled_aggregates: Literal[False] = False
    snapshot_at: datetime
    window: InventoryWindow
    drift: InventoryDrift
    items: tuple[InventoryItem, ...]

    @field_validator("snapshot_at")
    @classmethod
    def _snapshot(cls, value: datetime) -> datetime:
        return _aware(value, name="snapshot_at")


def _subject_hash(subject: str) -> str:
    normalized = " ".join(subject.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_date(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("message date must be ISO-8601") from exc
    else:
        raise ValueError("message date is required")
    return _aware(parsed, name="message date")


def _category(subject: str) -> InventoryCategory:
    normalized = subject.casefold()
    if any(term in normalized for term in ("security", "login", "sign-in")):
        return InventoryCategory.SECURITY
    if any(term in normalized for term in ("engagement", "likes", "comments")):
        return InventoryCategory.ENGAGEMENT
    if any(term in normalized for term in ("newsletter", "bulletin")):
        return InventoryCategory.NEWSLETTER
    if "receipt" in normalized:
        return InventoryCategory.RECEIPT
    if "invoice" in normalized:
        return InventoryCategory.INVOICE
    return InventoryCategory.OTHER


def _validated_hashes(values: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(str(value).casefold() for value in values)
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in normalized
    ):
        raise ValueError("drift inputs must be sanitized SHA-256 subject hashes")
    return normalized


def sanitize_inventory(
    messages: Sequence[Mapping[str, object]],
    *,
    snapshot_at: datetime,
    window_start: datetime,
    window_end: datetime,
    baseline_subject_hashes: Iterable[str] = (),
    enhanced_subject_hashes: Iterable[str] = (),
) -> NaturalInventory:
    """Reduce live metadata to non-reversible exploratory inventory facts."""

    snapshot = _aware(snapshot_at, name="snapshot_at")
    window = InventoryWindow(start=window_start, end=window_end)
    baseline = _validated_hashes(baseline_subject_hashes)
    enhanced = _validated_hashes(enhanced_subject_hashes)
    sanitized: dict[tuple[str, str, InventoryMarker], InventoryItem] = {}
    for raw in messages:
        subject = raw.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("inventory message subject must be nonempty text")
        occurred_at = _parse_date(raw.get("date"))
        if occurred_at < window.start or occurred_at > window.end:
            continue
        marker = (
            InventoryMarker.FIXTURE
            if subject.startswith(SUBJECT_PREFIX)
            else InventoryMarker.NATURAL
        )
        item = InventoryItem(
            category=_category(subject),
            subject_sha256=_subject_hash(subject),
            date_range=occurred_at.strftime("%Y-%m"),
            marker=marker,
        )
        sanitized[(item.subject_sha256, item.date_range, item.marker)] = item
    items = tuple(
        sorted(
            sanitized.values(),
            key=lambda item: (item.category.value, item.date_range, item.subject_sha256, item.marker.value),
        )
    )
    return NaturalInventory(
        snapshot_at=snapshot,
        window=window,
        drift=InventoryDrift(
            baseline_count=len(baseline),
            enhanced_count=len(enhanced),
            added_subject_hashes=tuple(sorted(enhanced - baseline)),
            removed_subject_hashes=tuple(sorted(baseline - enhanced)),
        ),
        items=items,
    )


def select_exploratory(
    inventory: NaturalInventory,
    *,
    limit: int,
    categories: Iterable[InventoryCategory | str] = (),
) -> tuple[InventoryItem, ...]:
    """Select stable natural-only cases; never feed controlled aggregates."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("exploratory selection limit must be positive")
    selected_categories = {
        value if isinstance(value, InventoryCategory) else InventoryCategory(value)
        for value in categories
    }
    candidates = [
        item
        for item in inventory.items
        if item.marker is InventoryMarker.NATURAL
        and (not selected_categories or item.category in selected_categories)
    ]
    return tuple(candidates[:limit])
