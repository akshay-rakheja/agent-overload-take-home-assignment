"""Privacy-safe natural-inbox inventory contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.live_lab.inventory import (
    InventoryMarker,
    NaturalInventory,
    sanitize_inventory,
    select_exploratory,
)


def _sha(subject: str) -> str:
    normalized = " ".join(subject.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _inventory() -> NaturalInventory:
    messages = [
        {
            "id": "gmail-live-id-001",
            "thread_id": "thread-live-001",
            "subject": "Unrelated login security warning",
            "from": "person@example.com",
            "to": "owner@example.com",
            "date": "2026-09-18T04:12:00+00:00",
            "snippet": "private customer words",
            "body": "Bearer super-secret-value",
            "tracking_url": "https://tracker.invalid/click?id=customer-99",
        },
        {
            "id": "gmail-live-id-002",
            "subject": "Monthly creator newsletter",
            "from": "publisher@example.net",
            "date": "2026-08-29T12:00:00+00:00",
            "snippet": "unrelated private newsletter body",
        },
        {
            "id": "fixture-id-003",
            "subject": "[OpenPoke Interview Fixture] run_A7k29mQ4 VidForge receipt VF-20481",
            "date": "2026-09-19T12:00:00+00:00",
            "body": "Fabricated fixture only",
        },
    ]
    baseline = {_sha("Unrelated login security warning")}
    enhanced = baseline | {_sha("Monthly creator newsletter")}
    return sanitize_inventory(
        messages,
        snapshot_at=datetime(2026, 9, 21, 15, 0, tzinfo=UTC),
        window_start=datetime(2026, 8, 1, tzinfo=UTC),
        window_end=datetime(2026, 9, 21, tzinfo=UTC),
        baseline_subject_hashes=baseline,
        enhanced_subject_hashes=enhanced,
    )


def test_inventory_retains_only_coarse_hash_window_marker_and_drift() -> None:
    inventory = _inventory()
    payload = inventory.model_dump(mode="json")
    rendered = json.dumps(payload, sort_keys=True)

    assert payload["track"] == "natural"
    assert payload["exploratory"] is True
    assert payload["include_in_controlled_aggregates"] is False
    assert payload["snapshot_at"] == "2026-09-21T15:00:00Z"
    assert payload["window"] == {
        "start": "2026-08-01T00:00:00Z",
        "end": "2026-09-21T00:00:00Z",
    }
    assert {tuple(item) for item in (
        (entry["category"], entry["date_range"], entry["marker"])
        for entry in payload["items"]
    )} == {
        ("security", "2026-09", "natural"),
        ("newsletter", "2026-08", "natural"),
        ("receipt", "2026-09", "fixture"),
    }
    assert payload["drift"]["added_subject_hashes"] == [_sha("Monthly creator newsletter")]
    assert payload["drift"]["removed_subject_hashes"] == []

    for forbidden in (
        "person@example.com",
        "owner@example.com",
        "publisher@example.net",
        "gmail-live-id-001",
        "thread-live-001",
        "private customer words",
        "super-secret-value",
        "tracker.invalid",
        "Unrelated login security warning",
        "Monthly creator newsletter",
        "VF-20481",
    ):
        assert forbidden not in rendered


def test_subject_hash_is_normalized_and_items_are_deduplicated() -> None:
    messages = [
        {"subject": "  Same   Natural Subject ", "date": "2026-09-01T00:00:00Z"},
        {"subject": "same natural subject", "date": "2026-09-15T00:00:00Z"},
    ]
    inventory = sanitize_inventory(
        messages,
        snapshot_at=datetime(2026, 9, 21, tzinfo=UTC),
        window_start=datetime(2026, 9, 1, tzinfo=UTC),
        window_end=datetime(2026, 9, 30, tzinfo=UTC),
    )

    assert len(inventory.items) == 1
    assert inventory.items[0].subject_sha256 == _sha("same natural subject")
    assert inventory.items[0].category == "other"


def test_exploratory_selection_is_deterministic_and_excludes_fixtures() -> None:
    inventory = _inventory()

    first = select_exploratory(inventory, limit=2)
    second = select_exploratory(inventory, limit=2)

    assert first == second
    assert len(first) == 2
    assert all(item.marker is InventoryMarker.NATURAL for item in first)
    assert inventory.include_in_controlled_aggregates is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snapshot_at": datetime(2026, 9, 21)},
        {
            "window_start": datetime(2026, 10, 1, tzinfo=UTC),
            "window_end": datetime(2026, 9, 1, tzinfo=UTC),
        },
        {"baseline_subject_hashes": {"not-a-sha"}},
    ],
)
def test_inventory_rejects_ambiguous_time_or_unsanitized_drift(kwargs) -> None:
    arguments = {
        "snapshot_at": datetime(2026, 9, 21, tzinfo=UTC),
        "window_start": datetime(2026, 9, 1, tzinfo=UTC),
        "window_end": datetime(2026, 9, 30, tzinfo=UTC),
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        sanitize_inventory(
            [{"subject": "Natural subject", "date": "2026-09-01T00:00:00Z"}],
            **arguments,
        )


def test_tracked_schema_forbids_raw_mail_fields() -> None:
    schema = json.loads(
        Path("evals/live_lab/scenarios/natural.schema.json").read_text(encoding="utf-8")
    )
    item = schema["properties"]["items"]["items"]

    assert item["additionalProperties"] is False
    assert set(item["properties"]) == {"category", "subject_sha256", "date_range", "marker"}
    assert schema["properties"]["include_in_controlled_aggregates"]["const"] is False
