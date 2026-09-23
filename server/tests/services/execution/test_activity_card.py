"""Unit tests for structured ActivityRecord and bounded ActivityCard generation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from server.services.execution.activity_card import (
    ActivityCard,
    ActivityCardGenerator,
    ActivityRecord,
    ActivityRecordSummary,
)


def test_activity_record_valid():
    agent_id = uuid4()
    episode_id = uuid4()
    now = datetime.now(timezone.utc)

    record = ActivityRecord(
        agent_id=agent_id,
        episode_id=episode_id,
        occurred_at=now,
        request_intent="Review security alert for Instagram login",
        enduring_responsibility="Account Security & Access Control",
        entity_bindings=("instagram.com", "security-alert"),
        actions=("search_email", "verify_ip"),
        rationale="Agent determined that login notification required verification.",
        disposition="completed",
        open_obligations=("monitor for subsequent attempts",),
        provenance="journal:sha256:abcd",
    )

    assert record.agent_id == agent_id
    assert record.episode_id == episode_id
    assert record.occurred_at == now
    assert record.disposition == "completed"
    assert record.schema_version == 1
    assert "instagram.com" in record.entity_bindings


def test_activity_record_requires_timezone_aware_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        ActivityRecord(
            agent_id=uuid4(),
            episode_id=uuid4(),
            occurred_at=datetime.now(),  # naive datetime
            request_intent="Check alert",
            enduring_responsibility="Security",
            entity_bindings=(),
            actions=(),
            rationale="Rationale",
            disposition="completed",
            open_obligations=(),
            provenance="journal:ref",
        )


def test_activity_card_bounded_and_digest():
    agent_id = uuid4()
    now = datetime.now(timezone.utc)

    summary = ActivityRecordSummary(
        episode_id=uuid4(),
        occurred_at=now,
        request_intent="Follow up on receipt",
        actions=("search_email",),
        disposition="completed",
    )

    card = ActivityCard(
        agent_id=agent_id,
        name="Receipt Reconciler",
        purpose="Track purchases and reconcile receipts",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=5,
        current_obligations=("reconcile pending invoice",),
        durable_summary="Specialized in processing digital invoices and purchase confirmations.",
        recent_episodes=(summary,),
        entity_claims=("invoices", "receipts"),
    )

    assert card.agent_id == agent_id
    assert card.digest is not None
    assert len(card.digest) == 64  # sha256 hex digest
    assert card.render_text() is not None
    assert len(card.render_text()) < 2000  # bounded representation


def test_activity_card_generator_redacts_and_bounds():
    agent_id = uuid4()
    now = datetime.now(timezone.utc)

    raw_journal = [
        {
            "episode_id": str(uuid4()),
            "occurred_at": now.isoformat(),
            "request_intent": "Find secret key sk-1234567890abcdef in user@example.com emails",
            "enduring_responsibility": "Email processing",
            "entity_bindings": ["user@example.com", "Bearer secret-token-xyz"],
            "actions": ["search_email: raw body with sensitive credit card info 1234-5678"],
            "rationale": "I searched the inbox for user credentials sk-987654321.",
            "disposition": "completed",
            "open_obligations": ["notify user"],
            "provenance": "test-journal",
        }
    ]

    generator = ActivityCardGenerator(
        agent_id=agent_id,
        name="Test Agent",
        purpose="Handle emails securely",
        status="active",
        created_at=now,
        last_used_at=now,
    )

    card = generator.generate_from_records(raw_journal)
    rendered = card.render_text()

    # Verify sensitive tokens and secrets are redacted
    assert "sk-1234567890abcdef" not in rendered
    assert "secret-token-xyz" not in rendered
    assert "Bearer" not in rendered or "[REDACTED" in rendered
    assert card.digest == generator.generate_from_records(raw_journal).digest
