"""Unit tests for provider-neutral JevClient, FakeJevClient, and TypeSafeJevClient."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from server.services.execution.activity_card import ActivityCard
from server.services.execution.jev_client import (
    AgentScore,
    FakeJevClient,
    JevClient,
    JevProviderError,
    JevRateLimitError,
    JevTimeoutError,
    TypeSafeJevClient,
)
from server.services.execution.routing import RoutingAction


@pytest.fixture
def sample_cards():
    now = datetime.now(timezone.utc)
    security_card = ActivityCard(
        agent_id=uuid4(),
        name="Instagram Security Monitor",
        purpose="Review Instagram security notices and account protection events",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=10,
        entity_claims=("instagram.com", "security"),
    )
    video_card = ActivityCard(
        agent_id=uuid4(),
        name="AI Video Newsletter Curator",
        purpose="Curate AI video generation tutorials and weekly newsletter",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=5,
        entity_claims=("video", "newsletter"),
    )
    return security_card, video_card


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_fake_jev_client_scoring(sample_cards):
    sec_card, vid_card = sample_cards
    client = FakeJevClient()

    query = "Please check the suspicious login notice on my Instagram account"
    score_sec = await client.score_agent(sec_card, query)
    score_vid = await client.score_agent(vid_card, query)

    assert score_sec.agent_id == sec_card.agent_id
    assert score_sec.composite_score > 0.7
    assert score_sec.affinity_score > 0.7
    assert score_vid.composite_score < 0.4
    assert score_sec.composite_score > score_vid.composite_score


@pytest.mark.anyio
async def test_fake_jev_client_reduce_reuse(sample_cards):
    sec_card, vid_card = sample_cards
    client = FakeJevClient()

    score_sec = AgentScore(
        agent_id=sec_card.agent_id,
        composite_score=0.92,
        affinity_score=0.95,
        continuity_score=0.90,
        risk_score=0.05,
        reasoning="Exact domain and task match",
        card_digest=sec_card.digest,
    )
    score_vid = AgentScore(
        agent_id=vid_card.agent_id,
        composite_score=0.20,
        affinity_score=0.15,
        continuity_score=0.10,
        risk_score=0.10,
        reasoning="Unrelated video domain",
        card_digest=vid_card.digest,
    )

    decision = await client.reduce_shortlist(
        [score_sec, score_vid], "Instagram login security check"
    )

    assert decision.action == RoutingAction.REUSE
    assert decision.recommended_agent_id == sec_card.agent_id
    assert decision.winner_margin >= 0.7
    assert decision.agreement_with_top_candidate is True


@pytest.mark.anyio
async def test_fake_jev_client_reduce_abstain_on_ambiguity(sample_cards):
    sec_card, vid_card = sample_cards
    client = FakeJevClient()

    # Close scores trigger ambiguity / abstain
    score1 = AgentScore(
        agent_id=sec_card.agent_id,
        composite_score=0.82,
        affinity_score=0.85,
        continuity_score=0.80,
        risk_score=0.1,
        reasoning="Candidate 1 matches",
        card_digest=sec_card.digest,
    )
    score2 = AgentScore(
        agent_id=vid_card.agent_id,
        composite_score=0.80,
        affinity_score=0.82,
        continuity_score=0.78,
        risk_score=0.1,
        reasoning="Candidate 2 matches closely",
        card_digest=vid_card.digest,
    )

    decision = await client.reduce_shortlist(
        [score1, score2], "Ambiguous request"
    )

    assert decision.action == RoutingAction.ABSTAIN
    assert decision.winner_margin < 0.15


@pytest.mark.anyio
async def test_fake_jev_client_failure_modes(sample_cards):
    sec_card, _ = sample_cards

    # Test timeout injection
    client_timeout = FakeJevClient(inject_timeout=True)
    with pytest.raises(JevTimeoutError):
        await client_timeout.score_agent(sec_card, "query")

    # Test 429 rate limit injection
    client_429 = FakeJevClient(inject_rate_limit=True)
    with pytest.raises(JevRateLimitError):
        await client_429.score_agent(sec_card, "query")

    # Test general failure injection
    client_err = FakeJevClient(inject_failure=True)
    with pytest.raises(JevProviderError):
        await client_err.score_agent(sec_card, "query")


def test_typesafe_jev_client_credential_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Verify .env permission check and missing key error without leaking credentials
    fake_env = tmp_path / ".env"
    fake_env.write_text("SOME_OTHER_KEY=123\n")
    fake_env.chmod(0o644)  # Not 0600!

    with pytest.raises(PermissionError, match="0600"):
        TypeSafeJevClient.verify_env_permissions(fake_env)

    fake_env.chmod(0o600)
    assert TypeSafeJevClient.verify_env_permissions(fake_env) is True

    # Missing JEV_API_KEY
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JEV_API_KEY is not configured"):
        TypeSafeJevClient(api_key=None, env_path=fake_env)
