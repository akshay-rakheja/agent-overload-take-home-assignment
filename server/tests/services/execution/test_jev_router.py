"""Unit tests for JevRouter map/reduce flow, attribution, and guardrails."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from server.services.execution.activity_card import ActivityCard
from server.services.execution.jev_client import FakeJevClient
from server.services.execution.jev_router import JevRouter, JevRoutingContext
from server.services.execution.routing import RoutingAction


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def roster():
    now = datetime.now(timezone.utc)
    sec_id = UUID("11111111-1111-4111-8111-111111111111")
    vid_id = UUID("22222222-2222-4222-8222-222222222222")
    cal_id = UUID("33333333-3333-4333-8333-333333333333")

    sec_card = ActivityCard(
        agent_id=sec_id,
        name="Instagram Security Monitor",
        purpose="Review Instagram security notices and account protection events",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=10,
        entity_claims=("instagram.com", "security"),
    )
    vid_card = ActivityCard(
        agent_id=vid_id,
        name="AI Video Newsletter Curator",
        purpose="Curate AI video generation tutorials and weekly newsletter",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=5,
        entity_claims=("video", "newsletter"),
    )
    cal_card = ActivityCard(
        agent_id=cal_id,
        name="Calendar Deadline Tracker",
        purpose="Track deadlines and schedule calendar events",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=3,
        entity_claims=("calendar", "deadlines"),
    )
    return [sec_card, vid_card, cal_card]


@pytest.mark.anyio
async def test_jev_router_confident_reuse(roster):
    client = FakeJevClient()
    router = JevRouter(client=client, max_concurrency=5)

    query = "Check the suspicious Instagram login alert"
    context = await router.route(query, roster)

    assert context.decision.action == RoutingAction.REUSE
    assert context.decision.recommended_agent_id == roster[0].agent_id
    assert context.authorized_ids == (str(roster[0].agent_id),)
    assert context.decision.winner_margin >= 0.15
    assert len(context.map_scores) == len(roster)
    assert len(context.shortlist) <= 3


@pytest.mark.anyio
async def test_jev_router_novel_task_creates_new(roster):
    client = FakeJevClient()
    router = JevRouter(client=client)

    query = "Design a 3D architectural blueprint for a warehouse"
    context = await router.route(query, roster)

    assert context.decision.action == RoutingAction.CREATE_NEW
    assert context.decision.recommended_agent_id is None
    assert context.authorized_ids == ()


@pytest.mark.anyio
async def test_jev_router_ambiguous_abstains():
    now = datetime.now(timezone.utc)
    id1 = uuid4()
    id2 = uuid4()

    card1 = ActivityCard(
        agent_id=id1,
        name="Security Auditor Alpha",
        purpose="Audit general account security",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=5,
        entity_claims=("security",),
    )
    card2 = ActivityCard(
        agent_id=id2,
        name="Security Auditor Beta",
        purpose="Audit general account security",
        status="hot",
        created_at=now,
        last_used_at=now,
        episode_count=5,
        entity_claims=("security",),
    )

    client = FakeJevClient(ambiguity_margin=0.15)
    router = JevRouter(client=client)

    query = "Perform a security audit"
    context = await router.route(query, [card1, card2])

    assert context.decision.action == RoutingAction.ABSTAIN
    assert context.decision.recommended_agent_id is None
    assert context.authorized_ids == ()
    assert context.decision.winner_margin < 0.15


@pytest.mark.anyio
async def test_jev_router_partial_failure_resilience(roster):
    # One card will fail or time out
    client = FakeJevClient()
    failing_id = roster[1].agent_id

    # Custom wrapper that fails only for roster[1]
    original_score = client.score_agent

    async def mock_score(card, req):
        if card.agent_id == failing_id:
            raise RuntimeError("Individual scoring timeout")
        return await original_score(card, req)

    client.score_agent = mock_score

    router = JevRouter(client=client)
    query = "Check the suspicious Instagram login alert"
    context = await router.route(query, roster)

    # Roster[0] (Instagram) should still be scored and recommended
    assert context.decision.action == RoutingAction.REUSE
    assert context.decision.recommended_agent_id == roster[0].agent_id
    assert context.authorized_ids == (str(roster[0].agent_id),)
    # Check that error is recorded in attribution
    assert str(failing_id) in context.partial_failures
