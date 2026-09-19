"""Tests for explicit reuse/create-new/abstain routing decisions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from server.services.execution.models import AgentRecord, AgentStatus
from server.services.execution.retrieval import AgentRetriever, RetrievalQuery
from server.services.execution.retrieval import AgentCandidate
from server.services.execution.routing import AgentRouter, RoutingAction


UTC = timezone.utc
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def record(key: str, name: str, purpose: str, aliases: tuple[str, ...] = ()) -> AgentRecord:
    return AgentRecord(
        agent_id=uuid5(NAMESPACE_URL, f"routing:{key}"),
        name=name,
        purpose=purpose,
        aliases=aliases,
        status=AgentStatus.HOT,
        created_at=NOW - timedelta(days=30),
        last_used_at=NOW - timedelta(days=2),
        use_count=3,
        memory_summary="",
    )


def route(query: RetrievalQuery, records: list[AgentRecord]):
    candidates = AgentRetriever(records, now=lambda: NOW).retrieve(query)
    return AgentRouter().route(query, candidates)


def test_unique_strong_match_reuses_stable_id() -> None:
    alice = record(
        "alice",
        "Alice correspondence",
        "Track proposal messages with Alice",
        ("Alice",),
    )

    decision = route(
        RetrievalQuery("Did Alice reply to the proposal?", ""),
        [alice],
    )

    assert decision.action is RoutingAction.REUSE
    assert decision.agent_id == alice.agent_id
    assert decision.confidence > 0


def test_novel_task_creates_new_identity() -> None:
    car = record("car", "Car service", "Schedule vehicle maintenance")

    decision = route(
        RetrievalQuery("Book a dentist appointment for next Thursday", ""),
        [car],
    )

    assert decision.action is RoutingAction.CREATE_NEW
    assert decision.agent_id is None


def test_materially_ambiguous_aliases_abstain() -> None:
    printer = record("printer", "Printer vendor", "Manage printer supplier", ("vendor",))
    caterer = record("caterer", "Catering vendor", "Manage caterer", ("vendor",))

    decision = route(
        RetrievalQuery("Follow up with the vendor", ""),
        [printer, caterer],
    )

    assert decision.action is RoutingAction.ABSTAIN
    assert decision.agent_id is None
    assert "ambiguous" in " ".join(decision.reasons).lower()


def test_shared_exact_entity_with_close_task_scores_abstains() -> None:
    designer = record(
        "alex-designer",
        "Alex design",
        "Coordinate homepage design with Alex",
        ("Alex",),
    )
    recruiter = record(
        "alex-recruiter",
        "Alex recruiting",
        "Coordinate engineering recruiting with Alex",
        ("Alex",),
    )

    decision = route(
        RetrievalQuery("Follow up with Alex about the homepage", ""),
        [designer, recruiter],
    )

    assert decision.action is RoutingAction.ABSTAIN


def test_router_never_returns_an_id_outside_candidates() -> None:
    alice = record("alice", "Alice correspondence", "Track Alice messages", ("Alice",))
    query = RetrievalQuery("Did Alice reply?", "")
    candidates = AgentRetriever([alice], now=lambda: NOW).retrieve(query)

    decision = AgentRouter().route(query, candidates)

    assert decision.agent_id is None or decision.agent_id in {
        candidate.agent_id for candidate in candidates
    }


def test_ambiguity_margin_applies_when_runner_up_is_just_below_reuse_threshold() -> None:
    first = record("first", "First", "First candidate")
    second = record("second", "Second", "Second candidate")
    candidates = [
        AgentCandidate(
            agent_id=first.agent_id,
            name=first.name,
            purpose=first.purpose,
            status=first.status,
            score=0.340,
            score_components={"token_overlap": 0.340},
            reasons=("first",),
        ),
        AgentCandidate(
            agent_id=second.agent_id,
            name=second.name,
            purpose=second.purpose,
            status=second.status,
            score=0.339,
            score_components={"token_overlap": 0.339},
            reasons=("second",),
        ),
    ]

    decision = AgentRouter().route(RetrievalQuery("ambiguous", ""), candidates)

    assert decision.action is RoutingAction.ABSTAIN
