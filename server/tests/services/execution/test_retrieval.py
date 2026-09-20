"""Tests for deterministic bounded hybrid agent retrieval."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from pydantic import ValidationError

from evals.fixtures import make_roster
from evals.schema import AgentFixture, RoutingCase, load_routing_corpus
from server.config import Settings
from server.services.execution.models import AgentRecord, AgentStatus
from server.services.execution.retrieval import AgentRetriever, RetrievalQuery


UTC = timezone.utc
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
CORPUS_PATH = Path(__file__).resolve().parents[4] / "evals" / "agent_routing_cases.jsonl"


def stable_fixture_id(value: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"openpoke-eval:{value}")


def to_record(agent: AgentFixture, *, index: int = 0) -> AgentRecord:
    return AgentRecord(
        agent_id=stable_fixture_id(agent.agent_id),
        name=agent.name,
        purpose=agent.purpose or f"Handle tasks related to {agent.name}",
        aliases=tuple(agent.aliases),
        status=AgentStatus(agent.status),
        created_at=NOW - timedelta(days=365),
        last_used_at=datetime.fromisoformat(agent.last_used_at.replace("Z", "+00:00")),
        use_count=agent.use_count,
        memory_summary=agent.memory_summary,
    )


def materialize(case: RoutingCase) -> list[AgentRecord]:
    if case.agents is not None:
        return [to_record(agent, index=index) for index, agent in enumerate(case.agents)]
    assert case.synthetic_roster is not None
    target = case.synthetic_roster.target
    fixture_roster = make_roster(
        case.synthetic_roster.size,
        target=target,
        target_index=case.synthetic_roster.target_index,
    )
    return [to_record(agent, index=index) for index, agent in enumerate(fixture_roster)]


def make_record(
    key: str,
    *,
    name: str,
    purpose: str,
    aliases: tuple[str, ...] = (),
    status: AgentStatus = AgentStatus.HOT,
    age_days: int = 0,
    use_count: int = 0,
) -> AgentRecord:
    return AgentRecord(
        agent_id=stable_fixture_id(key),
        name=name,
        purpose=purpose,
        aliases=aliases,
        status=status,
        created_at=NOW - timedelta(days=400),
        last_used_at=NOW - timedelta(days=age_days),
        use_count=use_count,
        memory_summary="",
    )


def test_default_retrieval_configuration_is_bounded() -> None:
    settings = Settings()

    assert settings.agent_retrieval_top_k == 5
    assert 0 < settings.agent_retrieval_min_score < 1
    assert 0 < settings.agent_route_reuse_threshold < 1
    assert 0 < settings.agent_route_ambiguity_margin < 1


def test_retrieval_configuration_rejects_more_than_five_candidates() -> None:
    with pytest.raises(ValidationError):
        Settings(agent_retrieval_top_k=6)


def test_environment_retrieval_configuration_above_five_is_rejected() -> None:
    environment = dict(os.environ)
    environment["OPENPOKE_AGENT_RETRIEVAL_TOP_K"] = "20"

    completed = subprocess.run(
        [sys.executable, "-c", "from server.config import Settings; Settings()"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "less than or equal to 5" in completed.stderr


def test_retrieval_hard_cap_survives_validation_bypass() -> None:
    records = [
        make_record(
            f"record-{index}",
            name=f"Alice project {index}",
            purpose="Coordinate Alice project follow-ups",
        )
        for index in range(30)
    ]
    unsafe_settings = Settings.model_construct(
        agent_retrieval_top_k=20,
        agent_retrieval_min_score=0.0,
    )

    candidates = AgentRetriever(
        records,
        now=lambda: NOW,
        settings=unsafe_settings,
    ).retrieve(RetrievalQuery(text="Follow up on Alice's project"))

    assert len(candidates) == 5


def test_candidate_count_never_exceeds_top_k_and_order_is_deterministic() -> None:
    records = [
        make_record(
            f"record-{index}",
            name=f"Alice project {index}",
            purpose="Coordinate Alice project follow-ups",
        )
        for index in range(30)
    ]
    retriever = AgentRetriever(records, now=lambda: NOW)
    query = RetrievalQuery(text="Follow up on Alice's project", conversation_context="")

    first = retriever.retrieve(query)
    second = retriever.retrieve(query)
    oversized_request = retriever.retrieve(query, limit=100)

    assert first == second
    assert len(first) == 5
    assert len(oversized_request) == 5
    assert all(candidate.score_components for candidate in first)
    assert all(candidate.reasons for candidate in first)


def test_exact_alias_outranks_unrelated_recent_high_use_record() -> None:
    target = make_record(
        "cybersafe",
        name="CyberSafe insurance renewal",
        purpose="Handle the cyber insurance renewal",
        aliases=("CyberSafe",),
        status=AgentStatus.DORMANT,
        age_days=300,
    )
    distractor = make_record(
        "lunch",
        name="Team lunch",
        purpose="Book the team lunch",
        age_days=0,
        use_count=200,
    )

    candidates = AgentRetriever([distractor, target], now=lambda: NOW).retrieve(
        RetrievalQuery(text="Reopen the CyberSafe renewal", conversation_context="")
    )

    assert candidates[0].agent_id == target.agent_id
    assert candidates[0].score_components["exact_match"] > 0
    assert "exact name or alias phrase" in candidates[0].reasons


def test_archived_exact_match_is_recoverable() -> None:
    archived = make_record(
        "atlas",
        name="Atlas audit",
        purpose="Collect Atlas audit evidence",
        aliases=("Atlas",),
        status=AgentStatus.ARCHIVED,
        age_days=700,
    )

    candidates = AgentRetriever([archived], now=lambda: NOW).retrieve(
        RetrievalQuery(text="Reopen the Atlas audit", conversation_context="")
    )

    assert candidates[0].agent_id == archived.agent_id
    assert candidates[0].score_components["lifecycle"] < 0


def test_development_corpus_reaches_top_five_recall_target() -> None:
    reuse_cases = [
        case
        for case in load_routing_corpus(CORPUS_PATH)
        if case.split == "development" and case.expected_action == "reuse"
    ]
    hits = 0
    for case in reuse_cases:
        candidates = AgentRetriever(materialize(case), now=lambda: NOW).retrieve(
            RetrievalQuery(text=case.query, conversation_context=case.conversation_context)
        )
        expected_id = stable_fixture_id(case.expected_agent_id or "")
        hits += expected_id in {candidate.agent_id for candidate in candidates}

    assert hits / len(reuse_cases) >= 0.95


def test_thousand_record_retrieval_is_bounded_and_records_duration() -> None:
    fixtures = make_roster(1_000)
    records = [to_record(agent, index=index) for index, agent in enumerate(fixtures)]
    retriever = AgentRetriever(records, now=lambda: NOW)

    started = perf_counter()
    candidates = retriever.retrieve(
        RetrievalQuery(text="Find account-00942 follow-ups", conversation_context="")
    )
    duration_ms = (perf_counter() - started) * 1_000

    assert len(candidates) <= 5
    assert candidates[0].agent_id == records[942].agent_id
    assert duration_ms >= 0


def test_callable_directory_provider_rebuilds_features_after_metadata_changes() -> None:
    records = [make_record("alice", name="Alice", purpose="Track Alice", aliases=("Alice",))]
    retriever = AgentRetriever(lambda: records, now=lambda: NOW)

    assert retriever.retrieve(RetrievalQuery("Alice"))[0].name == "Alice"

    records[:] = [make_record("bob", name="Bob", purpose="Track Bob", aliases=("Bob",))]

    candidates = retriever.retrieve(RetrievalQuery("Bob"))
    assert candidates[0].name == "Bob"
