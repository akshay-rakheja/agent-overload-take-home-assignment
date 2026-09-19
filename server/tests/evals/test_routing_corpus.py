"""Contract tests for the committed routing evaluation corpus."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.schema import (
    REQUIRED_CATEGORIES,
    AgentFixture,
    RoutingCase,
    SyntheticRosterReference,
    load_routing_corpus,
)
from server.tests.fixtures.agent_factory import make_agent, make_roster


CORPUS_PATH = Path(__file__).resolve().parents[3] / "evals" / "agent_routing_cases.jsonl"


def test_committed_corpus_has_balanced_development_and_held_out_splits() -> None:
    corpus = load_routing_corpus(CORPUS_PATH)

    assert len(corpus) >= 40
    assert sum(case.split == "development" for case in corpus) >= 20
    assert sum(case.split == "test" for case in corpus) >= 20
    assert len({case.case_id for case in corpus}) == len(corpus)


def test_committed_corpus_covers_required_categories_and_roster_scales() -> None:
    corpus = load_routing_corpus(CORPUS_PATH)

    assert REQUIRED_CATEGORIES <= {case.category for case in corpus}
    assert {10, 100, 500, 1_000} <= {
        case.synthetic_roster.size
        for case in corpus
        if case.synthetic_roster is not None
    }


def test_reuse_requires_an_expected_agent_id() -> None:
    with pytest.raises(ValidationError, match="expected_agent_id"):
        RoutingCase(
            case_id="missing-reuse-target",
            split="development",
            category="exact_named_follow_up",
            query="Did Alice reply?",
            conversation_context="",
            agents=[make_agent(agent_id="alice", name="Alice correspondence")],
            expected_action="reuse",
            notes="Invalid on purpose.",
        )


def test_case_requires_exactly_one_agent_source() -> None:
    agent = make_agent(agent_id="alice", name="Alice correspondence")

    with pytest.raises(ValidationError, match="exactly one"):
        RoutingCase(
            case_id="two-sources",
            split="development",
            category="exact_named_follow_up",
            query="Did Alice reply?",
            conversation_context="",
            agents=[agent],
            synthetic_roster=SyntheticRosterReference(size=10),
            expected_action="reuse",
            expected_agent_id="alice",
            notes="Invalid on purpose.",
        )


def test_synthetic_roster_is_deterministic_and_preserves_target() -> None:
    target = AgentFixture(
        agent_id="target-alice",
        name="Alice correspondence",
        purpose="Track messages with Alice",
        aliases=["Alice"],
        status="dormant",
        last_used_at="2026-09-01T12:00:00Z",
        use_count=7,
        memory_summary="Alice is waiting for the revised proposal.",
    )

    first = make_roster(size=1_000, target=target, target_index=723)
    second = make_roster(size=1_000, target=target, target_index=723)

    assert first == second
    assert len(first) == 1_000
    assert first[723] == target
    assert len({agent.agent_id for agent in first}) == 1_000


def test_invalid_jsonl_reports_the_source_line(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text('{"case_id": "broken"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid\.jsonl:1"):
        load_routing_corpus(invalid_path)
