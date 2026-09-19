"""Tests for bounded candidate prompt construction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from server.agents.interaction_agent.agent import (
    build_candidate_context,
    prepare_message_with_history,
)
from server.services.execution.models import AgentRecord, AgentStatus
from server.services.execution.routing import RoutingAction


UTC = timezone.utc
NOW = datetime.now(UTC)


class StubDirectory:
    def __init__(self, records: list[AgentRecord]) -> None:
        self.records = records

    def list_records(self) -> list[AgentRecord]:
        return list(self.records)


def record(
    index: int,
    *,
    name: str | None = None,
    purpose: str | None = None,
    aliases: tuple[str, ...] = (),
    status: AgentStatus = AgentStatus.HOT,
) -> AgentRecord:
    display_name = name or f"Synthetic workflow {index:05d}"
    return AgentRecord(
        agent_id=uuid5(NAMESPACE_URL, f"prompt-record:{index}"),
        name=display_name,
        purpose=purpose or f"Handle synthetic account {index:05d} follow-ups",
        aliases=aliases or (f"account-{index:05d}",),
        status=status,
        created_at=NOW - timedelta(days=300),
        last_used_at=NOW - timedelta(days=index % 40),
        use_count=index % 12,
        memory_summary="",
    )


def test_prompt_contains_only_five_candidates_with_thousand_record_directory() -> None:
    records = [record(index) for index in range(1_000)]
    directory = StubDirectory(records)

    messages = prepare_message_with_history(
        "Find account-00942 follow-ups",
        "",
        directory=directory,
    )
    content = messages[0]["content"]

    assert content.count("<agent_candidate ") <= 5
    assert str(records[942].agent_id) in content
    assert "Synthetic workflow 00001" not in content
    assert "<active_agents>" not in content
    assert "<agent_candidates" in content


def test_candidate_prompt_includes_stable_fields_and_relevance_hints_without_scores() -> None:
    alice = record(
        1,
        name="Alice correspondence",
        purpose="Track proposal messages with Alice",
        aliases=("Alice",),
        status=AgentStatus.DORMANT,
    )

    content = prepare_message_with_history(
        "Did Alice reply to the proposal?",
        "",
        directory=StubDirectory([alice]),
    )[0]["content"]

    assert f'id="{alice.agent_id}"' in content
    assert 'name="Alice correspondence"' in content
    assert 'status="dormant"' in content
    assert "Track proposal messages with Alice" in content
    assert "exact name or alias phrase" in content
    assert "score=" not in content


def test_candidate_context_exposes_safe_routing_guidance() -> None:
    first = record(1, name="Alex client", purpose="Manage Alex client updates", aliases=("Alex",))
    second = record(2, name="Alex candidate", purpose="Manage Alex recruiting", aliases=("Alex",))

    context = build_candidate_context(
        "What did Alex say?",
        "",
        directory=StubDirectory([first, second]),
    )

    assert context.decision.action is RoutingAction.ABSTAIN
    assert context.decision.agent_id is None
    assert len(context.candidates) == 2

