"""Convert portable corpus fixtures into production AgentRecord objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

from evals.fixtures import make_roster
from evals.schema import AgentFixture, RoutingCase
from server.services.execution.models import AgentRecord, AgentStatus


EVALUATION_NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def stable_fixture_id(value: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"openpoke-eval:{value}")


def fixture_to_record(agent: AgentFixture) -> AgentRecord:
    return AgentRecord(
        agent_id=stable_fixture_id(agent.agent_id),
        name=agent.name,
        purpose=agent.purpose or f"Handle tasks related to {agent.name}",
        aliases=tuple(agent.aliases),
        status=AgentStatus(agent.status),
        created_at=EVALUATION_NOW - timedelta(days=365),
        last_used_at=datetime.fromisoformat(agent.last_used_at.replace("Z", "+00:00")),
        use_count=agent.use_count,
        memory_summary=agent.memory_summary,
    )


@dataclass(frozen=True)
class MaterializedCase:
    case: RoutingCase
    records: tuple[AgentRecord, ...]
    expected_agent_id: str | None


def materialize_case(case: RoutingCase) -> MaterializedCase:
    if case.agents is not None:
        fixtures = case.agents
    else:
        assert case.synthetic_roster is not None
        fixtures = make_roster(
            case.synthetic_roster.size,
            target=case.synthetic_roster.target,
            target_index=case.synthetic_roster.target_index,
        )
    records = tuple(fixture_to_record(agent) for agent in fixtures)
    expected = (
        str(stable_fixture_id(case.expected_agent_id))
        if case.expected_agent_id is not None
        else None
    )
    return MaterializedCase(case=case, records=records, expected_agent_id=expected)

