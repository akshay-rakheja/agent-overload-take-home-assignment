"""Deterministic synthetic data shared by tests and offline evaluations."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from evals.schema import AgentFixture


HistoryEntry = tuple[str, str, str]


def make_agent(
    *,
    agent_id: str | None = None,
    name: str = "Synthetic Agent",
    purpose: str = "Handle a deterministic synthetic workflow",
    aliases: list[str] | None = None,
    status: str = "hot",
    index: int = 0,
) -> AgentFixture:
    """Create a stable fixture without randomness or the current clock."""

    stable_id = agent_id or str(uuid5(NAMESPACE_URL, f"openpoke-agent-{index}"))
    return AgentFixture(
        agent_id=stable_id,
        name=name,
        purpose=purpose,
        aliases=aliases or [],
        status=status,
        last_used_at=f"2026-01-{(index % 28) + 1:02d}T12:00:00Z",
        use_count=index % 17,
        memory_summary=f"Deterministic memory for {name}.",
    )


def make_roster(
    size: int,
    *,
    target: AgentFixture | None = None,
    target_index: int = 0,
) -> list[AgentFixture]:
    """Build a repeatable roster and optionally insert an authored target."""

    if size < 1:
        raise ValueError("size must be positive")
    if target is not None and not 0 <= target_index < size:
        raise ValueError("target_index must fit inside the roster")

    roster = [
        make_agent(
            index=index,
            name=f"Synthetic workflow {index:05d}",
            purpose=f"Handle synthetic account {index:05d} follow-ups",
            aliases=[f"account-{index:05d}"],
            status=("hot", "dormant", "archived")[index % 3],
        )
        for index in range(size)
    ]
    if target is not None:
        roster[target_index] = target
    return roster


def make_history(size: int) -> list[HistoryEntry]:
    """Return repeatable log entries spanning complete synthetic episodes."""

    if size < 0:
        raise ValueError("size cannot be negative")
    tags = ("agent_request", "agent_action", "tool_response", "agent_response")
    return [
        (
            tags[index % len(tags)],
            f"2026-01-{(index % 28) + 1:02d} 12:{index % 60:02d}:00",
            f"Synthetic history entry {index:05d} for deterministic depth measurement.",
        )
        for index in range(size)
    ]

