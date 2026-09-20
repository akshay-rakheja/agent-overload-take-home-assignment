"""Tests for stable-ID execution-agent dispatch."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID

from server.agents.interaction_agent.tools import DispatchContext, send_message_to_agent
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.models import AgentStatus
from server.services.execution.routing import RoutingAction


@dataclass
class Submitted:
    agent_name: str
    instructions: str
    agent_id: str | None


class FakeBatchManager:
    def __init__(self) -> None:
        self.submitted: list[Submitted] = []

    async def execute_agent(
        self,
        agent_name: str,
        instructions: str,
        request_id: str | None = None,
        agent_id: str | None = None,
    ):
        del request_id
        self.submitted.append(Submitted(agent_name, instructions, agent_id))
        return SimpleNamespace(success=True)


def dispatch_and_drain(**kwargs):
    async def run():
        result = send_message_to_agent(**kwargs)
        await asyncio.sleep(0)
        return result

    return asyncio.run(run())


def test_reuse_requires_stable_id_and_updates_lifecycle(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    record = directory.create(name="Alice", purpose="Track Alice messages", aliases=["Alice"])
    directory.transition(record.agent_id, AgentStatus.ARCHIVED)
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()

    result = dispatch_and_drain(
        agent_id=str(record.agent_id),
        instructions="Check for Alice's reply",
        dispatch_context=DispatchContext(),
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    updated = directory.require(record.agent_id)
    assert result.success is True
    assert result.payload["agent_id"] == str(record.agent_id)
    assert result.payload["new_agent_created"] is False
    assert updated.status is AgentStatus.HOT
    assert updated.use_count == 1
    assert logs.load_transcript(str(record.agent_id))
    assert batch.submitted == [Submitted("Alice", "Check for Alice's reply", str(record.agent_id))]


def test_unknown_id_fails_closed_without_logging_or_dispatch(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    missing = UUID("00000000-0000-0000-0000-000000000001")

    result = dispatch_and_drain(
        agent_id=str(missing),
        instructions="Do not run",
        dispatch_context=DispatchContext(),
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    assert result.success is False
    assert result.payload["code"] == "unknown_agent_id"
    assert batch.submitted == []
    assert logs.list_agents() == []


def test_new_agent_creation_is_idempotent_within_one_turn(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext()
    kwargs = {
        "agent_name": "Bob invoices",
        "agent_purpose": "Track and resolve Bob's invoices",
        "instructions": "Find the latest invoice",
        "dispatch_context": context,
        "directory": directory,
        "log_store": logs,
        "batch_manager": batch,
    }

    first = dispatch_and_drain(**kwargs)
    second = dispatch_and_drain(**kwargs)

    assert len(directory.list_records()) == 1
    assert first.payload["agent_id"] == second.payload["agent_id"]
    assert first.payload["new_agent_created"] is True
    assert second.payload["new_agent_created"] is False


def test_create_retry_reuses_identity_across_name_and_purpose_paraphrases(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext()

    first = dispatch_and_drain(
        agent_name="  Bob   Invoices ",
        agent_purpose="Track and resolve Bob's invoices",
        instructions="Find the latest invoice",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )
    second = dispatch_and_drain(
        agent_name="bob invoices",
        agent_purpose="Manage invoice follow-ups for Bob",
        instructions="Check again",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    assert len(directory.list_records()) == 1
    assert first.payload["agent_id"] == second.payload["agent_id"]
    assert second.payload["new_agent_created"] is False


def test_distinct_creation_intents_allow_two_same_name_identities(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext()

    first = dispatch_and_drain(
        agent_name="Vendor review",
        agent_purpose="Review the first vendor",
        creation_intent_id="vendor-one",
        instructions="Review vendor one",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )
    second = dispatch_and_drain(
        agent_name="Vendor review",
        agent_purpose="Review the second vendor",
        creation_intent_id="vendor-two",
        instructions="Review vendor two",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    assert len(directory.list_records()) == 2
    assert first.payload["agent_id"] != second.payload["agent_id"]
    assert first.payload["new_agent_created"] is True
    assert second.payload["new_agent_created"] is True


def test_missing_event_loop_does_not_create_or_log_identity(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")

    result = send_message_to_agent(
        agent_name="Bob invoices",
        agent_purpose="Track Bob invoices",
        instructions="Do not partially apply",
        dispatch_context=DispatchContext(routing_action=RoutingAction.CREATE_NEW),
        directory=directory,
        log_store=logs,
        batch_manager=FakeBatchManager(),
    )

    assert result.success is False
    assert directory.list_records() == []
    assert logs.list_agents() == []


def test_missing_event_loop_does_not_mark_or_log_reused_identity(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    record = directory.create(name="Alice", purpose="Track Alice")
    logs = ExecutionAgentLogStore(tmp_path / "logs")

    result = send_message_to_agent(
        agent_id=str(record.agent_id),
        instructions="Do not partially apply",
        dispatch_context=DispatchContext(
            routing_action=RoutingAction.REUSE,
            allowed_agent_ids=frozenset({record.agent_id}),
        ),
        directory=directory,
        log_store=logs,
        batch_manager=FakeBatchManager(),
    )

    assert result.success is False
    assert directory.require(record.agent_id).use_count == 0
    assert logs.list_agents() == []


def test_stable_log_keys_isolate_names_that_share_the_same_slug(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    first = directory.create(name="A B", purpose="First workflow")
    second = directory.create(name="A-B", purpose="Second workflow")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()

    for record, instruction in ((first, "first request"), (second, "second request")):
        dispatch_and_drain(
            agent_id=str(record.agent_id),
            instructions=instruction,
            dispatch_context=DispatchContext(),
            directory=directory,
            log_store=logs,
            batch_manager=batch,
        )

    assert "first request" in logs.load_transcript(str(first.agent_id))
    assert "second request" not in logs.load_transcript(str(first.agent_id))
    assert "second request" in logs.load_transcript(str(second.agent_id))


def test_abstain_context_rejects_all_dispatch(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    record = directory.create(name="Jordan", purpose="Ambiguous Jordan")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext(routing_action=RoutingAction.ABSTAIN)

    reuse = dispatch_and_drain(
        agent_id=str(record.agent_id),
        instructions="Do not run",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )
    create = dispatch_and_drain(
        agent_name="New Jordan",
        agent_purpose="Still ambiguous",
        instructions="Do not run",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    assert reuse.payload["code"] == "routing_not_authorized"
    assert create.payload["code"] == "routing_not_authorized"
    assert batch.submitted == []


def test_reuse_context_allows_only_recommended_candidate(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    allowed = directory.create(name="Alice", purpose="Alice messages")
    unseen = directory.create(name="Bob", purpose="Bob messages")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext(
        routing_action=RoutingAction.REUSE,
        allowed_agent_ids=frozenset({allowed.agent_id}),
    )

    rejected = dispatch_and_drain(
        agent_id=str(unseen.agent_id),
        instructions="Do not run",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )
    accepted = dispatch_and_drain(
        agent_id=str(allowed.agent_id),
        instructions="Run this",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    assert rejected.payload["code"] == "routing_not_authorized"
    assert accepted.success is True
    assert [item.agent_id for item in batch.submitted] == [str(allowed.agent_id)]


def test_create_context_rejects_reuse_but_allows_one_new_identity(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    existing = directory.create(name="Alice", purpose="Alice messages")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext(routing_action=RoutingAction.CREATE_NEW)

    rejected = dispatch_and_drain(
        agent_id=str(existing.agent_id),
        instructions="Do not run",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )
    accepted = dispatch_and_drain(
        agent_name="Bob invoices",
        agent_purpose="Track Bob invoices",
        instructions="Run this",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    assert rejected.payload["code"] == "routing_not_authorized"
    assert accepted.success is True
    assert accepted.payload["new_agent_created"] is True
