"""Trace contracts for enhanced interaction routing and dispatch."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest

from server.agents.interaction_agent import agent as interaction_agent
from server.agents.interaction_agent import runtime as interaction_runtime
from server.agents.interaction_agent.agent import (
    CandidateContext,
    build_candidate_context,
    prepare_message_with_history,
)
from server.agents.interaction_agent.runtime import (
    InteractionAgentRuntime,
    _LoopSummary,
    _ToolCall,
)
from server.agents.interaction_agent.tools import (
    DispatchContext,
    ToolResult,
    send_message_to_agent,
)
from server.services.evaluation_lab.models import TraceContext, TraceEventKind
from server.services.evaluation_lab.trace import NullTraceSink, trace_scope
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.models import AgentRecord, AgentStatus
from server.services.execution.routing import RoutingAction, RoutingDecision


UTC = timezone.utc
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


class CollectingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class FailingSink:
    def emit(self, event) -> None:  # pragma: no cover - failure sentinel
        raise RuntimeError("private sink failure")


class StubDirectory:
    def __init__(self, records: list[AgentRecord]) -> None:
        self.records = records
        self.list_calls = 0

    def list_records(self) -> list[AgentRecord]:
        self.list_calls += 1
        return list(self.records)


class FakeBatchManager:
    def __init__(self) -> None:
        self.submitted = []

    async def execute_agent(
        self,
        agent_name: str,
        instructions: str,
        request_id: str | None = None,
        agent_id: str | None = None,
        legacy_storage_key: str | None = None,
    ):
        del request_id, legacy_storage_key
        self.submitted.append((agent_name, instructions, agent_id))
        return SimpleNamespace(success=True)


def _trace_context() -> TraceContext:
    return TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture-revision",
        mode="fixture",
    )


def _record(
    index: int,
    *,
    name: str,
    purpose: str,
    aliases: tuple[str, ...],
) -> AgentRecord:
    return AgentRecord(
        agent_id=uuid5(NAMESPACE_URL, f"trace-candidate:{index}"),
        name=name,
        purpose=purpose,
        aliases=aliases,
        status=AgentStatus.HOT,
        created_at=NOW - timedelta(days=10),
        last_used_at=NOW - timedelta(days=index),
        use_count=index,
    )


def _events(sink: CollectingSink, kind: TraceEventKind):
    return [event for event in sink.events if event.kind is kind]


def _dispatch_and_drain(**kwargs):
    async def run():
        result = send_message_to_agent(**kwargs)
        await asyncio.sleep(0)
        return result

    return asyncio.run(run())


def _traced_dispatch(**kwargs):
    sink = CollectingSink()
    with trace_scope(_trace_context(), sink):
        result = _dispatch_and_drain(**kwargs)
    return result, sink.events


def test_candidate_trace_matches_returned_context_and_exact_xml_without_second_work(
    monkeypatch,
) -> None:
    alice = _record(
        1,
        name="Alice correspondence",
        purpose="Track Alice proposal messages",
        aliases=("Alice",),
    )
    bob = _record(
        2,
        name="Bob invoices",
        purpose="Track Bob invoices",
        aliases=("Bob",),
    )
    directory = StubDirectory([alice, bob])
    render_calls = 0
    original_render = interaction_agent.render_agent_candidates

    def counted_render(context):
        nonlocal render_calls
        render_calls += 1
        return original_render(context)

    monkeypatch.setattr(interaction_agent, "render_agent_candidates", counted_render)
    sink = CollectingSink()
    with trace_scope(_trace_context(), sink):
        candidate_context = build_candidate_context(
            "Did Alice reply?", "", directory=directory
        )
        messages = prepare_message_with_history(
            "Did Alice reply?",
            "",
            directory=directory,
            candidate_context=candidate_context,
        )

    candidate_event = _events(sink, TraceEventKind.CANDIDATES)[0]
    prompt_event = _events(sink, TraceEventKind.PROMPT_EXPOSURE)[0]
    traced_candidates = candidate_event.payload["candidates"]

    assert directory.list_calls == 1
    assert render_calls == 1
    assert [item["agent_id"] for item in traced_candidates] == [
        str(candidate.agent_id) for candidate in candidate_context.candidates
    ]
    assert [item["score"] for item in traced_candidates] == [
        candidate.score for candidate in candidate_context.candidates
    ]
    assert [dict(item["score_components"]) for item in traced_candidates] == [
        candidate.score_components for candidate in candidate_context.candidates
    ]
    assert [tuple(item["reasons"]) for item in traced_candidates] == [
        candidate.reasons for candidate in candidate_context.candidates
    ]
    assert tuple(prompt_event.payload["candidate_ids"]) == tuple(
        str(candidate.agent_id) for candidate in candidate_context.prompt_candidates
    )
    assert prompt_event.payload["candidate_xml"] in messages[0]["content"]
    assert prompt_event.payload["candidate_count"] == len(
        candidate_context.prompt_candidates
    )


def test_candidate_trace_boundary_never_exposes_more_than_five() -> None:
    candidates = tuple(
        interaction_agent.AgentCandidate(
            agent_id=uuid5(NAMESPACE_URL, f"manual-trace-candidate:{index}"),
            name=f"Candidate {index}",
            purpose=f"Purpose {index}",
            status=AgentStatus.HOT,
            score=1.0 - index / 100,
            score_components={"exact_match": 0.7},
            reasons=("fixture",),
        )
        for index in range(7)
    )
    context = CandidateContext(
        candidates=candidates,
        decision=RoutingDecision(
            action=RoutingAction.REUSE,
            agent_id=candidates[0].agent_id,
            confidence=candidates[0].score,
            reasons=("fixture",),
        ),
    )
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        message = prepare_message_with_history(
            "fixture", "", candidate_context=context
        )[0]["content"]

    prompt_event = _events(sink, TraceEventKind.PROMPT_EXPOSURE)[0]
    assert prompt_event.payload["candidate_count"] == 5
    assert len(prompt_event.payload["candidate_ids"]) == 5
    assert message.count("<agent_candidate ") == 5


def test_runtime_traces_exact_routing_decision_and_dispatch_authorization() -> None:
    alice = _record(
        1,
        name="Alice",
        purpose="Track Alice",
        aliases=("Alice",),
    )
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    runtime.agent_directory = StubDirectory([alice])
    runtime.dispatch_context = DispatchContext()
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        messages = runtime._prepare_turn_messages(
            "Did Alice reply?", "", message_type="user"
        )

    routing = _events(sink, TraceEventKind.ROUTING_DECISION)[0].payload
    authorization = _events(sink, TraceEventKind.AUTHORIZATION)[0].payload

    assert messages[0]["content"]
    assert routing == {
        "action": runtime.dispatch_context.routing_action.value,
        "agent_id": str(alice.agent_id),
        "confidence": routing["confidence"],
        "reasons": routing["reasons"],
        "recommendation": str(alice.agent_id),
    }
    assert authorization["routing_action"] == RoutingAction.REUSE.value
    assert tuple(authorization["authorized_ids"]) == (str(alice.agent_id),)
    assert runtime.dispatch_context.allowed_agent_ids == frozenset({alice.agent_id})


def test_interaction_model_tool_and_final_response_lifecycle_are_traced(
    monkeypatch,
) -> None:
    response = {"choices": [{"message": {"content": "fixture response"}}]}
    tool_result = ToolResult(success=True, payload={"status": "fixture"})
    model_calls = []
    tool_calls = []

    async def fake_chat_completion(**kwargs):
        model_calls.append(kwargs)
        return response

    def fake_handle_tool_call(name, arguments, *, dispatch_context):
        tool_calls.append((name, arguments, dispatch_context))
        return tool_result

    monkeypatch.setattr(
        interaction_runtime, "request_chat_completion", fake_chat_completion
    )
    monkeypatch.setattr(interaction_runtime, "handle_tool_call", fake_handle_tool_call)
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    runtime.model = "fixture-model"
    runtime.api_key = "fixture-key"
    runtime.tool_schemas = [{"fixture": True}]
    runtime.dispatch_context = DispatchContext()
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        actual_response = asyncio.run(runtime._make_llm_call("system", [{"role": "user"}]))
        actual_tool_result = runtime._execute_tool(
            _ToolCall(identifier="call-1", name="wait", arguments={"reason": "fixture"})
        )
        final_response = runtime._finalize_response(
            _LoopSummary(last_assistant_text="fixture response")
        )

    model_events = _events(sink, TraceEventKind.MODEL_CALL)
    tool_events = _events(sink, TraceEventKind.TOOL_CALL)
    final_event = _events(sink, TraceEventKind.FINAL_RESPONSE)[0]

    assert actual_response is response
    assert actual_tool_result is tool_result
    assert final_response == "fixture response"
    assert len(model_calls) == 1
    assert len(tool_calls) == 1
    assert [event.payload["stage"] for event in model_events] == ["started", "completed"]
    assert model_events[1].payload["elapsed_ns"] >= 0
    assert model_events[1].payload["started_monotonic_ns"] <= model_events[1].payload["finished_monotonic_ns"]
    assert [event.payload["stage"] for event in tool_events] == ["started", "completed"]
    assert tool_events[1].payload["success"] is True
    assert tool_events[1].payload["result"] == {"status": "fixture"}
    assert final_event.payload == {"response": "fixture response"}


def test_trace_sink_failure_preserves_interaction_results_and_exceptions(
    monkeypatch,
) -> None:
    response = {"choices": [{"message": {"content": "fixture"}}]}

    async def returning_chat_completion(**kwargs):
        del kwargs
        return response

    monkeypatch.setattr(
        interaction_runtime, "request_chat_completion", returning_chat_completion
    )
    runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    runtime.model = "fixture-model"
    runtime.api_key = "fixture-key"
    runtime.tool_schemas = []

    with trace_scope(_trace_context(), FailingSink()):
        assert asyncio.run(runtime._make_llm_call("system", [])) is response

    failure = RuntimeError("production failure")

    async def raising_chat_completion(**kwargs):
        del kwargs
        raise failure

    monkeypatch.setattr(
        interaction_runtime, "request_chat_completion", raising_chat_completion
    )
    with trace_scope(_trace_context(), FailingSink()):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(runtime._make_llm_call("system", []))

    assert exc_info.value is failure


def test_dispatch_traces_recommended_reuse_and_rejects_valid_nonrecommended_id(
    tmp_path,
) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    allowed = directory.create(name="Alice", purpose="Alice messages")
    unseen = directory.create(name="Bob", purpose="Bob messages")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext(
        routing_action=RoutingAction.REUSE,
        allowed_agent_ids=frozenset({allowed.agent_id}),
    )

    rejected, rejected_events = _traced_dispatch(
        agent_id=str(unseen.agent_id),
        instructions="Do not run",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )
    accepted, accepted_events = _traced_dispatch(
        agent_id=str(allowed.agent_id),
        instructions="Run this",
        dispatch_context=context,
        directory=directory,
        log_store=logs,
        batch_manager=batch,
    )

    rejected_result = next(
        event.payload
        for event in rejected_events
        if event.kind is TraceEventKind.DISPATCH_RESULT
    )
    accepted_result = next(
        event.payload
        for event in accepted_events
        if event.kind is TraceEventKind.DISPATCH_RESULT
    )
    identity = next(
        event.payload
        for event in accepted_events
        if event.kind is TraceEventKind.IDENTITY
    )

    assert rejected.payload["code"] == "routing_not_authorized"
    assert rejected_result["status"] == "rejected"
    assert rejected_result["code"] == "routing_not_authorized"
    assert rejected_result["directory_count_before"] == 2
    assert rejected_result["directory_count_after"] == 2
    assert accepted.success is True
    assert accepted_result["status"] == "accepted"
    assert accepted_result["selected_agent_id"] == str(allowed.agent_id)
    assert accepted_result["directory_count_before"] == 2
    assert accepted_result["directory_count_after"] == 2
    assert accepted_result["journal_sha256_before"] == sha256(b"").hexdigest()
    assert accepted_result["journal_sha256_after"] == sha256(
        logs.read_raw_bytes(str(allowed.agent_id))
    ).hexdigest()
    assert identity["selected"]["agent_id"] == str(allowed.agent_id)
    assert "created" not in identity
    assert batch.submitted == [("Alice", "Run this", str(allowed.agent_id))]


@pytest.mark.parametrize("routing_action", [RoutingAction.REUSE, RoutingAction.ABSTAIN])
def test_dispatch_traces_creation_rejected_by_reuse_or_abstain(
    tmp_path,
    routing_action,
) -> None:
    directory = AgentDirectory(tmp_path / f"{routing_action.value}.json")
    logs = ExecutionAgentLogStore(tmp_path / f"{routing_action.value}-logs")

    result, events = _traced_dispatch(
        agent_name="New workflow",
        agent_purpose="Handle new workflow",
        instructions="Do not run",
        dispatch_context=DispatchContext(routing_action=routing_action),
        directory=directory,
        log_store=logs,
        batch_manager=FakeBatchManager(),
    )

    traced = next(
        event.payload
        for event in events
        if event.kind is TraceEventKind.DISPATCH_RESULT
    )
    assert result.payload["code"] == "routing_not_authorized"
    assert traced["status"] == "rejected"
    assert traced["code"] == "routing_not_authorized"
    assert traced["directory_count_before"] == 0
    assert traced["directory_count_after"] == 0


def test_dispatch_traces_authorized_creation_and_idempotent_retry(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext(routing_action=RoutingAction.CREATE_NEW)
    kwargs = {
        "agent_name": "Vendor review",
        "agent_purpose": "Review vendor",
        "creation_intent_id": "vendor-one",
        "instructions": "Review vendor",
        "dispatch_context": context,
        "directory": directory,
        "log_store": logs,
        "batch_manager": batch,
    }

    first, first_events = _traced_dispatch(**kwargs)
    first_journal = logs.read_raw_bytes(first.payload["agent_id"])
    second, second_events = _traced_dispatch(**kwargs)

    first_result = next(
        event.payload
        for event in first_events
        if event.kind is TraceEventKind.DISPATCH_RESULT
    )
    second_result = next(
        event.payload
        for event in second_events
        if event.kind is TraceEventKind.DISPATCH_RESULT
    )
    first_identity = next(
        event.payload for event in first_events if event.kind is TraceEventKind.IDENTITY
    )
    second_identity = next(
        event.payload for event in second_events if event.kind is TraceEventKind.IDENTITY
    )

    assert first.payload["agent_id"] == second.payload["agent_id"]
    assert first_result["directory_count_before"] == 0
    assert first_result["directory_count_after"] == 1
    assert first_result["idempotent_creation"] is False
    assert first_identity["created"]["agent_id"] == first.payload["agent_id"]
    assert second_result["directory_count_before"] == 1
    assert second_result["directory_count_after"] == 1
    assert second_result["idempotent_creation"] is True
    assert second_result["journal_sha256_before"] == sha256(first_journal).hexdigest()
    assert second_identity["idempotent_creation"] is True
    assert "created" not in second_identity
    assert len(directory.list_records()) == 1


def test_dispatch_traces_intentional_multi_create_as_distinct_identities(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    batch = FakeBatchManager()
    context = DispatchContext(routing_action=RoutingAction.CREATE_NEW)

    results = []
    identity_events = []
    for intent in ("vendor-one", "vendor-two"):
        result, events = _traced_dispatch(
            agent_name="Vendor review",
            agent_purpose="Review vendor",
            creation_intent_id=intent,
            instructions=f"Review {intent}",
            dispatch_context=context,
            directory=directory,
            log_store=logs,
            batch_manager=batch,
        )
        results.append(result)
        identity_events.append(
            next(
                event.payload
                for event in events
                if event.kind is TraceEventKind.IDENTITY
            )
        )

    assert results[0].payload["agent_id"] != results[1].payload["agent_id"]
    assert [event["idempotent_creation"] for event in identity_events] == [False, False]
    assert [event["delta"]["directory_count_after"] for event in identity_events] == [1, 2]
    assert len(directory.list_records()) == 2


def test_null_trace_scope_adds_no_dispatch_observation_reads(tmp_path) -> None:
    class CountingDirectory(AgentDirectory):
        list_calls = 0

        def list_records(self):
            self.list_calls += 1
            return super().list_records()

    class ObservationGuardLogs(ExecutionAgentLogStore):
        def read_raw_bytes(self, agent_name):  # pragma: no cover - failure sentinel
            raise AssertionError(f"unexpected trace read for {agent_name}")

    directory = CountingDirectory(tmp_path / "roster.json")
    record = directory.create(name="Alice", purpose="Track Alice")
    logs = ObservationGuardLogs(tmp_path / "logs")

    with trace_scope(_trace_context(), NullTraceSink()):
        result = _dispatch_and_drain(
            agent_id=str(record.agent_id),
            instructions="Run this",
            dispatch_context=DispatchContext(
                routing_action=RoutingAction.REUSE,
                allowed_agent_ids=frozenset({record.agent_id}),
            ),
            directory=directory,
            log_store=logs,
            batch_manager=FakeBatchManager(),
        )

    assert result.success is True
    assert directory.list_calls == 0


def test_dispatch_trace_sink_failure_does_not_change_result(tmp_path) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    record = directory.create(name="Alice", purpose="Track Alice")
    logs = ExecutionAgentLogStore(tmp_path / "logs")

    with trace_scope(_trace_context(), FailingSink()):
        result = _dispatch_and_drain(
            agent_id=str(record.agent_id),
            instructions="Run this",
            dispatch_context=DispatchContext(
                routing_action=RoutingAction.REUSE,
                allowed_agent_ids=frozenset({record.agent_id}),
            ),
            directory=directory,
            log_store=logs,
            batch_manager=FakeBatchManager(),
        )

    assert result.success is True
    assert result.payload["agent_id"] == str(record.agent_id)
    assert json.loads((tmp_path / "roster.json").read_text())["agents"][0]["use_count"] == 1
