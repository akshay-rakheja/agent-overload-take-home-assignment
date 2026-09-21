"""Trace contracts for execution context, model, tool, and final response paths."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import pytest

from server.agents.execution_agent import runtime as execution_runtime
from server.agents.execution_agent.agent import ExecutionAgent
from server.agents.execution_agent.runtime import ExecutionAgentRuntime
from server.services.evaluation_lab.models import TraceContext, TraceEventKind
from server.services.evaluation_lab.trace import trace_scope
from server.services.execution.context_policy import ExecutionContextPolicy
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore


class CollectingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class FailingSink:
    def emit(self, event) -> None:  # pragma: no cover - failure sentinel
        raise RuntimeError("trace unavailable")


class FakeAgent:
    def __init__(self) -> None:
        self.name = "Fixture agent"
        self.responses = []
        self.tool_records = []

    def build_system_prompt_with_history(self) -> str:
        return "fixture system"

    def record_response(self, response: str) -> None:
        self.responses.append(response)

    def record_tool_execution(self, tool_name: str, arguments: str, result: str) -> None:
        self.tool_records.append((tool_name, arguments, result))


def _trace_context() -> TraceContext:
    return TraceContext(
        run_id=uuid4(),
        turn_id=uuid4(),
        system="enhanced",
        revision="fixture-revision",
        mode="fixture",
    )


def _events(sink: CollectingSink, kind: TraceEventKind):
    return [event for event in sink.events if event.kind is kind]


def _runtime() -> ExecutionAgentRuntime:
    runtime = ExecutionAgentRuntime.__new__(ExecutionAgentRuntime)
    runtime.agent = FakeAgent()
    runtime.api_key = "fixture-key"
    runtime.model = "fixture-model"
    runtime.lab_enabled = True
    runtime.tool_registry = {}
    runtime.tool_schemas = []
    return runtime


def test_context_trace_equals_last_metrics_after_real_builder_without_journal_mutation(
    tmp_path,
) -> None:
    directory = AgentDirectory(tmp_path / "roster.json")
    record = directory.create(
        name="Alice partnership",
        purpose="Track Alice",
        memory_summary="Alice prefers the two-year option.",
    )
    logs = ExecutionAgentLogStore(tmp_path / "logs")
    stable_id = str(record.agent_id)
    for index in range(8):
        logs.record_request(stable_id, f"request {index}")
        logs.record_agent_response(stable_id, f"response {index}")
    before = logs.read_raw_bytes(stable_id)
    agent = ExecutionAgent(
        record.name,
        storage_key=stable_id,
        agent_id=stable_id,
        log_store=logs,
        directory=directory,
        context_policy=ExecutionContextPolicy(
            max_recent_episodes=2,
            max_characters=450,
        ),
    )
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        prompt = agent.build_system_prompt_with_history()

    event = _events(sink, TraceEventKind.CONTEXT_METRICS)[0]
    assert event.payload == asdict(agent.last_context_metrics)
    assert event.payload["raw_entry_count"] == 16
    assert event.payload["included_episode_count"] <= 2
    assert event.payload["summary_used"] is True
    assert "# Execution History" in prompt
    assert logs.read_raw_bytes(stable_id) == before


def test_execution_runtime_traces_model_and_final_response_without_usage_or_cost(
    monkeypatch,
) -> None:
    response = {"choices": [{"message": {"content": "fixture complete"}}]}
    calls = []

    async def fake_chat_completion(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(
        execution_runtime, "request_chat_completion", fake_chat_completion
    )
    runtime = _runtime()
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        result = asyncio.run(runtime.execute("fixture instruction"))

    model_events = _events(sink, TraceEventKind.MODEL_CALL)
    final_event = _events(sink, TraceEventKind.FINAL_RESPONSE)[0]

    assert result.success is True
    assert result.response == "fixture complete"
    assert len(calls) == 1
    assert [event.payload["stage"] for event in model_events] == ["started", "completed"]
    assert model_events[1].payload["elapsed_ns"] >= 0
    assert model_events[1].payload["started_monotonic_ns"] <= model_events[1].payload["finished_monotonic_ns"]
    assert final_event.payload == {
        "response": "fixture complete",
        "agent_name": "Fixture agent",
    }
    assert _events(sink, TraceEventKind.USAGE) == []
    assert _events(sink, TraceEventKind.COST) == []


@pytest.mark.parametrize(
    ("tool_name", "expected_policy_code"),
    [
        ("gmail_create_draft", "mutation_blocked"),
        ("gmail_future_unclassified", "unknown_blocked"),
    ],
)
def test_execution_policy_rejections_trace_before_callable_execution(
    tool_name,
    expected_policy_code,
) -> None:
    called = False

    def forbidden_callable(**kwargs):  # pragma: no cover - failure sentinel
        nonlocal called
        called = True
        return kwargs

    runtime = _runtime()
    runtime.tool_registry = {tool_name: forbidden_callable}
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        success, result = asyncio.run(runtime._execute_tool(tool_name, {"body": "private"}))

    tool_event = _events(sink, TraceEventKind.TOOL_CALL)[-1]
    gmail_event = _events(sink, TraceEventKind.GMAIL_EVIDENCE)[0]

    assert success is False
    assert result["error"]["code"] == "lab_mutation_blocked"
    assert called is False
    assert tool_event.payload["stage"] == "rejected"
    assert gmail_event.payload == {
        "boundary": "execution_runtime",
        "operation_name": tool_name,
        "stage": "rejected",
        "allowed": False,
        "policy_code": expected_policy_code,
        "callable_executed": False,
    }


def test_execution_tool_lifecycle_traces_exact_computed_result() -> None:
    runtime = _runtime()
    runtime.lab_enabled = False
    expected = {"source": "fixture", "count": 3}
    runtime.tool_registry = {"fixture_tool": lambda **kwargs: expected}
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        success, result = asyncio.run(
            runtime._execute_tool("fixture_tool", {"page_size": 3})
        )

    tool_events = _events(sink, TraceEventKind.TOOL_CALL)
    assert success is True
    assert result is expected
    assert [event.payload["stage"] for event in tool_events] == ["started", "completed"]
    assert tool_events[1].payload["success"] is True
    assert tool_events[1].payload["result"] == expected


def test_execution_gmail_tool_trace_never_contains_mail_or_provider_payload() -> None:
    runtime = _runtime()
    private_markers = (
        "private-message-id",
        "private@example.invalid",
        "private clean body",
        "private provider payload",
    )
    expected = {
        "items": [
            {
                "message_id": private_markers[0],
                "sender": private_markers[1],
                "clean_text": private_markers[2],
            }
        ],
        "provider": private_markers[3],
    }
    runtime.tool_registry = {"gmail_get_contacts": lambda **kwargs: expected}
    sink = CollectingSink()

    with trace_scope(_trace_context(), sink):
        success, result = asyncio.run(runtime._execute_tool("gmail_get_contacts", {}))

    serialized = str(
        [event.model_dump(mode="json") for event in _events(sink, TraceEventKind.TOOL_CALL)]
    )
    assert success is True
    assert result is expected
    for marker in private_markers:
        assert marker not in serialized


def test_execution_sink_failure_preserves_success_and_failure_results(monkeypatch) -> None:
    runtime = _runtime()

    async def returning_chat_completion(**kwargs):
        del kwargs
        return {"choices": [{"message": {"content": "fixture complete"}}]}

    monkeypatch.setattr(
        execution_runtime, "request_chat_completion", returning_chat_completion
    )
    with trace_scope(_trace_context(), FailingSink()):
        success = asyncio.run(runtime.execute("fixture"))
    assert success.success is True
    assert success.response == "fixture complete"

    failure = RuntimeError("production model failure")

    async def raising_chat_completion(**kwargs):
        del kwargs
        raise failure

    monkeypatch.setattr(
        execution_runtime, "request_chat_completion", raising_chat_completion
    )
    runtime.agent = FakeAgent()
    with trace_scope(_trace_context(), FailingSink()):
        failed = asyncio.run(runtime.execute("fixture"))

    assert failed.success is False
    assert failed.error == "production model failure"
    assert failed.response == "Failed to complete task: production model failure"
