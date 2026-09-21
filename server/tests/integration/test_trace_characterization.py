"""Tracing must be observational across complete deterministic production turns."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from server.agents.execution_agent import agent as execution_agent_module
from server.agents.execution_agent import batch_manager as batch_manager_module
from server.agents.execution_agent import runtime as execution_runtime
from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.agents.interaction_agent import agent as interaction_agent
from server.agents.interaction_agent import runtime as interaction_runtime
from server.agents.interaction_agent import tools as interaction_tools
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.config import ModelCallConfig, ModelRole, Settings
from server.services.evaluation_lab.models import TraceContext
from server.services.evaluation_lab.trace import JsonlTraceStore, NullTraceSink, trace_scope
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore


UTC = timezone.utc
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
CREATED_ID = UUID("00000000-0000-4000-8000-000000000099")
RUN_ID = UUID("10000000-0000-4000-8000-000000000001")
TURN_ID = UUID("20000000-0000-4000-8000-000000000001")

# Keep immutable references so the trace-off and trace-on halves can install fresh
# capture wrappers without wrapping the wrapper from the first half.
_REAL_BUILD_CANDIDATE_CONTEXT = interaction_agent.build_candidate_context
_REAL_EXECUTION_RUNTIME = execution_runtime.ExecutionAgentRuntime


class FailingSink:
    def emit(self, event) -> None:
        del event
        raise RuntimeError("Bearer private-sink-token mailbox@example.invalid")


class CapturingBatchManager(ExecutionBatchManager):
    """Use real batch execution while terminating only the final feedback hop."""

    def __init__(self) -> None:
        super().__init__(timeout_seconds=10)
        self.dispatched_payloads: list[str] = []

    async def _dispatch_to_interaction_agent(self, payload: str) -> None:
        self.dispatched_payloads.append(payload)


class DeterministicInteractionTransport:
    """Fake only the external interaction-model transport."""

    def __init__(self, scenario: str, forbidden_agent_id: str | None) -> None:
        self.scenario = scenario
        self.forbidden_agent_id = forbidden_agent_id
        self.requests: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        model: str | None = None,
        config: ModelCallConfig | None = None,
        role: ModelRole | None = None,
        messages: list[dict[str, Any]],
        system: str,
        api_key: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del api_key, role
        effective_model = config.model_id if config is not None else model
        assert effective_model is not None
        self.requests.append(
            {
                "model": effective_model,
                "messages": copy.deepcopy(messages),
                "system": system,
                "tools": copy.deepcopy(tools),
            }
        )
        if len(self.requests) > 1 or self.scenario == "abstain":
            return self._response(
                content=f"interaction final: {self.scenario}",
                tool_calls=[],
            )

        candidate_ids = re.findall(
            r'<agent_candidate id="([^"]+)"', messages[0]["content"]
        )
        if self.scenario == "forbidden_dispatch":
            assert self.forbidden_agent_id is not None
            arguments = [
                {
                    "agent_id": self.forbidden_agent_id,
                    "instructions": "This must not run",
                }
            ]
        elif self.scenario in {"create", "repeated_create"}:
            create = {
                "agent_name": "Vendor contracts",
                "agent_purpose": "Track the new vendor contract",
                "creation_intent_id": "vendor-contract-turn",
                "instructions": "Review the vendor contract",
            }
            arguments = [create]
            if self.scenario == "repeated_create":
                arguments.append(dict(create))
        else:
            assert candidate_ids, "reuse scenario must expose a real rendered candidate"
            arguments = [
                {
                    "agent_id": candidate_ids[0],
                    "instructions": "Check the current status",
                }
            ]

        tool_calls = [
            {
                "id": f"interaction-call-{index}",
                "type": "function",
                "function": {
                    "name": "send_message_to_agent",
                    "arguments": json.dumps(argument, sort_keys=True),
                },
            }
            for index, argument in enumerate(arguments, start=1)
        ]
        return self._response(content="", tool_calls=tool_calls)

    @staticmethod
    def _response(*, content: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "choices": [
                {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
            ]
        }


class DeterministicExecutionTransport:
    """Fake only the external execution-model transport."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        model: str | None = None,
        config: ModelCallConfig | None = None,
        role: ModelRole | None = None,
        messages: list[dict[str, Any]],
        system: str,
        api_key: str,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        del api_key, role
        effective_model = config.model_id if config is not None else model
        assert effective_model is not None
        self.requests.append(
            {
                "model": effective_model,
                "messages": copy.deepcopy(messages),
                "system": system,
                "tools": copy.deepcopy(tools),
            }
        )
        instruction = messages[0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"execution final: {instruction}",
                        "tool_calls": [],
                    }
                }
            ]
        }


@dataclass(frozen=True)
class ScenarioSnapshot:
    files: tuple[tuple[str, bytes], ...]

    @classmethod
    def capture(cls, root: Path) -> "ScenarioSnapshot":
        return cls(
            files=tuple(
                (path.relative_to(root).as_posix(), path.read_bytes())
                for path in sorted(root.rglob("*"))
                if path.is_file() and not path.name.endswith(".lock")
            )
        )

    def restore(self, root: Path) -> None:
        root.mkdir(parents=True)
        for relative, payload in self.files:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


@dataclass(frozen=True)
class ScenarioOutcome:
    candidates: tuple[dict[str, object], ...]
    action: str
    recommendation: str | None
    allowed_dispatch_ids: tuple[str, ...]
    selected_identity: dict[str, object] | None
    created_identity: dict[str, object] | None
    directory_serialization: bytes
    idempotency_outcome: tuple[bool | None, ...]
    initial_interaction_messages: tuple[dict[str, Any], ...]
    final_interaction_messages: tuple[dict[str, Any], ...]
    interaction_model_requests: tuple[dict[str, Any], ...]
    interaction_tool_results: tuple[dict[str, Any], ...]
    execution_model_requests: tuple[dict[str, Any], ...]
    execution_contexts: tuple[dict[str, Any], ...]
    execution_batch_payloads: tuple[str, ...]
    raw_journal_bytes: tuple[tuple[str, bytes], ...]
    final_response: str


def _fixed_clock() -> datetime:
    return NOW


def _seed_snapshot(tmp_path: Path, scenario: str) -> ScenarioSnapshot:
    root = tmp_path / f"seed-{scenario}"
    execution_dir = root / "execution_agents"
    execution_dir.mkdir(parents=True)
    roster_path = execution_dir / "roster.json"

    if scenario == "legacy_migration":
        roster_path.write_text(json.dumps(["Legacy Alice"]), encoding="utf-8")
        (execution_dir / "legacy-alice.log").write_bytes(
            b'<agent_request timestamp="2026-09-21 08:00:00">legacy request</agent_request>\n'
        )
        return ScenarioSnapshot.capture(root)

    directory = AgentDirectory(
        roster_path,
        clock=_fixed_clock,
        id_factory=iter(
            UUID(f"00000000-0000-4000-8000-{index:012d}")
            for index in range(1, 20)
        ).__next__,
    )
    if scenario in {"reuse", "forbidden_dispatch", "long_history"}:
        alice = directory.create(
            name="Alice correspondence",
            purpose="Track Alice proposal messages",
            aliases=("Alice",),
            memory_summary="Alice prefers concise proposal updates.",
        )
        if scenario == "forbidden_dispatch":
            directory.create(
                name="Bob invoices",
                purpose="Track Bob invoices",
                aliases=("Bob",),
            )
        if scenario == "long_history":
            logs = ExecutionAgentLogStore(execution_dir)
            for index in range(120):
                logs.record_request(str(alice.agent_id), f"long request {index}")
                logs.record_agent_response(str(alice.agent_id), f"long response {index}")
    elif scenario == "abstain":
        directory.create(
            name="Jordan client",
            purpose="Manage client Jordan",
            aliases=("Jordan",),
        )
        directory.create(
            name="Jordan candidate",
            purpose="Recruit candidate Jordan",
            aliases=("Jordan",),
        )
    elif scenario not in {"create", "repeated_create"}:
        raise AssertionError(f"unknown scenario: {scenario}")
    return ScenarioSnapshot.capture(root)


def _candidate_facts(context) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "agent_id": str(candidate.agent_id),
            "name": candidate.name,
            "purpose": candidate.purpose,
            "status": candidate.status.value,
            "score": candidate.score,
            "score_components": dict(candidate.score_components),
            "reasons": tuple(candidate.reasons),
        }
        for candidate in context.candidates
    )


async def _run_scenario_async(
    root: Path,
    scenario: str,
    *,
    sink,
    monkeypatch: pytest.MonkeyPatch,
) -> ScenarioOutcome:
    execution_dir = root / "execution_agents"
    trace_context = TraceContext(
        run_id=RUN_ID,
        turn_id=TURN_ID,
        system="enhanced",
        revision="fixture-revision",
        mode="characterization",
    )
    settings = Settings(
        server_host="127.0.0.1",
        openrouter_api_key="fixture-key",
        lab_enabled=False,
        execution_context_max_recent_episodes=3,
        execution_context_max_characters=1_200,
    )
    queries = {
        "reuse": ("Did Alice reply to the proposal?", "Alice is reviewing it."),
        "create": ("Start tracking a brand new vendor contract", ""),
        "repeated_create": ("Start tracking a brand new vendor contract", ""),
        "abstain": ("Email Jordan the update", ""),
        "forbidden_dispatch": ("Did Alice reply to the proposal?", ""),
        "legacy_migration": ("Did Legacy Alice reply?", ""),
        "long_history": ("Did Alice reply to the proposal?", "Alice owns it."),
    }
    query, transcript = queries[scenario]

    with trace_scope(trace_context, sink):
        # Construction is intentionally inside the trace scope because it may
        # perform the production legacy-roster migration.
        directory = AgentDirectory(
            execution_dir / "roster.json",
            clock=_fixed_clock,
            id_factory=lambda: CREATED_ID,
        )
        logs = ExecutionAgentLogStore(execution_dir)
        forbidden_agent_id = next(
            (
                str(record.agent_id)
                for record in directory.list_records()
                if record.name == "Bob invoices"
            ),
            None,
        )
        interaction_transport = DeterministicInteractionTransport(
            scenario, forbidden_agent_id
        )
        execution_transport = DeterministicExecutionTransport()
        batch = CapturingBatchManager()
        captured_contexts = []
        execution_instances = []

        def capture_candidate_context(*args, **kwargs):
            context = _REAL_BUILD_CANDIDATE_CONTEXT(*args, **kwargs)
            captured_contexts.append(context)
            return context

        class CapturingExecutionRuntime(_REAL_EXECUTION_RUNTIME):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                execution_instances.append(self)

        monkeypatch.setattr(
            interaction_runtime, "build_candidate_context", capture_candidate_context
        )
        monkeypatch.setattr(
            interaction_runtime, "request_chat_completion", interaction_transport
        )
        monkeypatch.setattr(interaction_agent, "get_settings", lambda: settings)
        monkeypatch.setattr(interaction_tools, "get_agent_directory", lambda: directory)
        monkeypatch.setattr(
            interaction_tools, "get_execution_agent_logs", lambda: logs
        )
        monkeypatch.setattr(interaction_tools, "_EXECUTION_BATCH_MANAGER", batch)
        monkeypatch.setattr(
            execution_agent_module, "get_agent_directory", lambda: directory
        )
        monkeypatch.setattr(
            execution_agent_module, "get_execution_agent_logs", lambda: logs
        )
        monkeypatch.setattr(execution_agent_module, "get_settings", lambda: settings)
        monkeypatch.setattr(execution_runtime, "get_settings", lambda: settings)
        monkeypatch.setattr(
            execution_runtime, "request_chat_completion", execution_transport
        )
        monkeypatch.setattr(
            batch_manager_module, "ExecutionAgentRuntime", CapturingExecutionRuntime
        )

        runtime = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
        runtime.api_key = settings.openrouter_api_key
        runtime.model = settings.interaction_agent_model
        runtime.settings = settings
        runtime.tool_schemas = interaction_tools.get_tool_schemas()
        runtime.agent_directory = directory
        runtime.dispatch_context = interaction_tools.DispatchContext()

        messages = runtime._prepare_turn_messages(
            query, transcript, message_type="user"
        )
        initial_messages = copy.deepcopy(messages)
        summary = await runtime._run_interaction_loop(
            interaction_agent.build_system_prompt(), messages
        )
        final_response = runtime._finalize_response(summary)

        # Real send_message_to_agent schedules execution asynchronously. Await
        # every task created by this turn so persistence and batch dispatch are
        # captured before leaving the trace scope.
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        if pending:
            await asyncio.gather(*pending)

        assert len(captured_contexts) == 1
        candidate_context = captured_contexts[0]
        tool_results = tuple(
            json.loads(message["content"])
            for message in messages
            if message.get("role") == "tool"
        )
        successful_results = [
            item["result"]
            for item in tool_results
            if item["status"] == "success"
        ]
        selected = None
        created = None
        if successful_results:
            selected_record = directory.require(successful_results[-1]["agent_id"])
            selected = selected_record.model_dump(mode="json")
            if any(result["new_agent_created"] for result in successful_results):
                created = selected

        records = directory.list_records()
        outcome = ScenarioOutcome(
            candidates=_candidate_facts(candidate_context),
            action=candidate_context.decision.action.value,
            recommendation=(
                str(candidate_context.decision.agent_id)
                if candidate_context.decision.agent_id is not None
                else None
            ),
            allowed_dispatch_ids=tuple(
                sorted(str(value) for value in runtime.dispatch_context.allowed_agent_ids)
            ),
            selected_identity=selected,
            created_identity=created,
            directory_serialization=(execution_dir / "roster.json").read_bytes(),
            idempotency_outcome=tuple(
                result.get("new_agent_created") for result in successful_results
            ),
            initial_interaction_messages=tuple(initial_messages),
            final_interaction_messages=tuple(copy.deepcopy(messages)),
            interaction_model_requests=tuple(interaction_transport.requests),
            interaction_tool_results=tool_results,
            execution_model_requests=tuple(execution_transport.requests),
            execution_contexts=tuple(
                asdict(instance.agent.last_context_metrics)
                for instance in execution_instances
            ),
            execution_batch_payloads=tuple(batch.dispatched_payloads),
            raw_journal_bytes=tuple(
                (path.name, path.read_bytes())
                for path in sorted(execution_dir.glob("*.log"))
            ),
            final_response=final_response,
        )

        # Ensure the final serialized identity facts are from the same real
        # directory that was used by routing and execution.
        assert len(records) == len(
            json.loads(outcome.directory_serialization)["agents"]
        )
        return outcome


def _run_scenario(
    root: Path,
    scenario: str,
    *,
    sink,
    monkeypatch: pytest.MonkeyPatch,
) -> ScenarioOutcome:
    return asyncio.run(
        _run_scenario_async(
            root,
            scenario,
            sink=sink,
            monkeypatch=monkeypatch,
        )
    )


@pytest.mark.parametrize(
    ("scenario", "expected_action", "expected_tools", "expected_executions"),
    [
        ("reuse", "reuse", 1, 1),
        ("create", "create_new", 1, 1),
        ("abstain", "abstain", 0, 0),
        ("repeated_create", "create_new", 2, 2),
        ("forbidden_dispatch", "reuse", 1, 0),
        ("legacy_migration", "reuse", 1, 1),
        ("long_history", "reuse", 1, 1),
    ],
)
def test_trace_on_restores_the_same_snapshot_and_preserves_every_production_fact(
    tmp_path,
    monkeypatch,
    scenario: str,
    expected_action: str,
    expected_tools: int,
    expected_executions: int,
) -> None:
    monkeypatch.setattr(
        "server.services.execution.log_store.now_in_user_timezone",
        lambda _format: "2026-09-21 08:00:00",
    )
    snapshot = _seed_snapshot(tmp_path, scenario)
    trace_off_root = tmp_path / f"trace-off-{scenario}"
    trace_on_root = tmp_path / f"trace-on-{scenario}"
    snapshot.restore(trace_off_root)
    snapshot.restore(trace_on_root)
    assert ScenarioSnapshot.capture(trace_off_root) == snapshot
    assert ScenarioSnapshot.capture(trace_on_root) == snapshot

    with monkeypatch.context() as off_patch:
        trace_off = _run_scenario(
            trace_off_root,
            scenario,
            sink=NullTraceSink(),
            monkeypatch=off_patch,
        )
    trace_store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    with monkeypatch.context() as on_patch:
        trace_on = _run_scenario(
            trace_on_root,
            scenario,
            sink=trace_store,
            monkeypatch=on_patch,
        )

    assert trace_on == trace_off
    assert trace_store.path_for(RUN_ID).is_file()

    # Non-vacuous scenario evidence from the actual production turn.
    assert trace_on.action == expected_action
    assert len(trace_on.interaction_tool_results) == expected_tools
    assert len(trace_on.execution_model_requests) == expected_executions
    assert trace_on.final_response == f"interaction final: {scenario}"
    assert "<agent_candidates" in trace_on.initial_interaction_messages[0]["content"]
    assert trace_on.interaction_model_requests

    if scenario == "abstain":
        assert trace_on.allowed_dispatch_ids == ()
        assert trace_on.selected_identity is None
        assert trace_on.execution_batch_payloads == ()
    elif scenario == "forbidden_dispatch":
        assert trace_on.selected_identity is None
        assert trace_on.execution_batch_payloads == ()
        assert trace_on.interaction_tool_results[0]["status"] == "error"
        assert (
            trace_on.interaction_tool_results[0]["error"]["code"]
            == "routing_not_authorized"
        )
    else:
        assert trace_on.selected_identity is not None
        assert trace_on.execution_batch_payloads
        assert all("[SUCCESS]" in item for item in trace_on.execution_batch_payloads)

    if scenario == "repeated_create":
        assert trace_on.idempotency_outcome == (True, False)
        assert len(trace_on.execution_batch_payloads) == 1
        assert trace_on.execution_batch_payloads[0].count("[SUCCESS]") == 2
        assert len(
            {
                result["result"]["agent_id"]
                for result in trace_on.interaction_tool_results
            }
        ) == 1
    if scenario == "legacy_migration":
        assert trace_on.selected_identity["name"] == "Legacy Alice"
        assert trace_on.selected_identity["legacy_storage_key"] == "Legacy Alice"
    if scenario == "long_history":
        assert trace_on.execution_contexts[0]["omitted_entry_count"] > 0
        system_prompt = trace_on.execution_model_requests[0]["system"]
        assert "long request 119" in system_prompt
        assert "long request 0" not in system_prompt


def test_failing_sink_preserves_result_and_logs_only_sanitized_degradation(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        "server.services.execution.log_store.now_in_user_timezone",
        lambda _format: "2026-09-21 08:00:00",
    )
    snapshot = _seed_snapshot(tmp_path, "repeated_create")
    control_root = tmp_path / "control"
    failure_root = tmp_path / "failure"
    snapshot.restore(control_root)
    snapshot.restore(failure_root)

    with monkeypatch.context() as control_patch:
        control = _run_scenario(
            control_root,
            "repeated_create",
            sink=NullTraceSink(),
            monkeypatch=control_patch,
        )
    with monkeypatch.context() as failure_patch:
        failed = _run_scenario(
            failure_root,
            "repeated_create",
            sink=FailingSink(),
            monkeypatch=failure_patch,
        )

    assert failed == control
    assert failed.idempotency_outcome == (True, False)
    assert len(failed.execution_model_requests) == 2
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "server.services.evaluation_lab.trace"
    ]
    assert messages
    assert set(messages) == {"Evaluation trace unavailable; observability degraded."}
    assert "private-sink-token" not in caplog.text
    assert "mailbox@example.invalid" not in caplog.text


@pytest.mark.parametrize(
    ("target", "attribute"),
    [
        (InteractionAgentRuntime, "_prepare_turn_messages"),
        (InteractionAgentRuntime, "_run_interaction_loop"),
        (interaction_agent, "render_agent_candidates"),
    ],
)
def test_characterization_gate_breaks_when_production_boundary_breaks(
    tmp_path,
    monkeypatch,
    target,
    attribute: str,
) -> None:
    snapshot = _seed_snapshot(tmp_path, "reuse")
    root = tmp_path / f"broken-{attribute}"
    snapshot.restore(root)

    if attribute == "_run_interaction_loop":
        async def broken(*args, **kwargs):
            del args, kwargs
            raise AssertionError("production boundary is broken")
    else:
        def broken(*args, **kwargs):
            del args, kwargs
            raise AssertionError("production boundary is broken")

    monkeypatch.setattr(target, attribute, broken)
    with pytest.raises(AssertionError, match="production boundary is broken"):
        _run_scenario(
            root,
            "reuse",
            sink=NullTraceSink(),
            monkeypatch=monkeypatch,
        )
