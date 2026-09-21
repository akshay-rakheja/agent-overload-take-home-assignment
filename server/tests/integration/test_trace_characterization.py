"""Tracing must be observational for every enhanced routing boundary."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from server.agents.execution_agent.agent import ExecutionAgent
from server.agents.interaction_agent.agent import build_candidate_context
from server.agents.interaction_agent.runtime import InteractionAgentRuntime, _LoopSummary
from server.agents.interaction_agent.tools import DispatchContext, send_message_to_agent
from server.services.evaluation_lab.models import TraceContext
from server.services.evaluation_lab.trace import JsonlTraceStore, NullTraceSink, trace_scope
from server.services.execution.context_policy import ExecutionContextPolicy
from server.services.execution.directory import AgentDirectory
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.routing import RoutingAction


UTC = timezone.utc
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
CREATED_ID = UUID("00000000-0000-4000-8000-000000000099")


class RecordingBatchManager:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, str, str | None, str | None]] = []

    async def execute_agent(
        self,
        agent_name: str,
        instructions: str,
        request_id: str | None = None,
        agent_id: str | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        self.submissions.append((agent_name, instructions, request_id, agent_id))
        return SimpleNamespace(success=True)


class FailingSink:
    def emit(self, event) -> None:
        del event
        raise RuntimeError("Bearer private-sink-token mailbox@example.invalid")


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
    routing_context: dict[str, object]
    execution_prompt: str | None
    execution_context: dict[str, object] | None
    raw_journal_bytes: tuple[tuple[str, bytes], ...]
    tool_results: tuple[dict[str, object], ...]
    batch_submissions: tuple[tuple[str, str, str | None, str | None], ...]
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


def _drain_dispatch(**kwargs):
    async def run():
        result = send_message_to_agent(**kwargs)
        await asyncio.sleep(0)
        return result

    return asyncio.run(run())


def _run_scenario(
    root: Path,
    scenario: str,
    *,
    sink,
) -> ScenarioOutcome:
    execution_dir = root / "execution_agents"
    directory = AgentDirectory(
        execution_dir / "roster.json",
        clock=_fixed_clock,
        id_factory=lambda: CREATED_ID,
    )
    logs = ExecutionAgentLogStore(execution_dir)
    batch = RecordingBatchManager()
    trace_context = TraceContext(
        run_id=UUID("10000000-0000-4000-8000-000000000001"),
        turn_id=UUID("20000000-0000-4000-8000-000000000001"),
        system="enhanced",
        revision="fixture-revision",
        mode="characterization",
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
        candidate_context = build_candidate_context(
            query,
            transcript,
            directory=directory,
        )
        allowed_ids = (
            frozenset({candidate_context.decision.agent_id})
            if candidate_context.decision.action is RoutingAction.REUSE
            and candidate_context.decision.agent_id is not None
            else frozenset()
        )
        dispatch_context = DispatchContext(
            routing_action=candidate_context.decision.action,
            allowed_agent_ids=allowed_ids,
        )

        results = []
        if scenario in {"reuse", "legacy_migration", "long_history"}:
            assert candidate_context.decision.agent_id is not None
            results.append(
                _drain_dispatch(
                    agent_id=str(candidate_context.decision.agent_id),
                    instructions="Check the current status",
                    dispatch_context=dispatch_context,
                    directory=directory,
                    log_store=logs,
                    batch_manager=batch,
                )
            )
        elif scenario in {"create", "repeated_create"}:
            create_kwargs = {
                "agent_name": "Vendor contracts",
                "agent_purpose": "Track the new vendor contract",
                "creation_intent_id": "vendor-contract-turn",
                "instructions": "Review the vendor contract",
                "dispatch_context": dispatch_context,
                "directory": directory,
                "log_store": logs,
                "batch_manager": batch,
            }
            results.append(_drain_dispatch(**create_kwargs))
            if scenario == "repeated_create":
                results.append(_drain_dispatch(**create_kwargs))
        elif scenario == "forbidden_dispatch":
            forbidden = next(
                record
                for record in directory.list_records()
                if record.name == "Bob invoices"
            )
            results.append(
                _drain_dispatch(
                    agent_id=str(forbidden.agent_id),
                    instructions="This must not run",
                    dispatch_context=dispatch_context,
                    directory=directory,
                    log_store=logs,
                    batch_manager=batch,
                )
            )

        selected = None
        created = None
        if results and results[-1].success:
            selected_record = directory.require(results[-1].payload["agent_id"])
            selected = selected_record.model_dump(mode="json")
            if any(result.payload.get("new_agent_created") for result in results):
                created = selected

        execution_prompt = None
        execution_context = None
        if selected is not None:
            selected_record = directory.require(selected["agent_id"])
            agent = ExecutionAgent(
                selected_record.name,
                storage_key=str(selected_record.agent_id),
                agent_id=str(selected_record.agent_id),
                legacy_storage_key=selected_record.legacy_storage_key,
                log_store=logs,
                directory=directory,
                context_policy=ExecutionContextPolicy(
                    max_recent_episodes=3,
                    max_characters=1_200,
                ),
            )
            execution_prompt = agent.build_system_prompt_with_history()
            execution_context = asdict(agent.last_context_metrics)

        final_text = (
            "Submitted to the execution agent."
            if results and results[-1].success
            else "I need clarification before dispatching."
        )
        final_response = InteractionAgentRuntime._finalize_response(
            InteractionAgentRuntime.__new__(InteractionAgentRuntime),
            _LoopSummary(last_assistant_text=final_text),
        )

    records = directory.list_records()
    journal_bytes = tuple(
        (path.name, path.read_bytes())
        for path in sorted(execution_dir.glob("*.log"))
    )
    tool_results = tuple(
        {
            "success": result.success,
            "payload": result.payload,
            "user_message": result.user_message,
            "recorded_reply": result.recorded_reply,
        }
        for result in results
    )
    return ScenarioOutcome(
        candidates=_candidate_facts(candidate_context),
        action=candidate_context.decision.action.value,
        recommendation=(
            str(candidate_context.decision.agent_id)
            if candidate_context.decision.agent_id is not None
            else None
        ),
        allowed_dispatch_ids=tuple(sorted(str(value) for value in allowed_ids)),
        selected_identity=selected,
        created_identity=created,
        directory_serialization=(execution_dir / "roster.json").read_bytes(),
        idempotency_outcome=tuple(
            result.payload.get("new_agent_created") if result.success else None
            for result in results
        ),
        routing_context={
            "query": query,
            "transcript": transcript,
            "candidate_count": len(candidate_context.candidates),
            "action": candidate_context.decision.action.value,
            "agent_id": (
                str(candidate_context.decision.agent_id)
                if candidate_context.decision.agent_id is not None
                else None
            ),
            "confidence": candidate_context.decision.confidence,
            "reasons": tuple(candidate_context.decision.reasons),
        },
        execution_prompt=execution_prompt,
        execution_context=execution_context,
        raw_journal_bytes=journal_bytes,
        tool_results=tool_results,
        batch_submissions=tuple(batch.submissions),
        final_response=final_response,
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "reuse",
        "create",
        "abstain",
        "repeated_create",
        "forbidden_dispatch",
        "legacy_migration",
        "long_history",
    ],
)
def test_trace_on_restores_the_same_snapshot_and_preserves_every_production_fact(
    tmp_path,
    monkeypatch,
    scenario: str,
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

    trace_off = _run_scenario(trace_off_root, scenario, sink=NullTraceSink())
    trace_store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    trace_on = _run_scenario(trace_on_root, scenario, sink=trace_store)

    assert trace_on == trace_off
    assert trace_store.path_for(
        UUID("10000000-0000-4000-8000-000000000001")
    ).is_file()


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

    control = _run_scenario(control_root, "repeated_create", sink=NullTraceSink())
    failed = _run_scenario(failure_root, "repeated_create", sink=FailingSink())

    assert failed == control
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "server.services.evaluation_lab.trace"
    ]
    assert messages
    assert set(messages) == {"Evaluation trace unavailable; observability degraded."}
    assert "private-sink-token" not in caplog.text
    assert "mailbox@example.invalid" not in caplog.text
