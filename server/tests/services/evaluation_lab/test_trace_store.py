from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from server.services.evaluation_lab.models import (
    TraceContext,
    TraceEvent,
    TraceEventKind,
)
from server.services.evaluation_lab.trace import (
    JsonlTraceStore,
    NullTraceSink,
    emit_trace,
    trace_scope,
)
from server.services.evaluation_lab import trace as trace_module


RUN_A = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
RUN_B = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
TURN_A = UUID("11111111-1111-4111-8111-111111111111")
TURN_B = UUID("22222222-2222-4222-8222-222222222222")


def _context(run_id: UUID = RUN_A, turn_id: UUID = TURN_A) -> TraceContext:
    return TraceContext(
        run_id=run_id,
        turn_id=turn_id,
        system="enhanced",
        revision="enhanced",
        mode="lab",
    )


def _event(
    run_id: UUID = RUN_A,
    turn_id: UUID = TURN_A,
    sequence: int = 1,
    *,
    payload: dict[str, object] | None = None,
) -> TraceEvent:
    return TraceEvent(
        schema_version=1,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        occurred_at=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
        system="enhanced",
        kind=TraceEventKind.RUN_METADATA,
        payload=payload or {"revision": "enhanced"},
    )


class _CollectingSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


def _process_append_same_event(root: str, start: object, results: object) -> None:
    start.wait()
    try:
        JsonlTraceStore(Path(root)).emit(_event())
    except ValueError:
        results.put("rejected")
    else:
        results.put("written")


def test_store_appends_redacted_events_and_reads_stable_order(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    first = _event(payload={"revision": "enhanced", "api_key": "sk-test-private"})
    second = _event(sequence=2, payload={"mode": "lab"})

    store.emit(first)
    store.emit(second)

    assert store.read(RUN_A) == (first.model_copy(update={"payload": {"api_key": "[REDACTED]", "revision": "enhanced"}}), second)
    raw = store.path_for(RUN_A).read_bytes()
    assert raw.count(b"\n") == 2
    assert b"sk-test-private" not in raw
    assert os.stat(store.path_for(RUN_A)).st_mode & 0o777 == 0o600


def test_store_keeps_concurrent_runs_in_separate_files(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    events = [
        _event(RUN_A, TURN_A, payload={"run": "a"}),
        _event(RUN_B, TURN_B, payload={"run": "b"}),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(store.emit, events))

    assert [event.payload for event in store.read(RUN_A)] == [{"run": "a"}]
    assert [event.payload for event in store.read(RUN_B)] == [{"run": "b"}]
    assert store.path_for(RUN_A) != store.path_for(RUN_B)


def test_distinct_store_instances_serialize_same_run_append_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".lab" / "traces"
    stores = (JsonlTraceStore(root), JsonlTraceStore(root))
    writers_ready = threading.Barrier(2)
    real_write = os.write

    def synchronized_write(descriptor: int, data: object) -> int:
        try:
            writers_ready.wait(timeout=0.1)
        except threading.BrokenBarrierError:
            pass
        return real_write(descriptor, data)

    monkeypatch.setattr(trace_module.os, "write", synchronized_write)

    def attempt(store: JsonlTraceStore) -> str:
        try:
            store.emit(_event())
        except ValueError:
            return "rejected"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, stores))

    assert sorted(outcomes) == ["rejected", "written"]
    assert stores[0].read(RUN_A) == (_event(),)


def test_distinct_processes_serialize_same_run_append_validation(tmp_path: Path) -> None:
    root = tmp_path / ".lab" / "traces"
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_append_same_event,
            args=(str(root), start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0

    assert sorted(results.get(timeout=1) for _ in processes) == ["rejected", "written"]
    assert JsonlTraceStore(root).read(RUN_A) == (_event(),)


def test_store_ignores_a_truncated_final_line_but_refuses_further_append(
    tmp_path: Path,
) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    store.emit(_event())
    with store.path_for(RUN_A).open("ab") as handle:
        handle.write(b'{"schema_version":1,"run_id":"truncated')

    assert store.read(RUN_A) == (_event(),)
    with pytest.raises(ValueError, match="truncated final line"):
        store.emit(_event(sequence=2))


@pytest.mark.parametrize(
    "bad_event",
    [
        _event(sequence=1),
        _event(sequence=3),
        _event(turn_id=TURN_B, sequence=2),
        _event(sequence=2).model_copy(update={"system": "baseline"}),
    ],
)
def test_store_refuses_overwrite_gaps_and_run_or_turn_mismatch(
    tmp_path: Path, bad_event: TraceEvent
) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    store.emit(_event())
    original = store.path_for(RUN_A).read_bytes()

    with pytest.raises(ValueError):
        store.emit(bad_event)

    assert store.path_for(RUN_A).read_bytes() == original


def test_store_read_rejects_an_event_placed_under_the_wrong_run(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    store.root.mkdir(parents=True)
    store.path_for(RUN_A).write_text(_event(RUN_B, TURN_B).model_dump_json() + "\n")

    with pytest.raises(ValueError, match="requested run"):
        store.read(RUN_A)


def test_reset_removes_only_the_requested_run(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    store.emit(_event(RUN_A, TURN_A))
    store.emit(_event(RUN_B, TURN_B))

    assert store.reset(_context(RUN_A, TURN_A)) is True

    assert store.read(RUN_A) == ()
    assert store.read(RUN_B) == (_event(RUN_B, TURN_B),)
    assert store.reset(_context(RUN_A, TURN_A)) is False


def test_reset_requires_matching_owning_context(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    owner = _context(RUN_A, TURN_A)
    store.emit(_event(RUN_A, TURN_A))

    with pytest.raises(ValueError, match="owning TraceContext"):
        store.reset(RUN_A)
    with pytest.raises(ValueError, match="does not own"):
        store.reset(_context(RUN_A, TURN_B))

    assert store.read(RUN_A) == (_event(RUN_A, TURN_A),)
    assert store.reset(owner) is True


def test_reset_refuses_an_active_turn(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    owner = _context(RUN_A, TURN_A)
    store.emit(_event(RUN_A, TURN_A))

    with trace_scope(owner, NullTraceSink()):
        with pytest.raises(RuntimeError, match="active trace turn"):
            store.reset(owner)

    assert store.read(RUN_A) == (_event(RUN_A, TURN_A),)


def test_reset_refuses_any_active_turn_for_the_same_run(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    owner = _context(RUN_A, TURN_A)
    store.emit(_event(RUN_A, TURN_A))

    with trace_scope(_context(RUN_A, TURN_B), NullTraceSink()):
        with pytest.raises(RuntimeError, match="active trace turn"):
            store.reset(owner)

    assert store.read(RUN_A) == (_event(RUN_A, TURN_A),)


@pytest.mark.parametrize("run_id", ["../escape", "not-a-uuid", "a/b"])
def test_store_refuses_run_id_path_traversal(tmp_path: Path, run_id: str) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")

    with pytest.raises(ValueError, match="valid UUID"):
        store.read(run_id)


def test_store_refuses_symlink_root_and_run_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    lab_root = tmp_path / ".lab"
    lab_root.mkdir()
    root_link = lab_root / "trace-link"
    root_link.symlink_to(outside, target_is_directory=True)
    linked_store = JsonlTraceStore(root_link)

    with pytest.raises(ValueError, match="symlink"):
        linked_store.emit(_event())

    root = tmp_path / ".lab" / "traces"
    root.mkdir(parents=True)
    outside_file = outside / "events.jsonl"
    outside_file.write_text("sentinel")
    (root / f"{RUN_A}.jsonl").symlink_to(outside_file)
    store = JsonlTraceStore(root)

    with pytest.raises(ValueError, match="symlink"):
        store.emit(_event())
    assert outside_file.read_text() == "sentinel"


def test_trace_scope_sequences_and_isolates_concurrent_contexts() -> None:
    async def collect(context: TraceContext) -> list[TraceEvent]:
        sink = _CollectingSink()
        with trace_scope(context, sink):
            emit_trace(TraceEventKind.RUN_METADATA, {"revision": context.revision})
            await asyncio.sleep(0)
            emit_trace(TraceEventKind.FINAL_RESPONSE, {"response": "fixture complete"})
        return sink.events

    async def run_both() -> tuple[list[TraceEvent], list[TraceEvent]]:
        return await asyncio.gather(
            collect(_context(RUN_A, TURN_A)), collect(_context(RUN_B, TURN_B))
        )

    first, second = asyncio.run(run_both())

    assert [event.sequence for event in first] == [1, 2]
    assert [event.sequence for event in second] == [1, 2]
    assert {event.run_id for event in first} == {RUN_A}
    assert {event.run_id for event in second} == {RUN_B}


def test_trace_scope_allocates_unique_sequences_to_same_run_async_fanout() -> None:
    async def collect() -> list[TraceEvent]:
        sink = _CollectingSink()

        async def child(label: str) -> None:
            await asyncio.sleep(0)
            emit_trace(TraceEventKind.TOOL_CALL, {"operation_name": label})

        with trace_scope(_context(), sink):
            await asyncio.gather(child("first"), child("second"))
        return sink.events

    events = asyncio.run(collect())

    assert [event.sequence for event in events] == [1, 2]
    assert {event.payload["operation_name"] for event in events} == {"first", "second"}


def test_same_run_async_fanout_persists_every_event_without_sink_degradation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")

    async def emit_both() -> None:
        async def child(label: str) -> None:
            await asyncio.sleep(0)
            emit_trace(TraceEventKind.TOOL_CALL, {"operation_name": label})

        with trace_scope(_context(), store):
            await asyncio.gather(child("first"), child("second"))

    caplog.set_level(logging.WARNING)
    asyncio.run(emit_both())

    events = store.read(RUN_A)
    assert [event.sequence for event in events] == [1, 2]
    assert {event.payload["operation_name"] for event in events} == {"first", "second"}
    assert "observability degraded" not in caplog.text


def test_store_requires_explicit_vcs_ignored_lab_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.lab"):
        JsonlTraceStore(tmp_path / "unignored-traces")
    with pytest.raises(ValueError, match="traversal"):
        JsonlTraceStore(tmp_path / ".lab" / ".." / "escaped-traces")


def test_store_refuses_symlinked_directory_below_lab_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    lab_root = tmp_path / ".lab"
    lab_root.mkdir()
    (lab_root / "linked").symlink_to(outside, target_is_directory=True)
    store = JsonlTraceStore(lab_root / "linked" / "traces")

    with pytest.raises(ValueError, match="symlink"):
        store.emit(_event())
    assert not (outside / "traces").exists()


def test_store_revalidates_copied_event_schema_version(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / ".lab" / "traces")
    bypassed = _event().model_copy(update={"schema_version": 2})

    with pytest.raises(ValidationError, match="schema_version"):
        store.emit(bypassed)
    assert store.read(RUN_A) == ()


def test_emit_outside_scope_and_null_sink_are_no_ops() -> None:
    emit_trace(TraceEventKind.ERROR, {"provider_error": "private"})
    with trace_scope(_context(), NullTraceSink()):
        emit_trace(TraceEventKind.ERROR, {"provider_error": "private"})


def test_sink_failure_never_changes_caller_behavior_or_logs_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingSink:
        def emit(self, event: TraceEvent) -> None:
            raise RuntimeError("write failed for alice@example.test with secret-token")

    caplog.set_level(logging.WARNING)
    result = "caller-result"

    with trace_scope(_context(), FailingSink()):
        emit_trace(
            TraceEventKind.ERROR,
            {"provider_error": "Bearer secret-token for alice@example.test"},
        )

    assert result == "caller-result"
    assert "Evaluation trace unavailable; observability degraded." in caplog.text
    assert "alice@example.test" not in caplog.text
    assert "secret-token" not in caplog.text
