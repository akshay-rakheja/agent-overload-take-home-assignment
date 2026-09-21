"""Request-scoped trace emission and direct fact consolidation."""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from pydantic import JsonValue

from .models import (
    Availability,
    CostPlaceholder,
    ObservedValue,
    SystemRunResult,
    TraceContext,
    TraceEvent,
    TraceEventKind,
    UsagePlaceholder,
)
from .redaction import redact_value


logger = logging.getLogger(__name__)
_DEGRADED_WARNING = "Evaluation trace unavailable; observability degraded."


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class NullTraceSink:
    def emit(self, event: TraceEvent) -> None:
        return None


class _ScopeState:
    __slots__ = ("context", "sink")

    def __init__(self, context: TraceContext, sink: TraceSink) -> None:
        self.context = context
        self.sink = sink


_TRACE_SCOPE: ContextVar[_ScopeState | None] = ContextVar(
    "evaluation_lab_trace_scope", default=None
)
_TRACE_SEQUENCE: ContextVar[int] = ContextVar("evaluation_lab_trace_sequence", default=0)


@contextmanager
def trace_scope(context: TraceContext, sink: TraceSink) -> Iterator[None]:
    """Install an isolated sink for one run/turn and restore the prior scope."""

    scope_token = _TRACE_SCOPE.set(_ScopeState(context, sink))
    sequence_token = _TRACE_SEQUENCE.set(0)
    try:
        yield
    finally:
        _TRACE_SEQUENCE.reset(sequence_token)
        _TRACE_SCOPE.reset(scope_token)


def make_trace_event(
    context: TraceContext,
    sequence: int,
    kind: TraceEventKind,
    payload: Mapping[str, object],
    *,
    occurred_at: datetime | None = None,
) -> TraceEvent:
    """Create one immutable event after recursively redacting its payload."""

    safe_payload = redact_value(dict(payload))
    if not isinstance(safe_payload, dict):
        raise ValueError("trace payload must redact to a JSON object")
    return TraceEvent(
        run_id=context.run_id,
        turn_id=context.turn_id,
        sequence=sequence,
        occurred_at=occurred_at or datetime.now(UTC),
        system=context.system,
        kind=kind,
        payload=safe_payload,
    )


def emit_trace(kind: TraceEventKind, payload: Mapping[str, object]) -> None:
    """Emit into the active request scope without changing caller behavior."""

    state = _TRACE_SCOPE.get()
    if state is None:
        return
    sequence = _TRACE_SEQUENCE.get() + 1
    _TRACE_SEQUENCE.set(sequence)
    try:
        state.sink.emit(make_trace_event(state.context, sequence, kind, payload))
    except Exception:
        logger.warning(_DEGRADED_WARNING)


def _coerce_run_id(run_id: UUID | str) -> UUID:
    try:
        return run_id if isinstance(run_id, UUID) else UUID(run_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("run_id must be a valid UUID") from exc


class JsonlTraceStore:
    """Append-only, fsynced JSONL storage rooted at an explicit lab directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()

    def path_for(self, run_id: UUID | str) -> Path:
        safe_id = _coerce_run_id(run_id)
        return self.root / f"{safe_id}.jsonl"

    def _open_root(self, *, create: bool) -> int | None:
        if self.root.is_symlink():
            raise ValueError("trace root must not be a symlink")
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        elif not self.root.exists():
            return None
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            return os.open(self.root, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("trace root must not be a symlink") from exc
            raise

    @staticmethod
    def _open_run_file(root_fd: int, name: str, flags: int, mode: int = 0o600) -> int:
        try:
            return os.open(
                name,
                flags | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=root_fd,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise ValueError("trace run file must not be a symlink") from exc
            raise

    @staticmethod
    def _decode_events(
        raw: bytes, *, requested_run: UUID
    ) -> tuple[tuple[TraceEvent, ...], bool]:
        truncated = bool(raw) and not raw.endswith(b"\n")
        complete = raw.rsplit(b"\n", 1)[0] if truncated else raw
        lines = complete.splitlines()
        events: list[TraceEvent] = []
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise ValueError(f"invalid blank trace line {line_number}")
            try:
                event = TraceEvent.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid trace event at line {line_number}") from exc
            if event.run_id != requested_run:
                raise ValueError("trace event does not match requested run")
            events.append(event)
        if events:
            _validate_event_stream(events)
        return tuple(events), truncated

    def _read_from_root(
        self, root_fd: int, run_id: UUID
    ) -> tuple[tuple[TraceEvent, ...], bool]:
        name = f"{run_id}.jsonl"
        try:
            descriptor = self._open_run_file(root_fd, name, os.O_RDONLY)
        except FileNotFoundError:
            return (), False
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        return self._decode_events(b"".join(chunks), requested_run=run_id)

    def read(self, run_id: UUID | str) -> tuple[TraceEvent, ...]:
        safe_id = _coerce_run_id(run_id)
        with self._lock:
            root_fd = self._open_root(create=False)
            if root_fd is None:
                return ()
            try:
                events, _ = self._read_from_root(root_fd, safe_id)
                return events
            finally:
                os.close(root_fd)

    def emit(self, event: TraceEvent) -> None:
        safe_event_payload = redact_value(event.model_dump(mode="json"))
        safe_event = TraceEvent.model_validate(safe_event_payload)
        with self._lock:
            root_fd = self._open_root(create=True)
            assert root_fd is not None
            try:
                existing, truncated = self._read_from_root(root_fd, safe_event.run_id)
                if truncated:
                    raise ValueError("cannot append after a truncated final line")
                if existing:
                    first = existing[0]
                    if (safe_event.turn_id, safe_event.system) != (
                        first.turn_id,
                        first.system,
                    ):
                        raise ValueError("trace events must keep the same turn and system")
                    expected = existing[-1].sequence + 1
                else:
                    expected = 1
                if safe_event.sequence != expected:
                    raise ValueError(
                        f"trace sequence {safe_event.sequence} cannot overwrite or skip {expected}"
                    )

                serialized = (
                    json.dumps(
                        safe_event.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                descriptor = self._open_run_file(
                    root_fd,
                    f"{safe_event.run_id}.jsonl",
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                )
                try:
                    view = memoryview(serialized)
                    while view:
                        written = os.write(descriptor, view)
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                os.close(root_fd)

    def reset(self, run_id: UUID | str) -> bool:
        safe_id = _coerce_run_id(run_id)
        with self._lock:
            root_fd = self._open_root(create=False)
            if root_fd is None:
                return False
            name = f"{safe_id}.jsonl"
            try:
                try:
                    metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return False
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("trace run path must be a regular file, not a symlink")
                os.unlink(name, dir_fd=root_fd)
                os.fsync(root_fd)
                return True
            finally:
                os.close(root_fd)


def _available(value: Any) -> ObservedValue[Any]:
    return ObservedValue(availability=Availability.AVAILABLE, value=value)


def _validate_event_stream(events: Sequence[TraceEvent]) -> None:
    if not events:
        raise ValueError("trace must contain at least one event")
    first = events[0]
    if any(
        (event.run_id, event.turn_id, event.system)
        != (first.run_id, first.turn_id, first.system)
        for event in events
    ):
        raise ValueError("trace events must belong to the same run, turn, and system")
    if [event.sequence for event in events] != list(range(1, len(events) + 1)):
        raise ValueError("trace event sequences must be contiguous from one")


def _payload_value(payload: dict[str, JsonValue], key: str) -> JsonValue:
    return payload[key]


def consolidate_trace(events: Sequence[TraceEvent]) -> SystemRunResult:
    """Map only emitted event facts into the shared result schema."""

    _validate_event_stream(events)
    first = events[0]
    values: dict[str, Any] = {}
    gmail_evidence: list[JsonValue] = []
    timings: list[JsonValue] = []
    errors: list[JsonValue] = []
    usage: dict[str, ObservedValue[int]] = {}
    cost: dict[str, ObservedValue[Any]] = {}

    for event in events:
        payload = redact_value(event.payload)
        if not isinstance(payload, dict):
            raise ValueError("trace payload must redact to a JSON object")
        if event.kind is TraceEventKind.RUN_METADATA:
            for key in ("revision", "mode", "roster_count"):
                if key in payload:
                    values[key] = _available(_payload_value(payload, key))
        elif event.kind is TraceEventKind.ROSTER_SNAPSHOT and "roster_count" in payload:
            values["roster_count"] = _available(payload["roster_count"])
        elif event.kind is TraceEventKind.PROMPT_EXPOSURE:
            values["prompt_exposure"] = _available(payload)
        elif event.kind is TraceEventKind.CANDIDATES:
            candidates = payload.get("candidates")
            values["candidates"] = _available(
                candidates if isinstance(candidates, list) else [payload]
            )
        elif event.kind is TraceEventKind.ROUTING_DECISION:
            values["decision"] = _available(payload)
            if "recommendation" in payload:
                values["recommendation"] = _available(payload["recommendation"])
        elif event.kind is TraceEventKind.AUTHORIZATION and "authorized_ids" in payload:
            values["authorized_ids"] = _available(payload["authorized_ids"])
        elif event.kind is TraceEventKind.DISPATCH_ATTEMPT:
            values["attempted_dispatch"] = _available(payload)
        elif event.kind is TraceEventKind.DISPATCH_RESULT:
            values["accepted_dispatch"] = _available(payload)
        elif event.kind is TraceEventKind.IDENTITY:
            for source, target in (
                ("selected", "selected_identity"),
                ("created", "created_identity"),
                ("delta", "identity_delta"),
                ("duplicates", "duplicates"),
            ):
                if source in payload:
                    values[target] = _available(payload[source])
        elif event.kind is TraceEventKind.GMAIL_EVIDENCE:
            gmail_evidence.append(payload)
        elif event.kind is TraceEventKind.FINAL_RESPONSE and "response" in payload:
            values["final_response"] = _available(payload["response"])
        elif event.kind is TraceEventKind.CONTEXT_METRICS:
            values["context_metrics"] = _available(payload)
        elif event.kind is TraceEventKind.PHASE_TIMING:
            timings.append(payload)
        elif event.kind is TraceEventKind.USAGE:
            for key in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
                if key in payload:
                    usage[key] = _available(payload[key])
        elif event.kind is TraceEventKind.COST:
            for key in ("amount", "currency"):
                if key in payload:
                    cost[key] = _available(payload[key])
        elif event.kind in {TraceEventKind.ERROR, TraceEventKind.OBSERVABILITY_WARNING}:
            errors.append(payload)

    if gmail_evidence:
        values["gmail_evidence"] = _available(gmail_evidence)
    if timings:
        values["timings"] = _available(timings)
    if errors:
        values["errors"] = _available(errors)

    result = SystemRunResult(
        run_id=first.run_id,
        turn_id=first.turn_id,
        system=first.system,
        usage=UsagePlaceholder(**usage),
        cost=CostPlaceholder(**cost),
        **values,
    )
    availability = {
        field: observed.availability
        for field, observed in result.__dict__.items()
        if isinstance(observed, ObservedValue)
    }
    return result.model_copy(update={"availability_metadata": availability})


__all__ = [
    "JsonlTraceStore",
    "NullTraceSink",
    "TraceSink",
    "consolidate_trace",
    "emit_trace",
    "make_trace_event",
    "trace_scope",
]
