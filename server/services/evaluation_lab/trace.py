"""Request-scoped trace emission and direct fact consolidation."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import stat
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from pydantic import JsonValue, TypeAdapter

from .models import (
    Availability,
    CostPlaceholder,
    ObservedValue,
    SystemRunResult,
    TraceContext,
    TraceEvent,
    TraceEventKind,
    TraceRunStatus,
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
    __slots__ = ("active", "context", "lock", "sequence", "sink")

    def __init__(self, context: TraceContext, sink: TraceSink) -> None:
        self.active = True
        self.context = context
        self.lock = threading.RLock()
        self.sequence = 0
        self.sink = sink


class TraceTiming:
    """Monotonic phase timing with active trace-observation work removed."""

    __slots__ = (
        "_active",
        "_active_observations",
        "_clock",
        "_final_elapsed_ns",
        "_finished_monotonic_ns",
        "_lock",
        "_next_observation_id",
        "_trace_observation_intervals",
        "started_monotonic_ns",
    )

    def __init__(self, clock: Callable[[], int]) -> None:
        self._active = True
        self._active_observations: dict[int, int] = {}
        self._clock = clock
        self._final_elapsed_ns: int | None = None
        self._finished_monotonic_ns: int | None = None
        self._lock = threading.RLock()
        self._next_observation_id = 0
        self._trace_observation_intervals: list[tuple[int, int]] = []
        self.started_monotonic_ns = clock()

    def _observation_started(self) -> tuple[int, int] | None:
        with self._lock:
            if not self._active:
                return None
            try:
                value = self._clock()
            except Exception:
                return None
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            self._next_observation_id += 1
            observation_id = self._next_observation_id
            self._active_observations[observation_id] = value
            return observation_id, value

    def _observation_finished(self, observation: tuple[int, int]) -> None:
        observation_id, started = observation
        try:
            finished = self._clock()
        except Exception:
            finished = None
        with self._lock:
            self._active_observations.pop(observation_id, None)
            if (
                self._final_elapsed_ns is None
                and isinstance(finished, int)
                and not isinstance(finished, bool)
                and finished > started
            ):
                self._trace_observation_intervals.append((started, finished))

    def _elapsed_at_locked(self, finished: int) -> int:
        gross = max(0, finished - self.started_monotonic_ns)
        intervals = [
            *self._trace_observation_intervals,
            *((started, finished) for started in self._active_observations.values()),
        ]
        clipped = sorted(
            (
                max(self.started_monotonic_ns, started),
                min(finished, observation_finished),
            )
            for started, observation_finished in intervals
            if observation_finished > self.started_monotonic_ns
            and started < finished
        )

        trace_overhead = 0
        if clipped:
            union_start, union_end = clipped[0]
            for interval_start, interval_end in clipped[1:]:
                if interval_start <= union_end:
                    union_end = max(union_end, interval_end)
                    continue
                trace_overhead += max(0, union_end - union_start)
                union_start, union_end = interval_start, interval_end
            trace_overhead += max(0, union_end - union_start)
        return gross - min(gross, trace_overhead)

    def finish(self) -> int:
        """Freeze the phase endpoint before its terminal trace event is emitted."""

        with self._lock:
            if self._finished_monotonic_ns is None:
                self._finished_monotonic_ns = self._clock()
                self._active = False
                self._final_elapsed_ns = self._elapsed_at_locked(
                    self._finished_monotonic_ns
                )
            return self._finished_monotonic_ns

    @property
    def finished_monotonic_ns(self) -> int:
        return self.finish()

    @property
    def elapsed_ns(self) -> int:
        self.finish()
        with self._lock:
            assert self._final_elapsed_ns is not None
            return self._final_elapsed_ns


_TRACE_SCOPE: ContextVar[_ScopeState | None] = ContextVar(
    "evaluation_lab_trace_scope", default=None
)
_TRACE_TIMINGS: ContextVar[tuple[TraceTiming, ...]] = ContextVar(
    "evaluation_lab_trace_timings", default=()
)
_TRACE_EMIT_DEPTH: ContextVar[int] = ContextVar(
    "evaluation_lab_trace_emit_depth", default=0
)
_ACTIVE_TURNS: dict[tuple[UUID, UUID, str], int] = {}
_ACTIVE_TURNS_LOCK = threading.RLock()


def _turn_key(context: TraceContext) -> tuple[UUID, UUID, str]:
    return (context.run_id, context.turn_id, context.system)


def _register_active_turn(context: TraceContext) -> None:
    key = _turn_key(context)
    with _ACTIVE_TURNS_LOCK:
        _ACTIVE_TURNS[key] = _ACTIVE_TURNS.get(key, 0) + 1


def _unregister_active_turn(context: TraceContext) -> None:
    key = _turn_key(context)
    with _ACTIVE_TURNS_LOCK:
        remaining = _ACTIVE_TURNS.get(key, 0) - 1
        if remaining > 0:
            _ACTIVE_TURNS[key] = remaining
        else:
            _ACTIVE_TURNS.pop(key, None)


def _is_active_run(context: TraceContext) -> bool:
    with _ACTIVE_TURNS_LOCK:
        return any(
            run_id == context.run_id and count > 0
            for (run_id, _turn_id, _system), count in _ACTIVE_TURNS.items()
        )


@contextmanager
def trace_scope(context: TraceContext, sink: TraceSink) -> Iterator[None]:
    """Install an isolated sink for one run/turn and restore the prior scope."""

    state = _ScopeState(context, sink)
    scope_token = _TRACE_SCOPE.set(state)
    _register_active_turn(context)
    try:
        yield
    finally:
        with state.lock:
            state.active = False
        _unregister_active_turn(context)
        _TRACE_SCOPE.reset(scope_token)


@contextmanager
def trace_timing(clock: Callable[[], int]) -> Iterator[TraceTiming]:
    """Measure a phase while accounting for nested synchronous trace work."""

    timing = TraceTiming(clock)
    timing_token = _TRACE_TIMINGS.set((*_TRACE_TIMINGS.get(), timing))
    try:
        yield timing
    finally:
        timing.finish()
        _TRACE_TIMINGS.reset(timing_token)


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
    depth = _TRACE_EMIT_DEPTH.get()
    depth_token = _TRACE_EMIT_DEPTH.set(depth + 1)
    observations = (
        tuple(
            (timing, started)
            for timing in _TRACE_TIMINGS.get()
            if (started := timing._observation_started()) is not None
        )
        if depth == 0
        else ()
    )
    try:
        with state.lock:
            if not state.active:
                return
            state.sequence += 1
            state.sink.emit(
                make_trace_event(state.context, state.sequence, kind, payload)
            )
    except Exception:
        logger.warning(_DEGRADED_WARNING)
    finally:
        for timing, started in observations:
            timing._observation_finished(started)
        _TRACE_EMIT_DEPTH.reset(depth_token)


def trace_active() -> bool:
    """Return whether the current context has a live request-scoped trace sink."""

    state = _TRACE_SCOPE.get()
    if state is None:
        return False
    with state.lock:
        return state.active and not isinstance(state.sink, NullTraceSink)


def _coerce_run_id(run_id: UUID | str) -> UUID:
    try:
        return run_id if isinstance(run_id, UUID) else UUID(run_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("run_id must be a valid UUID") from exc


class JsonlTraceStore:
    """Append-only, fsynced JSONL storage rooted at an explicit lab directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        root_parts = self.root.parts
        if ".." in root_parts:
            raise ValueError("trace root must not contain path traversal")
        lab_indexes = [index for index, part in enumerate(root_parts) if part == ".lab"]
        if not lab_indexes:
            raise ValueError("trace root must be inside the VCS-ignored .lab directory")
        lab_index = lab_indexes[-1]
        if lab_index == len(root_parts) - 1:
            raise ValueError("trace root must be below the VCS-ignored .lab directory")
        parent_parts = root_parts[:lab_index]
        self._trusted_parent = Path(*parent_parts) if parent_parts else Path(".")
        self._root_components = root_parts[lab_index:]
        self._lock = threading.RLock()

    def path_for(self, run_id: UUID | str) -> Path:
        safe_id = _coerce_run_id(run_id)
        return self.root / f"{safe_id}.jsonl"

    def _open_root(self, *, create: bool) -> int | None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self._trusted_parent, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("trace root must not contain a symlink") from exc
            raise

        try:
            for component in self._root_components:
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        os.close(descriptor)
                        return None
                    raise
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError(
                            "trace root must not contain a symlink"
                        ) from exc
                    raise
                os.close(descriptor)
                descriptor = child
            return descriptor
        except Exception:
            os.close(descriptor)
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
            events = list(_validate_event_stream(events))
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

    @contextmanager
    def _store_lock(self, root_fd: int, *, exclusive: bool) -> Iterator[None]:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(root_fd, operation)
            yield
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)

    def read(self, run_id: UUID | str) -> tuple[TraceEvent, ...]:
        safe_id = _coerce_run_id(run_id)
        with self._lock:
            root_fd = self._open_root(create=False)
            if root_fd is None:
                return ()
            try:
                with self._store_lock(root_fd, exclusive=False):
                    events, _ = self._read_from_root(root_fd, safe_id)
                    return events
            finally:
                os.close(root_fd)

    def status(self, run_id: UUID | str) -> TraceRunStatus | None:
        """Return direct append-stream facts without creating storage."""

        safe_id = _coerce_run_id(run_id)
        with self._lock:
            root_fd = self._open_root(create=False)
            if root_fd is None:
                return None
            try:
                with self._store_lock(root_fd, exclusive=False):
                    try:
                        metadata = os.stat(
                            f"{safe_id}.jsonl",
                            dir_fd=root_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        return None
                    if not stat.S_ISREG(metadata.st_mode):
                        raise ValueError(
                            "trace run path must be a regular file, not a symlink"
                        )
                    events, truncated = self._read_from_root(root_fd, safe_id)
                    return TraceRunStatus(
                        run_id=safe_id,
                        event_count=len(events),
                        last_sequence=events[-1].sequence if events else None,
                        complete=not truncated,
                    )
            finally:
                os.close(root_fd)

    def emit(self, event: TraceEvent) -> None:
        safe_event_payload = redact_value(event.model_dump(mode="json"))
        safe_event = TraceEvent.model_validate_json(
            json.dumps(
                safe_event_payload,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        with self._lock:
            root_fd = self._open_root(create=True)
            assert root_fd is not None
            try:
                with self._store_lock(root_fd, exclusive=True):
                    existing, truncated = self._read_from_root(
                        root_fd, safe_event.run_id
                    )
                    if truncated:
                        raise ValueError("cannot append after a truncated final line")
                    if existing:
                        first = existing[0]
                        if (safe_event.turn_id, safe_event.system) != (
                            first.turn_id,
                            first.system,
                        ):
                            raise ValueError(
                                "trace events must keep the same turn and system"
                            )
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
                            allow_nan=False,
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

    def reset(self, context: TraceContext) -> bool:
        if not isinstance(context, TraceContext):
            raise ValueError("reset requires the owning TraceContext")
        safe_id = context.run_id
        if _is_active_run(context):
            raise RuntimeError("cannot reset an active trace turn")
        with self._lock:
            root_fd = self._open_root(create=False)
            if root_fd is None:
                return False
            name = f"{safe_id}.jsonl"
            try:
                with self._store_lock(root_fd, exclusive=True):
                    with _ACTIVE_TURNS_LOCK:
                        if _is_active_run(context):
                            raise RuntimeError("cannot reset an active trace turn")
                        events, truncated = self._read_from_root(root_fd, safe_id)
                        if not events and not truncated:
                            return False
                        if truncated:
                            raise ValueError(
                                "cannot reset a trace with a truncated final line"
                            )
                        first = events[0]
                        if (first.turn_id, first.system) != (
                            context.turn_id,
                            context.system,
                        ):
                            raise ValueError("TraceContext does not own this trace run")
                        try:
                            metadata = os.stat(
                                name, dir_fd=root_fd, follow_symlinks=False
                            )
                        except FileNotFoundError:
                            return False
                        if not stat.S_ISREG(metadata.st_mode):
                            raise ValueError(
                                "trace run path must be a regular file, not a symlink"
                            )
                        os.unlink(name, dir_fd=root_fd)
                        os.fsync(root_fd)
                        return True
            finally:
                os.close(root_fd)


def _available(value: Any) -> ObservedValue[Any]:
    return ObservedValue(availability=Availability.AVAILABLE, value=value)


def _validate_event_stream(events: Sequence[TraceEvent]) -> tuple[TraceEvent, ...]:
    if not events:
        raise ValueError("trace must contain at least one event")
    validated = tuple(
        TraceEvent.model_validate_json(
            json.dumps(
                event.model_dump(mode="json"),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for event in events
    )
    first = validated[0]
    if any(
        (event.run_id, event.turn_id, event.system)
        != (first.run_id, first.turn_id, first.system)
        for event in validated
    ):
        raise ValueError("trace events must belong to the same run, turn, and system")
    if [event.sequence for event in validated] != list(range(1, len(validated) + 1)):
        raise ValueError("trace event sequences must be contiguous from one")
    return validated


def _payload_value(payload: Mapping[str, JsonValue], key: str) -> JsonValue:
    return payload[key]


def _observed_value(value: object, value_type: type) -> ObservedValue[Any]:
    if isinstance(value, Mapping) and "availability" in value:
        return TypeAdapter(ObservedValue[value_type]).validate_json(
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        )
    return _available(value)


def _aggregate_observations(
    observations: Sequence[ObservedValue[Any]], *, fact: str
) -> ObservedValue[Any]:
    if not observations:
        return ObservedValue(
            availability=Availability.UNAVAILABLE,
            reason=f"{fact} was not emitted",
        )
    if not all(item.availability is Availability.AVAILABLE for item in observations):
        return ObservedValue(
            availability=Availability.UNAVAILABLE,
            reason=f"one or more model calls did not emit {fact}",
        )
    return _available(sum(item.value for item in observations if item.value is not None))


def _known_subtotal(
    observations: Sequence[ObservedValue[Any]], *, fact: str
) -> ObservedValue[Any]:
    available = [
        item.value
        for item in observations
        if item.availability is Availability.AVAILABLE and item.value is not None
    ]
    if observations and len(available) < len(observations) and available:
        return _available(sum(available))
    return ObservedValue(
        availability=Availability.UNAVAILABLE,
        reason=f"{fact} has no partial known subtotal",
    )


def _attempt_key(payload: Mapping[str, JsonValue]) -> tuple[str, int] | None:
    call_id = payload.get("call_id")
    attempt = payload.get("attempt")
    if (
        isinstance(call_id, str)
        and call_id
        and isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and attempt >= 0
    ):
        return call_id, attempt
    return None


def consolidate_trace(events: Sequence[TraceEvent]) -> SystemRunResult:
    """Map only emitted event facts into the shared result schema."""

    validated_events = _validate_event_stream(events)
    first = validated_events[0]
    values: dict[str, Any] = {}
    gmail_evidence: list[JsonValue] = []
    timings: list[JsonValue] = []
    errors: list[JsonValue] = []
    usage_by_attempt: dict[
        str, dict[tuple[str, int], ObservedValue[Any]]
    ] = {}
    legacy_usage: dict[str, list[ObservedValue[Any]]] = {}
    cost_by_attempt: dict[tuple[str, int], ObservedValue[Any]] = {}
    legacy_usage_cost: list[ObservedValue[Any]] = []
    legacy_cost_events: list[ObservedValue[Any]] = []
    expected_attempts: set[tuple[str, int]] = set()

    for event in validated_events:
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
        elif event.kind is TraceEventKind.MODEL_CALL:
            if payload.get("stage") in {
                "request",
                "attempt_start",
                "response",
                "error",
            }:
                if (key := _attempt_key(payload)) is not None:
                    expected_attempts.add(key)
        elif event.kind is TraceEventKind.USAGE:
            attempt_key = _attempt_key(payload)
            for key in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
                if key in payload:
                    observed = payload[key]
                    normalized = _observed_value(observed, int)
                    if attempt_key is None:
                        legacy_usage.setdefault(key, []).append(normalized)
                    else:
                        usage_by_attempt.setdefault(key, {})[attempt_key] = normalized
            for source, target in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
            ):
                observed = payload.get(source)
                if isinstance(observed, dict) and "availability" in observed:
                    normalized = _observed_value(observed, int)
                    if attempt_key is None:
                        legacy_usage.setdefault(target, []).append(normalized)
                    else:
                        usage_by_attempt.setdefault(target, {})[attempt_key] = normalized
            provider_cost = payload.get("provider_cost_usd")
            if isinstance(provider_cost, dict) and "availability" in provider_cost:
                normalized_cost = _observed_value(provider_cost, float)
                if attempt_key is None:
                    legacy_usage_cost.append(normalized_cost)
                else:
                    cost_by_attempt[attempt_key] = normalized_cost
        elif event.kind is TraceEventKind.COST:
            attempt_key = _attempt_key(payload)
            observed = payload.get("amount")
            provider_cost = payload.get("provider_cost_usd")
            normalized_cost: ObservedValue[Any] | None = None
            if observed is not None:
                normalized_cost = _observed_value(observed, float)
            elif isinstance(provider_cost, dict) and "availability" in provider_cost:
                normalized_cost = _observed_value(provider_cost, float)
            if normalized_cost is not None:
                if attempt_key is None:
                    legacy_cost_events.append(normalized_cost)
                else:
                    cost_by_attempt.setdefault(attempt_key, normalized_cost)
        elif event.kind in {TraceEventKind.ERROR, TraceEventKind.OBSERVABILITY_WARNING}:
            errors.append(payload)

    if gmail_evidence:
        values["gmail_evidence"] = _available(gmail_evidence)
    if timings:
        values["timings"] = _available(timings)
    if errors:
        values["errors"] = _available(errors)

    missing_attempt = ObservedValue(
        availability=Availability.UNAVAILABLE,
        reason="model attempt usage evidence was not emitted",
    )
    for attempt_key in expected_attempts:
        for fact in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
            usage_by_attempt.setdefault(fact, {}).setdefault(attempt_key, missing_attempt)
        cost_by_attempt.setdefault(attempt_key, missing_attempt)

    usage_values: dict[str, Any] = {}
    subtotal_names = {
        "input_tokens": "known_input_tokens_subtotal",
        "output_tokens": "known_output_tokens_subtotal",
        "cached_tokens": "known_cached_tokens_subtotal",
        "total_tokens": "known_total_tokens_subtotal",
    }
    for fact in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
        observations = [
            *usage_by_attempt.get(fact, {}).values(),
            *legacy_usage.get(fact, ()),
        ]
        usage_values[fact] = _aggregate_observations(
            observations, fact=fact
        ).model_dump()
        usage_values[subtotal_names[fact]] = _known_subtotal(
            observations, fact=fact
        ).model_dump()
    usage = UsagePlaceholder(**usage_values)
    legacy_cost = legacy_usage_cost or legacy_cost_events
    cost_observations = [*cost_by_attempt.values(), *legacy_cost]
    aggregate_cost = _aggregate_observations(
        cost_observations, fact="provider cost"
    )
    cost = CostPlaceholder(
        amount=aggregate_cost.model_dump(),
        known_amount_subtotal=_known_subtotal(
            cost_observations, fact="provider cost"
        ).model_dump(),
        currency=(
            _available("USD").model_dump()
            if aggregate_cost.availability is Availability.AVAILABLE
            else ObservedValue(
                availability=Availability.UNAVAILABLE,
                reason="provider cost currency was not emitted",
            ).model_dump()
        ),
    )
    result = SystemRunResult(
        run_id=first.run_id,
        turn_id=first.turn_id,
        system=first.system,
        usage=usage,
        cost=cost,
        **{
            key: value.model_dump() if isinstance(value, ObservedValue) else value
            for key, value in values.items()
        },
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
    "trace_active",
    "trace_scope",
]
