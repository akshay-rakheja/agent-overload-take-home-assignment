"""Serialized, restart-safe orchestration of isolated paired Evaluation Lab runs."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evals.live_lab.grading import ScenarioSequenceScorecard, grade_scenario_sequence
from evals.live_lab.scenarios import ScenarioDefinition, load_controlled_scenarios

from .budget import BudgetExceeded
from .models import SystemRunResult
from .repetitions import (
    MeasuredSystem,
    OutcomeStatus,
    ScheduledPair,
    build_repetition_schedule,
)
from .run_store import RunStore

if TYPE_CHECKING:
    from evals.live_lab.contracts import StateFingerprint


_RUN_NAMESPACE = UUID("7b672a83-03f5-5c2c-84db-791fb80c71d7")


class RunStatus(str, Enum):
    QUEUED = "queued"
    RESETTING = "resetting"
    BASELINE_RUNNING = "baseline_running"
    ENHANCED_RUNNING = "enhanced_running"
    GRADING = "grading"
    COMPLETE = "complete"
    PARTIAL_FAILURE = "partial_failure"
    BLOCKED = "blocked"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETE, self.PARTIAL_FAILURE, self.BLOCKED}


class RunConflict(RuntimeError):
    """Raised when a different state-changing run is already active."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionMode(str, Enum):
    MEASURED = "measured"
    OFFLINE_FAKE = "offline_fake"


class SnapshotContract(_FrozenModel):
    profile_id: str
    snapshot_id: str
    baseline_roster_fingerprint: str
    enhanced_roster_fingerprint: str
    fixture_fingerprint: str
    raw_journal_fingerprint: str

    @field_validator(
        "profile_id",
        "snapshot_id",
        "baseline_roster_fingerprint",
        "enhanced_roster_fingerprint",
        "fixture_fingerprint",
        "raw_journal_fingerprint",
    )
    @classmethod
    def _bounded_value(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 512:
            raise ValueError("snapshot contract values must be bounded non-empty text")
        return normalized

    def roster_fingerprint(self, system: MeasuredSystem) -> str:
        return (
            self.baseline_roster_fingerprint
            if system is MeasuredSystem.BASELINE
            else self.enhanced_roster_fingerprint
        )


class BudgetAuthorization(_FrozenModel):
    reservation_id: UUID
    authorized: Literal[True] = True


class StartRunRequest(_FrozenModel):
    request_id: UUID = Field(default_factory=uuid4)
    scenario_ids: tuple[str, ...] = ()

    @field_validator("scenario_ids")
    @classmethod
    def _valid_scenarios(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or len(item) > 200 for item in normalized):
            raise ValueError("scenario ids must be bounded non-empty text")
        if len(set(normalized)) != len(normalized):
            raise ValueError("scenario ids must be unique")
        return normalized


class RunHandle(_FrozenModel):
    run_id: UUID
    request_id: UUID
    status: RunStatus


class StateVerification(_FrozenModel):
    system: MeasuredSystem
    snapshot_id: str
    roster_fingerprint: str
    expected_roster_fingerprint: str
    fixture_fingerprint: str
    expected_fixture_fingerprint: str
    raw_journal_fingerprint: str
    expected_raw_journal_fingerprint: str

    @classmethod
    def from_fingerprints(
        cls,
        *,
        system: MeasuredSystem,
        snapshot_id: str,
        actual: "StateFingerprint",
        expected: "StateFingerprint",
    ) -> "StateVerification":
        """Map the Task 03 byte/state contract into the three run gates."""

        return cls(
            system=system,
            snapshot_id=snapshot_id,
            roster_fingerprint=actual.roster_sha256,
            expected_roster_fingerprint=expected.roster_sha256,
            fixture_fingerprint=actual.logical_identity_digest,
            expected_fixture_fingerprint=expected.logical_identity_digest,
            raw_journal_fingerprint=actual.raw_journal_digest,
            expected_raw_journal_fingerprint=expected.raw_journal_digest,
        )

    @field_validator("snapshot_id", "roster_fingerprint", "expected_roster_fingerprint", "fixture_fingerprint", "expected_fixture_fingerprint", "raw_journal_fingerprint", "expected_raw_journal_fingerprint")
    @classmethod
    def _bounded(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 512:
            raise ValueError("state verification values must be bounded non-empty text")
        return normalized

    @property
    def mismatches(self) -> tuple[str, ...]:
        pairs = (
            ("roster_fingerprint", self.roster_fingerprint, self.expected_roster_fingerprint),
            ("fixture_fingerprint", self.fixture_fingerprint, self.expected_fixture_fingerprint),
            ("raw_journal_fingerprint", self.raw_journal_fingerprint, self.expected_raw_journal_fingerprint),
        )
        return tuple(name for name, observed, expected in pairs if observed != expected)


class SideRunOutput(_FrozenModel):
    status: OutcomeStatus
    model_id: str | None = None
    results: tuple[SystemRunResult, ...] = ()
    reason: str | None = None

    @field_validator("model_id")
    @classmethod
    def _observed_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("observed model identity must be bounded non-empty text")
        return normalized

    @model_validator(mode="after")
    def _coherent(self) -> "SideRunOutput":
        if self.status is OutcomeStatus.SUCCESS:
            if self.model_id is None or not self.results or self.reason is not None:
                raise ValueError("successful side output requires model/results and no reason")
        elif self.reason is None:
            raise ValueError("non-success side output requires a reason")
        return self


class LateCompletion(_FrozenModel):
    status: OutcomeStatus
    model_id: str | None = None
    results: tuple[SystemRunResult, ...] = ()
    reason: str | None = None


class PersistedSideOutcome(_FrozenModel):
    system: MeasuredSystem
    status: OutcomeStatus
    model_id: str | None = None
    results: tuple[SystemRunResult, ...] = ()
    attempt_count: Literal[1] = 1
    reason: str | None = None
    late_completion: LateCompletion | None = None


class PairExecutionRecord(_FrozenModel):
    scheduled: ScheduledPair
    verifications: tuple[StateVerification, ...] = ()
    outcomes: tuple[PersistedSideOutcome, ...] = ()

    @model_validator(mode="after")
    def _unique_systems(self) -> "PairExecutionRecord":
        for values in (self.verifications, self.outcomes):
            systems = [item.system for item in values]
            if len(systems) != len(set(systems)):
                raise ValueError("paired record cannot repeat a system")
        return self


class PairedSequenceScorecard(ScenarioSequenceScorecard):
    pair_id: UUID | None = None
    repetition: int | None = Field(default=None, ge=1)


class RunTransition(_FrozenModel):
    sequence: int = Field(ge=1)
    status: RunStatus
    occurred_at: datetime
    detail: str | None = None


class OrchestrationTraceEvent(_FrozenModel):
    sequence: int = Field(ge=1)
    kind: Literal["transition", "reset", "outcome", "grade", "recovery", "late"]
    occurred_at: datetime
    pair_id: UUID | None = None
    system: MeasuredSystem | None = None
    detail: str


class PairedRunResult(_FrozenModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    request: StartRunRequest
    execution_mode: ExecutionMode = ExecutionMode.MEASURED
    snapshot_contracts: tuple[SnapshotContract, ...] = ()
    status: RunStatus
    generation: int = Field(ge=0)
    schedule: tuple[ScheduledPair, ...]
    pairs: tuple[PairExecutionRecord, ...] = ()
    scorecards: tuple[PairedSequenceScorecard, ...] = ()
    transitions: tuple[RunTransition, ...]
    trace: tuple[OrchestrationTraceEvent, ...]
    blocked_reason: str | None = None
    created_at: datetime
    updated_at: datetime


Resetter = Callable[..., StateVerification | Awaitable[StateVerification]]
Runner = Callable[..., SideRunOutput | SystemRunResult | Sequence[SystemRunResult] | Awaitable[Any]]
BudgetGuard = Callable[
    ...,
    BudgetAuthorization | Awaitable[BudgetAuthorization],
]


class _SideAttemptCancelled(asyncio.CancelledError):
    def __init__(self, outcome: PersistedSideOutcome) -> None:
        self.outcome = outcome
        super().__init__("paired run execution was cancelled")


@dataclass(frozen=True)
class _SideWork:
    task: asyncio.Task[Any]
    thread: threading.Thread | None = None
    thread_result: Future[Any] | None = None

    @property
    def done(self) -> bool:
        return self.task.done() if self.thread is None else not self.thread.is_alive()

    def cancel(self) -> None:
        if self.thread is None:
            self.task.cancel()

    def result(self) -> object:
        if self.thread is None:
            return self.task.result()
        if self.thread.is_alive() or self.thread_result is None:
            raise asyncio.InvalidStateError("synchronous side work is still running")
        return self.thread_result.result()


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class PairedRunOrchestrator:
    """Execute exactly one paired run at a time, with no implicit side retry."""

    def __init__(
        self,
        *,
        store: RunStore,
        scenarios: Sequence[ScenarioDefinition] | None = None,
        resetter: Resetter,
        runners: Mapping[MeasuredSystem, Runner],
        budget_guard: BudgetGuard | None = None,
        execution_mode: ExecutionMode = ExecutionMode.MEASURED,
        snapshot_contracts: Mapping[str, SnapshotContract] | None = None,
        side_timeout_seconds: float = 60.0,
        late_completion_grace_seconds: float = 0.1,
        clock: Callable[[], datetime] | None = None,
        recover_on_startup: bool = True,
    ) -> None:
        if side_timeout_seconds <= 0 or late_completion_grace_seconds < 0:
            raise ValueError("side timeouts must be positive and late grace non-negative")
        self.store = store
        loaded = tuple(scenarios or load_controlled_scenarios())
        self._scenarios = MappingProxyType({item.scenario_id: item for item in loaded})
        if len(self._scenarios) != len(loaded):
            raise ValueError("orchestrator scenarios must have unique ids")
        self._resetter = resetter
        self._runners = MappingProxyType(dict(runners))
        self._budget_guard = budget_guard
        self._execution_mode = ExecutionMode(execution_mode)
        self._snapshot_contracts = MappingProxyType(dict(snapshot_contracts or {}))
        self._side_timeout = side_timeout_seconds
        self._late_grace = late_completion_grace_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._state_lock = asyncio.Lock()
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._late_monitors: set[asyncio.Task[Any]] = set()
        self._deferred_leases: set[tuple[UUID, UUID]] = set()
        if recover_on_startup:
            self._recover_interrupted()

    @property
    def active_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(
            task
            for task in (*self._active_tasks, *self._late_monitors)
            if not task.done()
        )

    @staticmethod
    def new_record(
        request: StartRunRequest,
        *,
        scenarios: Sequence[ScenarioDefinition],
        execution_mode: ExecutionMode = ExecutionMode.MEASURED,
        snapshot_contracts: Sequence[SnapshotContract] = (),
        now: datetime | None = None,
    ) -> PairedRunResult:
        available = {scenario.scenario_id: scenario for scenario in scenarios}
        selected_ids = request.scenario_ids or tuple(available)
        unknown = set(selected_ids) - set(available)
        if unknown:
            raise ValueError(f"unknown predeclared scenarios: {sorted(unknown)}")
        schedule = tuple(
            pair
            for scenario_id in selected_ids
            for pair in build_repetition_schedule(
                (scenario_id,), available[scenario_id].repetitions
            )
        )
        run_id = uuid5(_RUN_NAMESPACE, f"paired-run:v1:{request.request_id}")
        occurred_at = now or datetime.now(UTC)
        transition = RunTransition(sequence=1, status=RunStatus.QUEUED, occurred_at=occurred_at)
        event = OrchestrationTraceEvent(
            sequence=1,
            kind="transition",
            occurred_at=occurred_at,
            detail=RunStatus.QUEUED.value,
        )
        return PairedRunResult(
            run_id=run_id,
            request=request,
            execution_mode=execution_mode,
            snapshot_contracts=tuple(snapshot_contracts),
            status=RunStatus.QUEUED,
            generation=0,
            schedule=schedule,
            transitions=(transition,),
            trace=(event,),
            created_at=occurred_at,
            updated_at=occurred_at,
        )

    def _recover_interrupted(self) -> None:
        lease = self.store.execution_lease()
        if lease is not None and not self.store.execution_owner_is_alive(lease.run_id):
            self.store.retain_orphaned_execution(lease.run_id)
        for record in self.store.list_records():
            if record.status in {RunStatus.QUEUED, RunStatus.COMPLETE, RunStatus.PARTIAL_FAILURE, RunStatus.BLOCKED}:
                continue
            if self.store.execution_owner_is_alive(record.run_id):
                continue
            if not self.store.execution_block_exists(record.run_id):
                self.store.retain_orphaned_execution(record.run_id)
            now = self._clock()
            terminal = RunStatus.PARTIAL_FAILURE if any(pair.outcomes for pair in record.pairs) else RunStatus.BLOCKED
            reason = "run interrupted by process restart; no side was retried"
            if record.execution_mode is not self._execution_mode:
                reason += "; persisted execution mode is incompatible with controller mode"
            recovered = record.model_copy(
                update={
                    "status": terminal,
                    "generation": record.generation + 1,
                    "blocked_reason": reason if terminal is RunStatus.BLOCKED else None,
                    "transitions": record.transitions
                    + (
                        RunTransition(
                            sequence=len(record.transitions) + 1,
                            status=terminal,
                            occurred_at=now,
                            detail=reason,
                        ),
                    ),
                    "trace": record.trace
                    + (
                        OrchestrationTraceEvent(
                            sequence=len(record.trace) + 1,
                            kind="recovery",
                            occurred_at=now,
                            detail=reason,
                        ),
                    ),
                    "updated_at": now,
                }
            )
            self.store.save(recovered)

    def _handle(self, record: PairedRunResult) -> RunHandle:
        return RunHandle(
            run_id=record.run_id,
            request_id=record.request.request_id,
            status=record.status,
        )

    def _require_compatible_mode(self, record: PairedRunResult) -> None:
        if record.execution_mode is not self._execution_mode:
            raise RunConflict(
                "persisted run execution mode is incompatible with controller mode"
            )

    async def start(self, request: StartRunRequest) -> RunHandle:
        candidate = self.new_record(
            request,
            scenarios=tuple(self._scenarios.values()),
            execution_mode=self._execution_mode,
            snapshot_contracts=tuple(self._snapshot_contracts.values()),
            now=self._clock(),
        )
        existing = self.store.get(candidate.run_id)
        if existing is not None:
            if existing.request != request:
                raise RunConflict("idempotency key belongs to a different request")
            self._require_compatible_mode(existing)
            return self._handle(existing)
        if (
            self._state_lock.locked()
            or self.active_tasks
            or self.store.execution_block_exists()
        ):
            raise RunConflict("a state-changing Evaluation Lab run is active")
        await self._state_lock.acquire()
        try:
            active = next(
                (record for record in self.store.list_records() if not record.status.terminal),
                None,
            )
            if active is not None:
                raise RunConflict("a state-changing Evaluation Lab run is active")
            self.store.create(candidate)
            return self._handle(candidate)
        finally:
            self._state_lock.release()

    def status(self, run_id: UUID) -> PairedRunResult:
        record = self.store.get(run_id)
        if record is None:
            raise KeyError(str(run_id))
        return record

    def _persist(self, record: PairedRunResult, **updates: Any) -> PairedRunResult:
        now = self._clock()
        changed = record.model_copy(
            update={
                **updates,
                "generation": record.generation + 1,
                "updated_at": now,
            }
        )
        self.store.save(changed)
        return changed

    def _transition(
        self,
        record: PairedRunResult,
        status: RunStatus,
        *,
        detail: str | None = None,
    ) -> PairedRunResult:
        now = self._clock()
        transition = RunTransition(
            sequence=len(record.transitions) + 1,
            status=status,
            occurred_at=now,
            detail=detail,
        )
        event = OrchestrationTraceEvent(
            sequence=len(record.trace) + 1,
            kind="transition",
            occurred_at=now,
            detail=detail or status.value,
        )
        return self._persist(
            record,
            status=status,
            transitions=record.transitions + (transition,),
            trace=record.trace + (event,),
        )

    @staticmethod
    def _replace_pair(
        record: PairedRunResult, pair: PairExecutionRecord
    ) -> tuple[PairExecutionRecord, ...]:
        values = list(record.pairs)
        for index, existing in enumerate(values):
            if existing.scheduled.pair_id == pair.scheduled.pair_id:
                values[index] = pair
                break
        else:
            values.append(pair)
        order = {item.pair_id: index for index, item in enumerate(record.schedule)}
        values.sort(key=lambda item: order[item.scheduled.pair_id])
        return tuple(values)

    def _persist_pair_event(
        self,
        record: PairedRunResult,
        pair: PairExecutionRecord,
        *,
        kind: Literal["reset", "outcome"],
        system: MeasuredSystem,
        detail: str,
    ) -> PairedRunResult:
        event = OrchestrationTraceEvent(
            sequence=len(record.trace) + 1,
            kind=kind,
            occurred_at=self._clock(),
            pair_id=pair.scheduled.pair_id,
            system=system,
            detail=detail,
        )
        return self._persist(
            record,
            pairs=self._replace_pair(record, pair),
            trace=record.trace + (event,),
        )

    @staticmethod
    def _normalize_output(value: object) -> SideRunOutput:
        if isinstance(value, SideRunOutput):
            return value
        if isinstance(value, SystemRunResult) or (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and all(isinstance(item, SystemRunResult) for item in value)
        ):
            raise ValueError("bare side evidence has no observed model provenance")
        raise ValueError("side runner returned malformed output")

    @staticmethod
    def _outcome_from_output(
        output: SideRunOutput,
        *,
        record: PairedRunResult,
        scenario: ScenarioDefinition,
        system: MeasuredSystem,
    ) -> PersistedSideOutcome:
        invalid_sequence = (
            output.status is OutcomeStatus.SUCCESS
            and (
                len(output.results) != len(scenario.turns)
                or len({result.turn_id for result in output.results})
                != len(output.results)
            )
        )
        invalid_identity = any(
            result.run_id != record.run_id or result.system != system.value
            for result in output.results
        )
        if invalid_sequence or invalid_identity:
            return PersistedSideOutcome(
                system=system,
                status=OutcomeStatus.MALFORMED,
                model_id=output.model_id,
                results=output.results,
                reason=(
                    "side evidence does not match the run, system, or complete "
                    "distinct predeclared turn sequence"
                ),
            )
        return PersistedSideOutcome(
            system=system,
            status=output.status,
            model_id=output.model_id,
            results=output.results,
            reason=output.reason,
        )

    @classmethod
    def _completed_outcome(
        cls,
        work: _SideWork,
        *,
        record: PairedRunResult,
        scenario: ScenarioDefinition,
        system: MeasuredSystem,
    ) -> PersistedSideOutcome:
        try:
            output = cls._normalize_output(work.result())
        except asyncio.CancelledError:
            return PersistedSideOutcome(
                system=system,
                status=OutcomeStatus.FAILURE,
                reason="side task was cancelled",
            )
        except ValueError as exc:
            return PersistedSideOutcome(
                system=system,
                status=OutcomeStatus.MALFORMED,
                reason=str(exc),
            )
        except Exception as exc:
            return PersistedSideOutcome(
                system=system,
                status=OutcomeStatus.FAILURE,
                reason=f"{type(exc).__name__}: side execution failed",
            )
        return cls._outcome_from_output(
            output,
            record=record,
            scenario=scenario,
            system=system,
        )

    @staticmethod
    def _start_side_work(
        runner: Runner,
        kwargs: dict[str, object],
        *,
        name: str,
    ) -> _SideWork:
        if inspect.iscoroutinefunction(runner):
            return _SideWork(task=asyncio.create_task(runner(**kwargs), name=name))

        result: Future[Any] = Future()

        def _invoke() -> None:
            try:
                value = runner(**kwargs)
                if inspect.isawaitable(value):
                    value = asyncio.run(value)
                result.set_result(value)
            except BaseException as exc:
                result.set_exception(exc)

        thread = threading.Thread(target=_invoke, name=f"{name}-thread")
        thread.start()

        async def _observe_thread() -> object:
            while thread.is_alive():
                await asyncio.sleep(0.005)
            return result.result()

        return _SideWork(
            task=asyncio.create_task(_observe_thread(), name=name),
            thread=thread,
            thread_result=result,
        )

    def _watch_task(self, task: asyncio.Task[Any]) -> None:
        self._active_tasks.add(task)

        def _done(completed: asyncio.Task[Any]) -> None:
            self._active_tasks.discard(completed)
            if not completed.cancelled():
                try:
                    completed.exception()
                except BaseException:
                    pass

        task.add_done_callback(_done)

    @staticmethod
    async def _finalize_completed_work(
        work: _SideWork,
        *,
        cancellation_resistant: bool = False,
    ) -> None:
        """Retire observer bookkeeping once authoritative work is terminal."""

        if work.thread is None or not work.done or work.task.done():
            return
        work.task.cancel()
        try:
            await work.task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if (
                not cancellation_resistant
                and current is not None
                and current.cancelling()
            ):
                raise
        except Exception:
            pass

    @staticmethod
    async def _join_supervised_work(work: _SideWork) -> None:
        """Retain ownership until work ends, even if the observer is cancelled."""

        while not work.done:
            try:
                if work.task.done():
                    await asyncio.sleep(0.005)
                else:
                    await asyncio.shield(work.task)
            except asyncio.CancelledError:
                continue
            except Exception:
                pass
        await PairedRunOrchestrator._finalize_completed_work(
            work,
            cancellation_resistant=True,
        )

    @staticmethod
    def _late_from_outcome(outcome: PersistedSideOutcome) -> LateCompletion:
        return LateCompletion(
            status=outcome.status,
            model_id=outcome.model_id,
            results=outcome.results,
            reason=outcome.reason,
        )

    async def _schedule_late_monitor(
        self,
        work: _SideWork,
        *,
        record: PairedRunResult,
        scenario: ScenarioDefinition,
        scheduled: ScheduledPair,
        system: MeasuredSystem,
        owner_id: UUID,
    ) -> None:
        lease_key = (record.run_id, owner_id)
        self._deferred_leases.add(lease_key)
        started = asyncio.Event()

        async def _observe() -> None:
            started.set()
            evidence_persisted = False
            try:
                await self._join_supervised_work(work)
                outcome = self._completed_outcome(
                    work,
                    record=record,
                    scenario=scenario,
                    system=system,
                )
                late = self._late_from_outcome(outcome)
                await self._state_lock.acquire()
                try:
                    current = self.status(record.run_id)
                    pair = next(
                        (
                            item
                            for item in current.pairs
                            if item.scheduled.pair_id == scheduled.pair_id
                        ),
                        None,
                    )
                    if pair is not None:
                        matching_timeout = any(
                            item.system is system
                            and item.status is OutcomeStatus.TIMEOUT
                            for item in pair.outcomes
                        )
                        outcomes = tuple(
                            item.model_copy(update={"late_completion": late})
                            if item.system is system
                            and item.status is OutcomeStatus.TIMEOUT
                            else item
                            for item in pair.outcomes
                        )
                        already_persisted = any(
                            item.system is system
                            and item.late_completion is not None
                            for item in pair.outcomes
                        )
                        if matching_timeout:
                            updated_pair = pair.model_copy(update={"outcomes": outcomes})
                            event = OrchestrationTraceEvent(
                                sequence=len(current.trace) + 1,
                                kind="late",
                                occurred_at=self._clock(),
                                pair_id=scheduled.pair_id,
                                system=system,
                                detail=f"late_{late.status.value}",
                            )
                            self._persist(
                                current,
                                pairs=self._replace_pair(current, updated_pair),
                                trace=current.trace + (event,),
                            )
                            evidence_persisted = True
                        elif already_persisted:
                            evidence_persisted = True
                finally:
                    self._state_lock.release()
            finally:
                try:
                    if evidence_persisted and work.done:
                        self.store.release_execution(record.run_id, owner_id)
                    else:
                        self.store.retain_orphaned_execution(
                            record.run_id,
                            owner_id=owner_id,
                        )
                finally:
                    self._deferred_leases.discard(lease_key)

        monitor = asyncio.create_task(
            _observe(),
            name=f"lab-late-{record.run_id}-{scheduled.pair_id}-{system.value}",
        )
        self._late_monitors.add(monitor)
        monitor.add_done_callback(self._late_monitors.discard)
        await started.wait()

    async def _run_side(
        self,
        *,
        record: PairedRunResult,
        scenario: ScenarioDefinition,
        scheduled: ScheduledPair,
        system: MeasuredSystem,
        owner_id: UUID,
    ) -> PersistedSideOutcome:
        runner = self._runners.get(system)
        if runner is None:
            return PersistedSideOutcome(
                system=system,
                status=OutcomeStatus.UNAVAILABLE,
                reason="side runner is not configured",
            )
        work = self._start_side_work(
            runner,
            {
                "run_id": record.run_id,
                "system": system,
                "scheduled": scheduled,
                "scenario": scenario,
            },
            name=f"lab-{record.run_id}-{scheduled.pair_id}-{system.value}",
        )
        self._watch_task(work.task)
        cancellation_sent = False
        try:
            await asyncio.wait({work.task}, timeout=self._side_timeout)
            if work.done:
                await self._finalize_completed_work(work)
                return self._completed_outcome(
                    work,
                    record=record,
                    scenario=scenario,
                    system=system,
                )

            if work.thread is None:
                work.cancel()
                cancellation_sent = True
            late: LateCompletion | None = None
            if self._late_grace:
                await asyncio.wait({work.task}, timeout=self._late_grace)
                if work.done:
                    await self._finalize_completed_work(work)
                    late = self._late_from_outcome(
                        self._completed_outcome(
                            work,
                            record=record,
                            scenario=scenario,
                            system=system,
                        )
                    )
            if not work.done:
                await self._schedule_late_monitor(
                    work,
                    record=record,
                    scenario=scenario,
                    scheduled=scheduled,
                    system=system,
                    owner_id=owner_id,
                )
            return PersistedSideOutcome(
                system=system,
                status=OutcomeStatus.TIMEOUT,
                reason="side exceeded the configured timeout; no retry was attempted",
                late_completion=late,
            )
        except asyncio.CancelledError:
            if work.thread is None and not cancellation_sent:
                work.cancel()
            await self._join_supervised_work(work)
            completed = self._completed_outcome(
                work,
                record=record,
                scenario=scenario,
                system=system,
            )
            raise _SideAttemptCancelled(
                PersistedSideOutcome(
                    system=system,
                    status=OutcomeStatus.FAILURE,
                    reason="paired run execution was cancelled; side was joined",
                    late_completion=self._late_from_outcome(completed),
                )
            )

    async def execute(self, run_id: UUID) -> PairedRunResult:
        record = self.status(run_id)
        self._require_compatible_mode(record)
        if record.status.terminal:
            return record
        if record.status is not RunStatus.QUEUED:
            raise RunConflict("run is already active")
        if self._state_lock.locked():
            raise RunConflict("a state-changing Evaluation Lab run is active")
        await self._state_lock.acquire()
        owner_id = uuid4()
        lease_acquired = False
        try:
            record = self.status(run_id)
            if record.status.terminal:
                return record
            try:
                self.store.acquire_execution(run_id, owner_id)
            except RuntimeError as exc:
                raise RunConflict("a state-changing Evaluation Lab run is active") from exc
            lease_acquired = True

            contracts = {
                contract.profile_id: contract for contract in record.snapshot_contracts
            }
            executable = any(
                system in self._runners
                for scheduled in record.schedule
                for system in scheduled.order.systems
            )
            missing_contracts = sorted(
                {
                    self._scenarios[scheduled.scenario_id].reset_profile.profile_id
                    for scheduled in record.schedule
                    if self._scenarios[
                        scheduled.scenario_id
                    ].reset_profile.profile_id
                    not in contracts
                }
            )
            if executable and missing_contracts:
                reason = "immutable snapshot contract is unavailable"
                return self._transition(
                    self._persist(record, blocked_reason=reason),
                    RunStatus.BLOCKED,
                    detail=reason,
                )
            if (
                executable
                and record.execution_mode is ExecutionMode.MEASURED
                and self._budget_guard is None
            ):
                reason = "measured execution requires a budget reservation guard"
                return self._transition(
                    self._persist(record, blocked_reason=reason),
                    RunStatus.BLOCKED,
                    detail=reason,
                )

            for scheduled in record.schedule:
                scenario = self._scenarios[scheduled.scenario_id]
                contract = contracts.get(scenario.reset_profile.profile_id)
                pair = PairExecutionRecord(scheduled=scheduled)
                for system in scheduled.order.systems:
                    record = self._transition(record, RunStatus.RESETTING)
                    try:
                        verification = await _await(
                            self._resetter(
                                run_id=record.run_id,
                                system=system,
                                scheduled=scheduled,
                                scenario=scenario,
                            )
                        )
                        if not isinstance(verification, StateVerification):
                            raise ValueError("resetter returned malformed verification")
                    except Exception as exc:
                        reason = f"reset failed before transport: {type(exc).__name__}"
                        return self._transition(
                            self._persist(record, blocked_reason=reason),
                            RunStatus.BLOCKED,
                            detail=reason,
                        )
                    pair = pair.model_copy(
                        update={"verifications": pair.verifications + (verification,)}
                    )
                    mismatches = list(verification.mismatches)
                    if verification.system is not system:
                        mismatches.append("system")
                    if contract is None:
                        mismatches.append("snapshot_contract")
                    else:
                        expected_roster = contract.roster_fingerprint(system)
                        pinned = (
                            (
                                "snapshot_id",
                                verification.snapshot_id,
                                contract.snapshot_id,
                            ),
                            (
                                "roster_fingerprint",
                                verification.roster_fingerprint,
                                expected_roster,
                            ),
                            (
                                "expected_roster_fingerprint",
                                verification.expected_roster_fingerprint,
                                expected_roster,
                            ),
                            (
                                "fixture_fingerprint",
                                verification.fixture_fingerprint,
                                contract.fixture_fingerprint,
                            ),
                            (
                                "expected_fixture_fingerprint",
                                verification.expected_fixture_fingerprint,
                                contract.fixture_fingerprint,
                            ),
                            (
                                "raw_journal_fingerprint",
                                verification.raw_journal_fingerprint,
                                contract.raw_journal_fingerprint,
                            ),
                            (
                                "expected_raw_journal_fingerprint",
                                verification.expected_raw_journal_fingerprint,
                                contract.raw_journal_fingerprint,
                            ),
                        )
                        mismatches.extend(
                            name for name, observed, expected in pinned if observed != expected
                        )
                    mismatches = list(dict.fromkeys(mismatches))
                    record = self._persist_pair_event(
                        record,
                        pair,
                        kind="reset",
                        system=system,
                        detail="verified" if not mismatches else ",".join(mismatches),
                    )
                    if mismatches:
                        reason = "state verification mismatch: " + ", ".join(mismatches)
                        record = self._persist(record, blocked_reason=reason)
                        return self._transition(record, RunStatus.BLOCKED, detail=reason)

                    if record.execution_mode is ExecutionMode.MEASURED:
                        assert self._budget_guard is not None
                        try:
                            authorization = await _await(
                                self._budget_guard(
                                    run_id=record.run_id,
                                    system=system,
                                    scheduled=scheduled,
                                    scenario=scenario,
                                )
                            )
                        except BudgetExceeded:
                            outcome = PersistedSideOutcome(
                                system=system,
                                status=OutcomeStatus.BUDGET_STOP,
                                reason="cost reservation refused before side transport",
                            )
                            pair = pair.model_copy(
                                update={"outcomes": pair.outcomes + (outcome,)}
                            )
                            record = self._persist_pair_event(
                                record,
                                pair,
                                kind="outcome",
                                system=system,
                                detail=outcome.status.value,
                            )
                            continue
                        except Exception as exc:
                            reason = (
                                "budget reservation failed before transport: "
                                f"{type(exc).__name__}"
                            )
                            return self._transition(
                                self._persist(record, blocked_reason=reason),
                                RunStatus.BLOCKED,
                                detail=reason,
                            )
                        if not isinstance(authorization, BudgetAuthorization):
                            reason = "budget reservation authorization was malformed"
                            return self._transition(
                                self._persist(record, blocked_reason=reason),
                                RunStatus.BLOCKED,
                                detail=reason,
                            )

                    running = (
                        RunStatus.BASELINE_RUNNING
                        if system is MeasuredSystem.BASELINE
                        else RunStatus.ENHANCED_RUNNING
                    )
                    record = self._transition(record, running)
                    try:
                        outcome = await self._run_side(
                            record=record,
                            scenario=scenario,
                            scheduled=scheduled,
                            system=system,
                            owner_id=owner_id,
                        )
                    except _SideAttemptCancelled as exc:
                        outcome = exc.outcome
                        pair = pair.model_copy(
                            update={"outcomes": pair.outcomes + (outcome,)}
                        )
                        record = self._persist_pair_event(
                            record,
                            pair,
                            kind="outcome",
                            system=system,
                            detail=outcome.status.value,
                        )
                        self._transition(
                            record,
                            RunStatus.PARTIAL_FAILURE,
                            detail="paired run execution was cancelled after joining side",
                        )
                        raise
                    pair = pair.model_copy(
                        update={"outcomes": pair.outcomes + (outcome,)}
                    )
                    record = self._persist_pair_event(
                        record,
                        pair,
                        kind="outcome",
                        system=system,
                        detail=outcome.status.value,
                    )
                    if self.active_tasks:
                        reason = (
                            "timed-out side task remained active after cancellation; "
                            "the paired run was blocked before another reset"
                        )
                        record = self._persist(record, blocked_reason=reason)
                        return self._transition(
                            record,
                            RunStatus.BLOCKED,
                            detail=reason,
                        )

            record = self._transition(record, RunStatus.GRADING)
            scorecards: list[PairedSequenceScorecard] = []
            for pair in record.pairs:
                scenario = self._scenarios[pair.scheduled.scenario_id]
                if len(pair.outcomes) != 2:
                    continue
                for outcome in pair.outcomes:
                    if outcome.status is OutcomeStatus.SUCCESS:
                        try:
                            graded = grade_scenario_sequence(scenario, outcome.results)
                            scorecards.append(
                                PairedSequenceScorecard(
                                    **graded.model_dump(exclude_computed_fields=True),
                                    pair_id=pair.scheduled.pair_id,
                                    repetition=pair.scheduled.repetition,
                                )
                            )
                        except Exception as exc:
                            malformed = outcome.model_copy(
                                update={
                                    "status": OutcomeStatus.MALFORMED,
                                    "reason": (
                                        "deterministic grading rejected terminal evidence: "
                                        f"{type(exc).__name__}"
                                    ),
                                }
                            )
                            outcomes = tuple(
                                malformed if item.system is outcome.system else item
                                for item in pair.outcomes
                            )
                            pair = pair.model_copy(update={"outcomes": outcomes})
                            record = self._persist_pair_event(
                                record,
                                pair,
                                kind="outcome",
                                system=outcome.system,
                                detail=OutcomeStatus.MALFORMED.value,
                            )
            if scorecards:
                event = OrchestrationTraceEvent(
                    sequence=len(record.trace) + 1,
                    kind="grade",
                    occurred_at=self._clock(),
                    detail=f"graded {len(scorecards)} terminal successful sides",
                )
                record = self._persist(
                    record,
                    scorecards=tuple(scorecards),
                    trace=record.trace + (event,),
                )
            outcomes = [outcome for pair in record.pairs for outcome in pair.outcomes]
            complete = (
                len(outcomes) == len(record.schedule) * 2
                and all(outcome.status is OutcomeStatus.SUCCESS for outcome in outcomes)
            )
            return self._transition(
                record,
                RunStatus.COMPLETE if complete else RunStatus.PARTIAL_FAILURE,
            )
        finally:
            self._state_lock.release()
            if lease_acquired and (run_id, owner_id) not in self._deferred_leases:
                self.store.release_execution(run_id, owner_id)


def project_lab_run(record: PairedRunResult) -> PairedRunResult:
    """Project a PairedRunResult, verifying and disclosing controlled fixture bodies fail-closed."""
    from evals.live_lab.fixture_email import (
        _TEMPLATES,
        build_fact_manifest,
        render_fixture_messages,
    )
    from evals.live_lab.scenarios import ScenarioTrack, load_controlled_scenarios

    scenarios = {item.scenario_id: item for item in load_controlled_scenarios()}
    canonical_templates = {template.fact_id: template for template in _TEMPLATES}

    new_pairs = []
    for pair in record.pairs:
        scenario = scenarios.get(pair.scheduled.scenario_id)
        is_controlled = (
            scenario is not None
            and getattr(scenario, "track", None) == ScenarioTrack.CONTROLLED
        )

        new_outcomes = []
        for outcome in pair.outcomes:
            new_results = []
            for result in outcome.results:
                if (
                    result.gmail_evidence is None
                    or not result.gmail_evidence.value
                ):
                    new_results.append(result)
                    continue

                new_events = []
                modified_events = False
                for event in result.gmail_evidence.value:
                    if not isinstance(event, dict) or "controlled_fixture_evidence" not in event:
                        new_events.append(event)
                        continue

                    if not is_controlled:
                        event_copy = dict(event)
                        event_copy.pop("controlled_fixture_evidence", None)
                        new_events.append(event_copy)
                        modified_events = True
                        continue

                    raw_evidence = event["controlled_fixture_evidence"]
                    if not isinstance(raw_evidence, (list, tuple)):
                        event_copy = dict(event)
                        event_copy.pop("controlled_fixture_evidence", None)
                        new_events.append(event_copy)
                        modified_events = True
                        continue

                    verified_items = []
                    evidence_valid = True
                    allowed_keys = {
                        "fact_id",
                        "fabricated",
                        "content",
                        "fixture_run_id",
                        "manifest_sha256",
                    }
                    for item in raw_evidence:
                        if not isinstance(item, dict):
                            evidence_valid = False
                            break
                        if not set(item.keys()).issubset(allowed_keys):
                            evidence_valid = False
                            break
                        if item.get("fabricated") is not True:
                            evidence_valid = False
                            break
                        fact_id = item.get("fact_id")
                        if not isinstance(fact_id, str) or fact_id not in canonical_templates:
                            evidence_valid = False
                            break

                        canonical_template = canonical_templates[fact_id]
                        canonical_body = canonical_template.body

                        fixture_run_id = item.get("fixture_run_id")
                        manifest_sha256 = item.get("manifest_sha256")
                        if fixture_run_id is not None or manifest_sha256 is not None:
                            if not isinstance(fixture_run_id, str) or not isinstance(manifest_sha256, str):
                                evidence_valid = False
                                break
                            try:
                                expected_manifest = build_fact_manifest(
                                    render_fixture_messages(fixture_run_id)
                                )
                            except Exception:
                                evidence_valid = False
                                break
                            if manifest_sha256 != expected_manifest.manifest_sha256:
                                evidence_valid = False
                                break
                            if fact_id not in expected_manifest.facts:
                                evidence_valid = False
                                break

                        if "content" in item:
                            if item["content"] != canonical_body:
                                evidence_valid = False
                                break

                        verified_items.append(
                            {
                                "fact_id": fact_id,
                                "fabricated": True,
                                "content": canonical_body,
                            }
                        )

                    event_copy = dict(event)
                    if evidence_valid and verified_items:
                        event_copy["controlled_fixture_evidence"] = verified_items
                    else:
                        event_copy.pop("controlled_fixture_evidence", None)
                    new_events.append(event_copy)
                    modified_events = True

                if modified_events:
                    updated_gmail = result.gmail_evidence.model_copy(
                        update={"value": new_events}
                    )
                    new_results.append(
                        result.model_copy(update={"gmail_evidence": updated_gmail})
                    )
                else:
                    new_results.append(result)

            new_outcomes.append(outcome.model_copy(update={"results": tuple(new_results)}))

        new_pairs.append(pair.model_copy(update={"outcomes": tuple(new_outcomes)}))

    return record.model_copy(update={"pairs": tuple(new_pairs)})


__all__ = [
    "BudgetAuthorization",
    "ExecutionMode",
    "LateCompletion",
    "OrchestrationTraceEvent",
    "PairExecutionRecord",
    "PairedSequenceScorecard",
    "PairedRunOrchestrator",
    "PairedRunResult",
    "PersistedSideOutcome",
    "RunConflict",
    "RunHandle",
    "RunStatus",
    "RunTransition",
    "SideRunOutput",
    "SnapshotContract",
    "StartRunRequest",
    "StateVerification",
    "project_lab_run",
]
