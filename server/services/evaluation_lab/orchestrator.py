"""Serialized, restart-safe orchestration of isolated paired Evaluation Lab runs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    result_count: int = Field(ge=0)
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


class RunTransition(_FrozenModel):
    sequence: int = Field(ge=1)
    status: RunStatus
    occurred_at: datetime
    detail: str | None = None


class OrchestrationTraceEvent(_FrozenModel):
    sequence: int = Field(ge=1)
    kind: Literal["transition", "reset", "outcome", "grade", "recovery"]
    occurred_at: datetime
    pair_id: UUID | None = None
    system: MeasuredSystem | None = None
    detail: str


class PairedRunResult(_FrozenModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    request: StartRunRequest
    status: RunStatus
    generation: int = Field(ge=0)
    schedule: tuple[ScheduledPair, ...]
    pairs: tuple[PairExecutionRecord, ...] = ()
    scorecards: tuple[ScenarioSequenceScorecard, ...] = ()
    transitions: tuple[RunTransition, ...]
    trace: tuple[OrchestrationTraceEvent, ...]
    blocked_reason: str | None = None
    created_at: datetime
    updated_at: datetime


Resetter = Callable[..., StateVerification | Awaitable[StateVerification]]
Runner = Callable[..., SideRunOutput | SystemRunResult | Sequence[SystemRunResult] | Awaitable[Any]]
BudgetGuard = Callable[..., None | Awaitable[None]]


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
        self._side_timeout = side_timeout_seconds
        self._late_grace = late_completion_grace_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._state_lock = asyncio.Lock()
        self._active_tasks: set[asyncio.Task[Any]] = set()
        if recover_on_startup:
            self._recover_interrupted()

    @property
    def active_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(task for task in self._active_tasks if not task.done())

    @staticmethod
    def new_record(
        request: StartRunRequest,
        *,
        scenarios: Sequence[ScenarioDefinition],
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
            status=RunStatus.QUEUED,
            generation=0,
            schedule=schedule,
            transitions=(transition,),
            trace=(event,),
            created_at=occurred_at,
            updated_at=occurred_at,
        )

    def _recover_interrupted(self) -> None:
        for record in self.store.list_records():
            if record.status in {RunStatus.QUEUED, RunStatus.COMPLETE, RunStatus.PARTIAL_FAILURE, RunStatus.BLOCKED}:
                continue
            now = self._clock()
            terminal = RunStatus.PARTIAL_FAILURE if any(pair.outcomes for pair in record.pairs) else RunStatus.BLOCKED
            reason = "run interrupted by process restart; no side was retried"
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

    async def start(self, request: StartRunRequest) -> RunHandle:
        candidate = self.new_record(
            request,
            scenarios=tuple(self._scenarios.values()),
            now=self._clock(),
        )
        existing = self.store.get(candidate.run_id)
        if existing is not None:
            if existing.request != request:
                raise RunConflict("idempotency key belongs to a different request")
            return self._handle(existing)
        if self._state_lock.locked() or self.active_tasks:
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
        if isinstance(value, SystemRunResult):
            return SideRunOutput(
                status=OutcomeStatus.SUCCESS,
                model_id="openai/gpt-4.1-mini",
                results=(value,),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
            isinstance(item, SystemRunResult) for item in value
        ):
            return SideRunOutput(
                status=OutcomeStatus.SUCCESS,
                model_id="openai/gpt-4.1-mini",
                results=tuple(value),
            )
        raise ValueError("side runner returned malformed output")

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

    async def _run_side(
        self,
        *,
        record: PairedRunResult,
        scenario: ScenarioDefinition,
        scheduled: ScheduledPair,
        system: MeasuredSystem,
    ) -> PersistedSideOutcome:
        runner = self._runners.get(system)
        if runner is None:
            return PersistedSideOutcome(
                system=system,
                status=OutcomeStatus.UNAVAILABLE,
                reason="side runner is not configured",
            )
        task = asyncio.create_task(
            _await(
                runner(
                    run_id=record.run_id,
                    system=system,
                    scheduled=scheduled,
                    scenario=scenario,
                )
            ),
            name=f"lab-{record.run_id}-{scheduled.pair_id}-{system.value}",
        )
        self._watch_task(task)
        done, _pending = await asyncio.wait({task}, timeout=self._side_timeout)
        if done:
            try:
                output = self._normalize_output(task.result())
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
            if any(
                result.run_id != record.run_id or result.system != system.value
                for result in output.results
            ) or (
                output.status is OutcomeStatus.SUCCESS
                and len(output.results) != len(scenario.turns)
            ):
                return PersistedSideOutcome(
                    system=system,
                    status=OutcomeStatus.MALFORMED,
                    model_id=output.model_id,
                    results=output.results,
                    reason="side evidence does not match run, system, or predeclared turns",
                )
            return PersistedSideOutcome(
                system=system,
                status=output.status,
                model_id=output.model_id,
                results=output.results,
                reason=output.reason,
            )

        task.cancel()
        late: LateCompletion | None = None
        if self._late_grace:
            late_done, _ = await asyncio.wait({task}, timeout=self._late_grace)
            if late_done and not task.cancelled():
                try:
                    output = self._normalize_output(task.result())
                    late = LateCompletion(
                        status=output.status,
                        result_count=len(output.results),
                        reason=output.reason,
                    )
                except BaseException as exc:
                    late = LateCompletion(
                        status=OutcomeStatus.FAILURE,
                        result_count=0,
                        reason=f"{type(exc).__name__}: late completion failed",
                    )
        return PersistedSideOutcome(
            system=system,
            status=OutcomeStatus.TIMEOUT,
            reason="side exceeded the configured timeout; no retry was attempted",
            late_completion=late,
        )

    async def execute(self, run_id: UUID) -> PairedRunResult:
        record = self.status(run_id)
        if record.status.terminal:
            return record
        if record.status is not RunStatus.QUEUED:
            raise RunConflict("run is already active")
        if self._state_lock.locked():
            raise RunConflict("a state-changing Evaluation Lab run is active")
        await self._state_lock.acquire()
        try:
            record = self.status(run_id)
            if record.status.terminal:
                return record
            for scheduled in record.schedule:
                scenario = self._scenarios[scheduled.scenario_id]
                pair = PairExecutionRecord(scheduled=scheduled)
                expected_snapshot: str | None = None
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
                    if expected_snapshot is None:
                        expected_snapshot = verification.snapshot_id
                    elif verification.snapshot_id != expected_snapshot:
                        mismatches.append("snapshot_id")
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

                    if self._budget_guard is not None:
                        try:
                            reservation_ready = await _await(
                                self._budget_guard(
                                    run_id=record.run_id,
                                    system=system,
                                    scheduled=scheduled,
                                    scenario=scenario,
                                )
                            )
                        except BudgetExceeded:
                            reservation_ready = False
                        if reservation_ready is False:
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

                    running = (
                        RunStatus.BASELINE_RUNNING
                        if system is MeasuredSystem.BASELINE
                        else RunStatus.ENHANCED_RUNNING
                    )
                    record = self._transition(record, running)
                    outcome = await self._run_side(
                        record=record,
                        scenario=scenario,
                        scheduled=scheduled,
                        system=system,
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
            scorecards: list[ScenarioSequenceScorecard] = []
            for pair in record.pairs:
                scenario = self._scenarios[pair.scheduled.scenario_id]
                if len(pair.outcomes) != 2:
                    continue
                for outcome in pair.outcomes:
                    if outcome.status is OutcomeStatus.SUCCESS:
                        scorecards.append(grade_scenario_sequence(scenario, outcome.results))
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


__all__ = [
    "LateCompletion",
    "OrchestrationTraceEvent",
    "PairExecutionRecord",
    "PairedRunOrchestrator",
    "PairedRunResult",
    "PersistedSideOutcome",
    "RunConflict",
    "RunHandle",
    "RunStatus",
    "RunTransition",
    "SideRunOutput",
    "StartRunRequest",
    "StateVerification",
]
