"""Credential-free paired-run orchestration contracts."""

from __future__ import annotations

import asyncio
import os
import time
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid5

import pytest

from evals.live_lab.contracts import StateFingerprint
from evals.live_lab.scenarios import load_controlled_scenarios
from server.services.evaluation_lab.budget import BudgetExceeded, LedgerSnapshot
from server.services.evaluation_lab.models import SystemRunResult
from server.services.evaluation_lab.orchestrator import (
    BudgetAuthorization,
    ExecutionMode,
    PairedRunOrchestrator,
    RunConflict,
    RunStatus,
    SideRunOutput,
    SnapshotContract,
    StartRunRequest,
    StateVerification,
)
from server.services.evaluation_lab.repetitions import MeasuredSystem, OutcomeStatus
from server.services.evaluation_lab.run_store import RunStore


REQUEST_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _scenario():
    return load_controlled_scenarios()[0]


def _verification(system: MeasuredSystem, snapshot_id: str = "snapshot-standard-100"):
    return StateVerification(
        system=system,
        snapshot_id=snapshot_id,
        roster_fingerprint=f"{system.value}-roster",
        expected_roster_fingerprint=f"{system.value}-roster",
        fixture_fingerprint="fixture-1313-100",
        expected_fixture_fingerprint="fixture-1313-100",
        raw_journal_fingerprint="journal-bytes-identical",
        expected_raw_journal_fingerprint="journal-bytes-identical",
    )


def _snapshot_contract() -> SnapshotContract:
    return SnapshotContract(
        profile_id="standard-100",
        snapshot_id="snapshot-standard-100",
        baseline_roster_fingerprint="baseline-roster",
        enhanced_roster_fingerprint="enhanced-roster",
        fixture_fingerprint="fixture-1313-100",
        raw_journal_fingerprint="journal-bytes-identical",
    )


def _offline_orchestrator(**kwargs) -> PairedRunOrchestrator:
    kwargs.setdefault("execution_mode", ExecutionMode.OFFLINE_FAKE)
    kwargs.setdefault(
        "snapshot_contracts", {"standard-100": _snapshot_contract()}
    )
    return PairedRunOrchestrator(**kwargs)


def _result(run_id: UUID, system: MeasuredSystem, pair_id: UUID, turn: int = 0):
    return SystemRunResult(
        run_id=run_id,
        turn_id=uuid5(pair_id, f"{system.value}:turn:{turn}"),
        system=system.value,
    )


def test_successful_side_output_requires_bounded_observed_model_provenance() -> None:
    result = _result(REQUEST_ID, MeasuredSystem.BASELINE, REQUEST_ID)

    with pytest.raises(ValueError, match="model"):
        SideRunOutput(status=OutcomeStatus.SUCCESS, model_id="   ", results=(result,))


def test_state_verification_maps_the_existing_fixture_fingerprint_contract() -> None:
    expected = StateFingerprint(
        file_sha256={"execution_agents/roster.json": "a" * 64},
        roster_sha256="b" * 64,
        roster_count=100,
        logical_identity_digest="c" * 64,
        raw_journal_digest="d" * 64,
        journal_bytes=1000,
        journal_entries=12,
        sentinel_checks={"fixture": True},
    )
    actual = expected.model_copy(update={"raw_journal_digest": "e" * 64})

    verification = StateVerification.from_fingerprints(
        system=MeasuredSystem.ENHANCED,
        snapshot_id="standard-100",
        actual=actual,
        expected=expected,
    )

    assert verification.roster_fingerprint == "b" * 64
    assert verification.fixture_fingerprint == "c" * 64
    assert verification.mismatches == ("raw_journal_fingerprint",)


@pytest.mark.anyio
async def test_runs_three_alternating_pairs_with_independent_verified_resets(tmp_path) -> None:
    scenario = _scenario()
    resets: list[tuple[UUID, MeasuredSystem]] = []
    budget_checks: list[tuple[UUID, MeasuredSystem]] = []
    calls: list[tuple[UUID, MeasuredSystem]] = []

    async def resetter(*, system, scheduled, **_kwargs):
        resets.append((scheduled.pair_id, system))
        return _verification(system)

    async def budget_guard(*, system, scheduled, **_kwargs):
        budget_checks.append((scheduled.pair_id, system))
        return BudgetAuthorization(reservation_id=scheduled.pair_id)

    async def runner(*, run_id, system, scheduled, scenario, **_kwargs):
        calls.append((scheduled.pair_id, system))
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=tuple(
                _result(run_id, system, scheduled.pair_id, index)
                for index, _turn in enumerate(scenario.turns)
            ),
        )

    orchestrator = PairedRunOrchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
        budget_guard=budget_guard,
        snapshot_contracts={"standard-100": _snapshot_contract()},
    )
    handle = await orchestrator.start(
        StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
    )
    result = await orchestrator.execute(handle.run_id)

    expected_orders = [
        (MeasuredSystem.BASELINE, MeasuredSystem.ENHANCED),
        (MeasuredSystem.ENHANCED, MeasuredSystem.BASELINE),
        (MeasuredSystem.BASELINE, MeasuredSystem.ENHANCED),
    ]
    assert result.status is RunStatus.COMPLETE
    assert len(result.pairs) == 3
    assert [tuple(item.system for item in pair.outcomes) for pair in result.pairs] == expected_orders
    assert calls == resets == budget_checks
    assert len(result.scorecards) == 6
    assert [item.status for item in result.transitions] == [
        RunStatus.QUEUED,
        RunStatus.RESETTING,
        RunStatus.BASELINE_RUNNING,
        RunStatus.RESETTING,
        RunStatus.ENHANCED_RUNNING,
        RunStatus.RESETTING,
        RunStatus.ENHANCED_RUNNING,
        RunStatus.RESETTING,
        RunStatus.BASELINE_RUNNING,
        RunStatus.RESETTING,
        RunStatus.BASELINE_RUNNING,
        RunStatus.RESETTING,
        RunStatus.ENHANCED_RUNNING,
        RunStatus.GRADING,
        RunStatus.COMPLETE,
    ]


@pytest.mark.anyio
async def test_mismatched_fingerprint_blocks_before_budget_or_side_transport(tmp_path) -> None:
    scenario = _scenario()
    budget_calls = 0
    side_calls = 0

    async def resetter(*, system, **_kwargs):
        valid = _verification(system)
        return valid.model_copy(update={"raw_journal_fingerprint": "mutated"})

    async def budget_guard(**_kwargs):
        nonlocal budget_calls
        budget_calls += 1

    async def runner(**_kwargs):
        nonlocal side_calls
        side_calls += 1
        raise AssertionError("side transport must not run after reset mismatch")

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
        budget_guard=budget_guard,
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.BLOCKED
    assert result.pairs[0].outcomes == ()
    assert result.scorecards == ()
    assert budget_calls == side_calls == 0
    assert "raw_journal_fingerprint" in (result.blocked_reason or "")


@pytest.mark.anyio
async def test_second_side_failure_preserves_first_success_and_never_retries(tmp_path) -> None:
    scenario = _scenario()
    attempts = {system: 0 for system in MeasuredSystem}

    async def resetter(*, system, **_kwargs):
        return _verification(system)

    async def runner(*, run_id, system, scheduled, **_kwargs):
        attempts[system] += 1
        if system is MeasuredSystem.ENHANCED:
            raise RuntimeError("sanitized enhanced failure")
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=(_result(run_id, system, scheduled.pair_id),),
        )

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.PARTIAL_FAILURE
    assert attempts == {MeasuredSystem.BASELINE: 3, MeasuredSystem.ENHANCED: 3}
    first = result.pairs[0]
    assert first.outcomes[0].status is OutcomeStatus.SUCCESS
    assert first.outcomes[0].results
    assert first.outcomes[1].status is OutcomeStatus.FAILURE
    assert first.outcomes[1].attempt_count == 1
    assert len(result.scorecards) == 3


@pytest.mark.anyio
async def test_timeout_late_completion_is_recorded_without_becoming_success(tmp_path) -> None:
    scenario = _scenario()

    async def resetter(*, system, **_kwargs):
        return _verification(system)

    async def runner(*, run_id, system, scheduled, **_kwargs):
        if system is MeasuredSystem.BASELINE:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                return SideRunOutput(
                    status=OutcomeStatus.SUCCESS,
                    model_id="openai/gpt-4.1-mini",
                    results=(_result(run_id, system, scheduled.pair_id),),
                )
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=(_result(run_id, system, scheduled.pair_id),),
        )

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
        side_timeout_seconds=0.001,
        late_completion_grace_seconds=0.05,
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    baseline = [
        outcome
        for pair in result.pairs
        for outcome in pair.outcomes
        if outcome.system is MeasuredSystem.BASELINE
    ]
    assert all(item.status is OutcomeStatus.TIMEOUT for item in baseline)
    assert all(item.late_completion is not None for item in baseline)
    assert all(item.late_completion.status is OutcomeStatus.SUCCESS for item in baseline)
    assert result.status is RunStatus.PARTIAL_FAILURE
    assert not orchestrator.active_tasks


@pytest.mark.anyio
async def test_cancellation_resistant_side_blocks_before_other_side_reset(tmp_path) -> None:
    scenario = _scenario()
    release = asyncio.Event()
    calls: list[MeasuredSystem] = []

    async def resetter(*, system, **_kwargs):
        calls.append(system)
        return _verification(system)

    async def runner(*, run_id, system, scheduled, **_kwargs):
        if system is MeasuredSystem.BASELINE:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await release.wait()
                return SideRunOutput(
                    status=OutcomeStatus.SUCCESS,
                    model_id="openai/gpt-4.1-mini",
                    results=(_result(run_id, system, scheduled.pair_id),),
                )
        raise AssertionError("the second side must not start while a late task is active")

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
        side_timeout_seconds=0.001,
        late_completion_grace_seconds=0.001,
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.BLOCKED
    assert calls == [MeasuredSystem.BASELINE]
    assert result.pairs[0].outcomes[0].status is OutcomeStatus.TIMEOUT
    assert orchestrator.active_tasks
    release.set()
    await asyncio.gather(*orchestrator.active_tasks)
    assert not orchestrator.active_tasks


@pytest.mark.anyio
async def test_budget_malformed_and_policy_failures_are_terminal_without_retry(tmp_path) -> None:
    scenario = _scenario()
    attempts = {system: 0 for system in MeasuredSystem}

    async def resetter(*, system, **_kwargs):
        return _verification(system)

    async def budget_guard(*, system, scheduled, **_kwargs):
        if system is MeasuredSystem.BASELINE and scheduled.repetition == 1:
            raise BudgetExceeded(
                LedgerSnapshot(
                    cap_usd=Decimal("1"),
                    spent_usd=Decimal("1"),
                    reserved_usd=Decimal("0"),
                    remaining_usd=Decimal("0"),
                    entries=(),
                ),
                Decimal("0.01"),
            )
        return BudgetAuthorization(reservation_id=scheduled.pair_id)

    async def runner(*, system, scheduled, **_kwargs):
        attempts[system] += 1
        if system is MeasuredSystem.ENHANCED and scheduled.repetition == 1:
            return {"unexpected": "malformed"}
        return SideRunOutput(
            status=OutcomeStatus.FAILURE,
            reason="policy rejected before callable execution",
        )

    orchestrator = PairedRunOrchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
        budget_guard=budget_guard,
        snapshot_contracts={"standard-100": _snapshot_contract()},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    first = result.pairs[0].outcomes
    assert first[0].status is OutcomeStatus.BUDGET_STOP
    assert first[1].status is OutcomeStatus.MALFORMED
    assert attempts[MeasuredSystem.BASELINE] == 2
    assert attempts[MeasuredSystem.ENHANCED] == 3
    assert all(item.attempt_count == 1 for pair in result.pairs for item in pair.outcomes)
    assert result.status is RunStatus.PARTIAL_FAILURE


@pytest.mark.anyio
async def test_outer_cancellation_retains_ownership_until_side_is_joined(tmp_path) -> None:
    scenario = _scenario()
    started = asyncio.Event()
    release = asyncio.Event()

    async def resetter(*, system, **_kwargs):
        return _verification(system)

    async def runner(*, run_id, system, scheduled, **_kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=(_result(run_id, system, scheduled.pair_id),),
        )

    root = tmp_path / ".lab" / "runs"
    owner = _offline_orchestrator(
        store=RunStore(root),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
    )
    run_id = (
        await owner.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id
    execution = asyncio.create_task(owner.execute(run_id))
    await started.wait()
    execution.cancel()
    await asyncio.sleep(0)

    try:
        assert not execution.done(), "outer cancellation must join or isolate the live side"
        second = _offline_orchestrator(
            store=RunStore(root),
            scenarios=(scenario,),
            resetter=resetter,
            runners={system: runner for system in MeasuredSystem},
        )
        with pytest.raises(RunConflict, match="active"):
            await second.start(
                StartRunRequest(
                    request_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
                    scenario_ids=(scenario.scenario_id,),
                )
            )
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await execution
    terminal = owner.status(run_id)
    assert terminal.status is RunStatus.PARTIAL_FAILURE
    assert terminal.pairs[0].outcomes[0].status is OutcomeStatus.FAILURE


@pytest.mark.anyio
async def test_synchronous_runner_is_supervised_without_blocking_event_loop(tmp_path) -> None:
    scenario = _scenario()

    async def resetter(*, system, **_kwargs):
        return _verification(system)

    def runner(**_kwargs):
        time.sleep(0.1)
        raise RuntimeError("synthetic synchronous failure")

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
        side_timeout_seconds=0.001,
        late_completion_grace_seconds=0.001,
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id
    started_at = time.monotonic()
    heartbeat = asyncio.create_task(asyncio.sleep(0.01))

    result = await orchestrator.execute(run_id)
    await heartbeat
    heartbeat_elapsed = time.monotonic() - started_at

    assert heartbeat_elapsed < 0.08
    assert result.status is RunStatus.BLOCKED
    if orchestrator.active_tasks:
        await asyncio.gather(*orchestrator.active_tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_cross_system_and_repetition_fingerprints_are_immutable(tmp_path) -> None:
    scenario = _scenario()
    transports = 0

    async def resetter(*, system, scheduled, **_kwargs):
        value = _verification(system)
        return value.model_copy(
            update={
                "fixture_fingerprint": f"{system.value}-fixture-{scheduled.repetition}",
                "expected_fixture_fingerprint": f"{system.value}-fixture-{scheduled.repetition}",
                "raw_journal_fingerprint": f"{system.value}-journal-{scheduled.repetition}",
                "expected_raw_journal_fingerprint": f"{system.value}-journal-{scheduled.repetition}",
            }
        )

    async def runner(*, run_id, system, scheduled, **_kwargs):
        nonlocal transports
        transports += 1
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=(_result(run_id, system, scheduled.pair_id),),
        )

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=resetter,
        runners={system: runner for system in MeasuredSystem},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.BLOCKED
    assert transports == 0


@pytest.mark.anyio
async def test_measured_runners_cannot_execute_without_budget_authorization(tmp_path) -> None:
    scenario = _scenario()
    transports = 0

    async def runner(**_kwargs):
        nonlocal transports
        transports += 1
        raise AssertionError("measured transport must be budget authorized")

    orchestrator = PairedRunOrchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
        snapshot_contracts={"standard-100": _snapshot_contract()},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.BLOCKED
    assert transports == 0


@pytest.mark.anyio
async def test_malformed_budget_authorization_blocks_before_transport(tmp_path) -> None:
    scenario = _scenario()
    transports = 0

    async def runner(**_kwargs):
        nonlocal transports
        transports += 1
        raise AssertionError("malformed authorization must fail closed")

    orchestrator = PairedRunOrchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
        budget_guard=lambda **_kwargs: True,
        snapshot_contracts={"standard-100": _snapshot_contract()},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.BLOCKED
    assert transports == 0


@pytest.mark.anyio
async def test_duplicate_turn_sequence_is_malformed_and_grading_terminalizes(tmp_path) -> None:
    scenario = next(item for item in load_controlled_scenarios() if len(item.turns) == 2)

    async def runner(*, run_id, system, scheduled, **_kwargs):
        duplicated = _result(run_id, system, scheduled.pair_id)
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=(duplicated, duplicated),
        )

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.PARTIAL_FAILURE
    assert all(
        outcome.status is OutcomeStatus.MALFORMED
        for pair in result.pairs
        for outcome in pair.outcomes
    )


@pytest.mark.anyio
async def test_grading_exception_is_contained_as_terminal_malformed_evidence(
    tmp_path, monkeypatch
) -> None:
    scenario = _scenario()

    async def runner(*, run_id, system, scheduled, **_kwargs):
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=(_result(run_id, system, scheduled.pair_id),),
        )

    def reject_grading(*_args, **_kwargs):
        raise ValueError("synthetic private grading detail")

    monkeypatch.setattr(
        "server.services.evaluation_lab.orchestrator.grade_scenario_sequence",
        reject_grading,
    )
    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    assert result.status is RunStatus.PARTIAL_FAILURE
    assert not result.scorecards
    assert all(
        outcome.status is OutcomeStatus.MALFORMED
        for pair in result.pairs
        for outcome in pair.outcomes
    )
    assert b"synthetic private grading detail" not in (tmp_path / ".lab" / "runs" / f"{run_id}.json").read_bytes()


@pytest.mark.anyio
async def test_bare_result_never_invents_model_provenance(tmp_path) -> None:
    scenario = _scenario()

    async def runner(*, run_id, system, scheduled, **_kwargs):
        return _result(run_id, system, scheduled.pair_id)

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id

    result = await orchestrator.execute(run_id)

    outcomes = [outcome for pair in result.pairs for outcome in pair.outcomes]
    assert all(outcome.model_id is None for outcome in outcomes)
    assert all(outcome.status is OutcomeStatus.MALFORMED for outcome in outcomes)


@pytest.mark.anyio
async def test_late_completion_after_grace_persists_full_evidence(tmp_path) -> None:
    scenario = _scenario()
    release = asyncio.Event()

    async def runner(*, run_id, system, scheduled, **_kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await release.wait()
            return SideRunOutput(
                status=OutcomeStatus.SUCCESS,
                model_id="openai/gpt-4.1-mini",
                results=(_result(run_id, system, scheduled.pair_id),),
            )

    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
        side_timeout_seconds=0.001,
        late_completion_grace_seconds=0.001,
    )
    run_id = (
        await orchestrator.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id
    result = await orchestrator.execute(run_id)
    assert result.status is RunStatus.BLOCKED

    release.set()
    await asyncio.gather(*orchestrator.active_tasks)
    await asyncio.sleep(0)
    observed = orchestrator.status(run_id).pairs[0].outcomes[0]

    assert observed.status is OutcomeStatus.TIMEOUT
    assert observed.late_completion is not None
    assert observed.late_completion.status is OutcomeStatus.SUCCESS
    assert observed.late_completion.results[0].run_id == run_id


@pytest.mark.anyio
async def test_cancelling_late_observer_cannot_release_live_execution_ownership(
    tmp_path,
) -> None:
    scenario = _scenario()
    release = asyncio.Event()

    async def runner(*, run_id, system, scheduled, **_kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await release.wait()
            return SideRunOutput(
                status=OutcomeStatus.SUCCESS,
                model_id="openai/gpt-4.1-mini",
                results=(_result(run_id, system, scheduled.pair_id),),
            )

    root = tmp_path / ".lab" / "runs"
    owner = _offline_orchestrator(
        store=RunStore(root),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
        side_timeout_seconds=0.001,
        late_completion_grace_seconds=0.001,
    )
    run_id = (
        await owner.start(
            StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))
        )
    ).run_id
    result = await owner.execute(run_id)
    assert result.status is RunStatus.BLOCKED
    monitor = next(
        task for task in owner.active_tasks if task.get_name().startswith("lab-late-")
    )
    monitor.cancel()
    await asyncio.sleep(0)
    assert not monitor.done()

    second = _offline_orchestrator(
        store=RunStore(root),
        scenarios=(scenario,),
        resetter=lambda **kwargs: _verification(kwargs["system"]),
        runners={system: runner for system in MeasuredSystem},
    )
    with pytest.raises(RunConflict, match="active"):
        await second.start(
            StartRunRequest(
                request_id=UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
                scenario_ids=(scenario.scenario_id,),
            )
        )

    release.set()
    await asyncio.gather(*owner.active_tasks, return_exceptions=True)
    await asyncio.sleep(0)
    observed = owner.status(run_id).pairs[0].outcomes[0]
    assert observed.late_completion is not None
    assert observed.late_completion.status is OutcomeStatus.SUCCESS
    assert RunStore(root).execution_lease() is None


@pytest.mark.anyio
async def test_duplicate_start_is_idempotent_and_other_active_run_is_refused(tmp_path) -> None:
    scenario = _scenario()
    orchestrator = _offline_orchestrator(
        store=RunStore(tmp_path / ".lab" / "runs"),
        scenarios=(scenario,),
        resetter=lambda **_kwargs: _verification(MeasuredSystem.BASELINE),
        runners={},
    )
    request = StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,))

    first = await orchestrator.start(request)
    duplicate = await orchestrator.start(request)

    assert duplicate == first
    with pytest.raises(RunConflict, match="active"):
        await orchestrator.start(
            StartRunRequest(
                request_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
                scenario_ids=(scenario.scenario_id,),
            )
        )
    assert orchestrator.status(first.run_id).status is RunStatus.QUEUED


def test_run_store_atomic_failure_preserves_previous_record(tmp_path, monkeypatch) -> None:
    scenario = _scenario()
    store = RunStore(tmp_path / ".lab" / "runs")
    record = PairedRunOrchestrator.new_record(
        StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,)),
        scenarios=(scenario,),
    )
    store.create(record)
    original = store.get(record.run_id)
    assert original is not None
    changed = original.model_copy(
        update={"generation": original.generation + 1, "status": RunStatus.RESETTING}
    )

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected atomic replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.save(changed)

    assert store.get(record.run_id) == original
    assert not any(path.name.endswith(".tmp") for path in store.root.iterdir())


def test_run_store_short_write_failure_leaves_no_partial_or_temporary_file(
    tmp_path, monkeypatch
) -> None:
    scenario = _scenario()
    store = RunStore(tmp_path / ".lab" / "runs")
    record = PairedRunOrchestrator.new_record(
        StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,)),
        scenarios=(scenario,),
    )
    real_write = os.write
    calls = 0

    def fail_first_write(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected short write failure")
        return real_write(descriptor, payload)

    monkeypatch.setattr(os, "write", fail_first_write)
    with pytest.raises(OSError, match="injected"):
        store.create(record)

    assert store.get(record.run_id) is None
    assert not any(path.name.endswith(".tmp") for path in store.root.iterdir())


def test_restart_recovery_terminalizes_interrupted_run_without_retry(tmp_path) -> None:
    scenario = _scenario()
    store = RunStore(tmp_path / ".lab" / "runs")
    record = PairedRunOrchestrator.new_record(
        StartRunRequest(request_id=REQUEST_ID, scenario_ids=(scenario.scenario_id,)),
        scenarios=(scenario,),
    )
    store.create(record)
    interrupted = record.model_copy(
        update={"generation": 1, "status": RunStatus.BASELINE_RUNNING}
    )
    store.save(interrupted)

    _offline_orchestrator(
        store=RunStore(store.root),
        scenarios=(scenario,),
        resetter=lambda **_kwargs: _verification(MeasuredSystem.BASELINE),
        runners={},
    )
    recovered = RunStore(store.root).get(record.run_id)

    assert recovered is not None
    assert recovered.status is RunStatus.BLOCKED
    assert "restart" in (recovered.blocked_reason or "")
    assert recovered.generation == 2


def test_run_store_rejects_traversal_and_symlink_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="traversal"):
        RunStore(tmp_path / ".lab" / ".." / "runs")

    lab = tmp_path / ".lab"
    lab.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = lab / "linked-runs"
    linked.symlink_to(outside, target_is_directory=True)
    store = RunStore(linked)

    with pytest.raises(ValueError, match="symlink"):
        store.list_records()
    assert not list(outside.iterdir())
