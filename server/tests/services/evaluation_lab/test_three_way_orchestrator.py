"""Unit tests for Three-Way Orchestrator (Baseline vs Deterministic vs Jev)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import pytest

from evals.live_lab.scenarios import load_controlled_scenarios
from server.services.evaluation_lab.models import (
    Availability,
    ObservedValue,
    SystemRunResult,
)
from server.services.evaluation_lab.orchestrator import (
    ExecutionMode,
    PairedRunOrchestrator,
    PairedRunResult,
    RunStatus,
    SideRunOutput,
    SnapshotContract,
    StartRunRequest,
    StateVerification,
)
from server.services.evaluation_lab.repetitions import (
    MeasuredSystem,
    OutcomeStatus,
)
from server.services.evaluation_lab.run_store import RunStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_result(
    run_id: UUID,
    system: MeasuredSystem,
    turn_id: UUID,
    action: str = "reuse",
) -> SystemRunResult:
    return SystemRunResult(
        schema_version=1,
        run_id=run_id,
        turn_id=turn_id,
        system=system.value,
        mode=ObservedValue[str](availability=Availability.AVAILABLE, value="offline_fake"),
        decision=ObservedValue[dict](
            availability=Availability.AVAILABLE, value={"action": action}
        ),
        final_response=ObservedValue[str](
            availability=Availability.AVAILABLE, value="Turn completed"
        ),
    )


def _snapshot_contract() -> SnapshotContract:
    return SnapshotContract(
        profile_id="standard-100",
        snapshot_id="snapshot-standard-100",
        baseline_roster_fingerprint="baseline-roster",
        enhanced_roster_fingerprint="enhanced-roster",
        fixture_fingerprint="fixture-digest",
        raw_journal_fingerprint="journal-digest",
    )


def _verification(system: MeasuredSystem) -> StateVerification:
    roster_fp = (
        "baseline-roster" if system is MeasuredSystem.BASELINE else "enhanced-roster"
    )
    return StateVerification(
        system=system,
        snapshot_id="snapshot-standard-100",
        roster_fingerprint=roster_fp,
        expected_roster_fingerprint=roster_fp,
        fixture_fingerprint="fixture-digest",
        expected_fixture_fingerprint="fixture-digest",
        raw_journal_fingerprint="journal-digest",
        expected_raw_journal_fingerprint="journal-digest",
    )


@pytest.mark.anyio
async def test_three_way_orchestrator_execution(tmp_path: Path):
    store = RunStore(tmp_path / ".lab" / "runs")
    scenarios = [load_controlled_scenarios()[0]]

    executed_systems: list[MeasuredSystem] = []

    async def resetter(*, system: MeasuredSystem, **kwargs):
        return _verification(system)

    async def baseline_runner(*, run_id, system, scheduled, scenario, **kwargs):
        executed_systems.append(MeasuredSystem.BASELINE)
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=tuple(
                _make_result(
                    run_id=run_id,
                    system=system,
                    turn_id=uuid5(scheduled.pair_id, f"{system.value}:turn:{index}"),
                )
                for index, _turn in enumerate(scenario.turns)
            ),
        )

    async def deterministic_runner(*, run_id, system, scheduled, scenario, **kwargs):
        executed_systems.append(MeasuredSystem.ENHANCED_DETERMINISTIC)
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=tuple(
                _make_result(
                    run_id=run_id,
                    system=system,
                    turn_id=uuid5(scheduled.pair_id, f"{system.value}:turn:{index}"),
                )
                for index, _turn in enumerate(scenario.turns)
            ),
        )

    async def jev_runner(*, run_id, system, scheduled, scenario, **kwargs):
        executed_systems.append(MeasuredSystem.ENHANCED_JEV)
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="typesafe/jev-routing-v1",
            results=tuple(
                _make_result(
                    run_id=run_id,
                    system=system,
                    turn_id=uuid5(scheduled.pair_id, f"{system.value}:turn:{index}"),
                )
                for index, _turn in enumerate(scenario.turns)
            ),
        )

    runners = {
        MeasuredSystem.BASELINE: baseline_runner,
        MeasuredSystem.ENHANCED_DETERMINISTIC: deterministic_runner,
        MeasuredSystem.ENHANCED_JEV: jev_runner,
    }

    orch = PairedRunOrchestrator(
        store=store,
        scenarios=scenarios,
        resetter=resetter,
        runners=runners,
        execution_mode=ExecutionMode.OFFLINE_FAKE,
        snapshot_contracts={"standard-100": _snapshot_contract()},
        three_way=True,
    )

    request = StartRunRequest(scenario_ids=(scenarios[0].scenario_id,))
    handle = await orch.start(request)
    result = await orch.execute(handle.run_id)

    record = store.get(handle.run_id)
    assert record is not None
    assert record.status == RunStatus.COMPLETE
    assert record.schema_version == 2
    assert len(record.pairs) == scenarios[0].repetitions

    # Check each pair has 3 outcomes (Baseline, Deterministic, Jev)
    for pair in record.pairs:
        assert len(pair.outcomes) == 3
        systems_in_pair = {outcome.system for outcome in pair.outcomes}
        assert systems_in_pair == {
            MeasuredSystem.BASELINE,
            MeasuredSystem.ENHANCED_DETERMINISTIC,
            MeasuredSystem.ENHANCED_JEV,
        }

    # Verify that each system was executed across all repetitions
    assert executed_systems.count(MeasuredSystem.BASELINE) == scenarios[0].repetitions
    assert (
        executed_systems.count(MeasuredSystem.ENHANCED_DETERMINISTIC)
        == scenarios[0].repetitions
    )
    assert executed_systems.count(MeasuredSystem.ENHANCED_JEV) == scenarios[0].repetitions


@pytest.mark.anyio
async def test_three_way_orchestrator_failure_isolation(tmp_path: Path):
    store = RunStore(tmp_path / ".lab" / "runs")
    scenarios = [load_controlled_scenarios()[0]]

    async def resetter(*, system: MeasuredSystem, **kwargs):
        return _verification(system)

    async def baseline_runner(*, run_id, system, scheduled, scenario, **kwargs):
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=tuple(
                _make_result(
                    run_id=run_id,
                    system=system,
                    turn_id=uuid5(scheduled.pair_id, f"{system.value}:turn:{index}"),
                )
                for index, _turn in enumerate(scenario.turns)
            ),
        )

    async def deterministic_runner(*, run_id, system, scheduled, scenario, **kwargs):
        return SideRunOutput(
            status=OutcomeStatus.SUCCESS,
            model_id="openai/gpt-4.1-mini",
            results=tuple(
                _make_result(
                    run_id=run_id,
                    system=system,
                    turn_id=uuid5(scheduled.pair_id, f"{system.value}:turn:{index}"),
                )
                for index, _turn in enumerate(scenario.turns)
            ),
        )

    async def failing_jev_runner(**kwargs):
        return SideRunOutput(
            status=OutcomeStatus.FAILURE,
            reason="Jev provider simulated outage",
        )

    runners = {
        MeasuredSystem.BASELINE: baseline_runner,
        MeasuredSystem.ENHANCED_DETERMINISTIC: deterministic_runner,
        MeasuredSystem.ENHANCED_JEV: failing_jev_runner,
    }

    orch = PairedRunOrchestrator(
        store=store,
        scenarios=scenarios,
        resetter=resetter,
        runners=runners,
        execution_mode=ExecutionMode.OFFLINE_FAKE,
        snapshot_contracts={"standard-100": _snapshot_contract()},
        three_way=True,
    )

    request = StartRunRequest(scenario_ids=(scenarios[0].scenario_id,))
    handle = await orch.start(request)
    result = await orch.execute(handle.run_id)

    record = store.get(handle.run_id)
    assert record is not None
    # A failure in Jev arm does not crash or erase Baseline or Deterministic
    assert record.status == RunStatus.PARTIAL_FAILURE
    for pair in record.pairs:
        assert len(pair.outcomes) == 3
        b_out = next(o for o in pair.outcomes if o.system == MeasuredSystem.BASELINE)
        d_out = next(
            o
            for o in pair.outcomes
            if o.system == MeasuredSystem.ENHANCED_DETERMINISTIC
        )
        j_out = next(o for o in pair.outcomes if o.system == MeasuredSystem.ENHANCED_JEV)

        assert b_out.status == OutcomeStatus.SUCCESS
        assert d_out.status == OutcomeStatus.SUCCESS
        assert j_out.status == OutcomeStatus.FAILURE
        assert "outage" in (j_out.reason or "")
