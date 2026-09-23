"""Unit tests for three-way evaluation models, schedule, and schema-v1 compatibility."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from server.services.evaluation_lab.models import Availability, ObservedValue, SystemRunResult
from server.services.evaluation_lab.orchestrator import (
    PairExecutionRecord,
    PairedRunResult,
    RunStatus,
    SnapshotContract,
)
from server.services.evaluation_lab.repetitions import (
    MeasuredSystem,
    PairOrder,
    ScheduledPair,
    build_repetition_schedule,
)
from server.services.evaluation_lab.run_store import RunStore


def test_measured_system_supports_three_way_and_legacy():
    assert MeasuredSystem.BASELINE.value == "baseline"
    assert MeasuredSystem.ENHANCED_DETERMINISTIC.value == "enhanced_deterministic"
    assert MeasuredSystem.ENHANCED_JEV.value == "enhanced_jev"
    # Legacy alias
    assert MeasuredSystem("enhanced") in {
        MeasuredSystem.ENHANCED,
        MeasuredSystem.ENHANCED_DETERMINISTIC,
    }


def test_pair_order_three_way_systems():
    order1 = PairOrder.BASELINE_DETERMINISTIC_JEV
    assert order1.systems == (
        MeasuredSystem.BASELINE,
        MeasuredSystem.ENHANCED_DETERMINISTIC,
        MeasuredSystem.ENHANCED_JEV,
    )

    order2 = PairOrder.DETERMINISTIC_JEV_BASELINE
    assert order2.systems == (
        MeasuredSystem.ENHANCED_DETERMINISTIC,
        MeasuredSystem.ENHANCED_JEV,
        MeasuredSystem.BASELINE,
    )


def test_build_repetition_schedule_three_way():
    schedule = build_repetition_schedule(
        ["exact-instagram-security"], repetitions=3, three_way=True
    )
    assert len(schedule) == 3
    assert schedule[0].order == PairOrder.BASELINE_DETERMINISTIC_JEV
    assert schedule[1].order == PairOrder.DETERMINISTIC_JEV_BASELINE
    assert schedule[2].order == PairOrder.JEV_BASELINE_DETERMINISTIC


def test_system_run_result_jev_trace_fields():
    run_id = uuid4()
    turn_id = uuid4()

    res = SystemRunResult(
        run_id=run_id,
        turn_id=turn_id,
        system="enhanced_jev",
        jev_winner_margin=ObservedValue[float](
            availability=Availability.AVAILABLE, value=0.45
        ),
        jev_map_latency_ms=ObservedValue[float](
            availability=Availability.AVAILABLE, value=12.5
        ),
        jev_api_calls_count=ObservedValue[int](
            availability=Availability.AVAILABLE, value=4
        ),
    )

    assert res.system == "enhanced_jev"
    assert res.jev_winner_margin.value == 0.45
    assert res.jev_map_latency_ms.value == 12.5
    assert res.jev_api_calls_count.value == 4
    # Default unavailable
    assert res.jev_reduce_latency_ms.availability == Availability.UNAVAILABLE


def test_existing_schema_v1_artifact_backwards_compatibility():
    runs_dir = Path(".lab/runs")
    if not runs_dir.exists():
        pytest.skip(".lab/runs does not exist")

    store = RunStore(runs_dir)
    records = store.list_records()
    assert len(records) > 0

    for rec in records:
        assert rec.run_id is not None
        assert rec.schema_version in (1, 2)
        assert len(rec.pairs) > 0
