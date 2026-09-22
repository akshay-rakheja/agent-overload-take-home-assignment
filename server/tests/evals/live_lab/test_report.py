"""Tests for deterministic JSON and Markdown paired evaluation reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from evals.live_lab.report import (
    render_paired_run_json,
    render_paired_run_markdown,
    write_reports,
)
from server.services.evaluation_lab.models import (
    Availability,
    ObservedValue,
    SystemRunResult,
    UsagePlaceholder,
)
from server.services.evaluation_lab.orchestrator import (
    ExecutionMode,
    PairedRunResult,
    PairedSequenceScorecard,
    PairExecutionRecord,
    PersistedSideOutcome,
    RunStatus,
    ScheduledPair,
    StartRunRequest,
)


def _make_dummy_run_result() -> PairedRunResult:
    run_id = UUID("11111111-1111-4111-8111-111111111111")
    pair_id = UUID("22222222-2222-4222-8222-222222222222")
    now = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc)

    baseline_result = SystemRunResult(
        run_id=run_id,
        turn_id=UUID("33333333-3333-4333-8333-333333333333"),
        system="baseline",
        mode=ObservedValue(availability=Availability.INFERRED, value="full_roster"),
        roster_count=ObservedValue(availability=Availability.AVAILABLE, value=100),
        prompt_exposure=ObservedValue(
            availability=Availability.AVAILABLE,
            value={"exposed_name_count": 100, "prompt_characters": 18000},
        ),
        final_response=ObservedValue(availability=Availability.AVAILABLE, value="Baseline answered Dr. Smith"),
        usage=UsagePlaceholder(
            input_tokens=ObservedValue(availability=Availability.AVAILABLE, value=4000),
            output_tokens=ObservedValue(availability=Availability.AVAILABLE, value=150),
            total_tokens=ObservedValue(availability=Availability.AVAILABLE, value=4150),
        ),
    )

    enhanced_result = SystemRunResult(
        run_id=run_id,
        turn_id=UUID("44444444-4444-4444-8444-444444444444"),
        system="enhanced",
        mode=ObservedValue(availability=Availability.AVAILABLE, value="top_k"),
        roster_count=ObservedValue(availability=Availability.AVAILABLE, value=100),
        candidates=ObservedValue(
            availability=Availability.AVAILABLE,
            value=[{"id": "agent-001", "name": "Appointment Assistant"}],
        ),
        final_response=ObservedValue(availability=Availability.AVAILABLE, value="Enhanced answered Dr. Smith"),
        usage=UsagePlaceholder(
            input_tokens=ObservedValue(availability=Availability.AVAILABLE, value=850),
            output_tokens=ObservedValue(availability=Availability.AVAILABLE, value=120),
            total_tokens=ObservedValue(availability=Availability.AVAILABLE, value=970),
        ),
    )

    from evals.live_lab.grading import GradeStatus, LayerGrade

    scorecard = PairedSequenceScorecard(
        scenario_id="scenario-controlled-dr-appointment",
        system="enhanced",
        pair_id=pair_id,
        repetition=1,
        turns=(),
        identity_continuity=LayerGrade(layer="identity", status=GradeStatus.PASS),
    )

    from server.services.evaluation_lab.repetitions import OutcomeStatus, PairOrder

    return PairedRunResult(
        run_id=run_id,
        request=StartRunRequest(
            scenario_ids=("scenario-controlled-dr-appointment",),
        ),
        execution_mode=ExecutionMode.MEASURED,
        status=RunStatus.COMPLETE,
        generation=1,
        schedule=(
            ScheduledPair(
                pair_id=pair_id,
                scenario_id="scenario-controlled-dr-appointment",
                repetition=1,
                order=PairOrder.BASELINE_THEN_ENHANCED,
            ),
        ),
        pairs=(
            PairExecutionRecord(
                scheduled=ScheduledPair(
                    pair_id=pair_id,
                    scenario_id="scenario-controlled-dr-appointment",
                    repetition=1,
                    order=PairOrder.BASELINE_THEN_ENHANCED,
                ),
                outcomes=(
                    PersistedSideOutcome(
                        system="baseline",
                        status=OutcomeStatus.SUCCESS,
                        model_id="openai/gpt-4.1-mini",
                        results=(baseline_result,),
                    ),
                    PersistedSideOutcome(
                        system="enhanced",
                        status=OutcomeStatus.SUCCESS,
                        model_id="openai/gpt-4.1-mini",
                        results=(enhanced_result,),
                    ),
                ),
            ),
        ),
        scorecards=(scorecard,),
        transitions=(),
        trace=(),
        created_at=now,
        updated_at=now,
    )


def test_render_paired_run_json_deterministic() -> None:
    run = _make_dummy_run_result()
    out1 = render_paired_run_json(run)
    out2 = render_paired_run_json(run)
    assert out1 == out2
    parsed = json.loads(out1)
    assert parsed["run_id"] == "11111111-1111-4111-8111-111111111111"
    assert parsed["status"] == "complete"
    assert len(parsed["pairs"]) == 1


def test_render_paired_run_markdown_content() -> None:
    run = _make_dummy_run_result()
    md = render_paired_run_markdown(run)
    assert "# Evaluation Lab Paired Run Report" in md
    assert "11111111-1111-4111-8111-111111111111" in md
    assert "scenario-controlled-dr-appointment" in md
    assert "openai/gpt-4.1-mini" in md
    assert "Baseline" in md
    assert "Enhanced" in md
    assert "4,000" in md or "4000" in md
    assert "850" in md


def test_write_reports(tmp_path: Path) -> None:
    run = _make_dummy_run_result()
    json_path, md_path = write_reports(run, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert md_path.read_text(encoding="utf-8").startswith("#")
