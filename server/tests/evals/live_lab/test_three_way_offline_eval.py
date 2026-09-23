"""Unit tests for Three-Way Offline Evaluation and CLI Reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.live_lab.cli import main as cli_main
from evals.live_lab.offline_eval import run_offline_evaluation
from server.services.evaluation_lab.repetitions import MeasuredSystem, OutcomeStatus


def test_offline_evaluation_schema_v1_backwards_compatible() -> None:
    """Verifies that default three_way=False preserves 2-way evaluation."""
    result = run_offline_evaluation(
        scenario_ids=["exact-instagram-security"],
        repetitions=1,
        three_way=False,
    )
    assert result.status.value == "complete"
    assert len(result.pairs) == 1
    assert len(result.pairs[0].outcomes) == 2
    systems = {o.system for o in result.pairs[0].outcomes}
    assert MeasuredSystem.BASELINE in systems or "baseline" in systems
    assert MeasuredSystem.ENHANCED in systems or "enhanced" in systems


def test_offline_evaluation_three_way_execution() -> None:
    """Verifies that three_way=True executes Baseline, Deterministic, and Jev."""
    result = run_offline_evaluation(
        scenario_ids=["exact-instagram-security"],
        repetitions=1,
        three_way=True,
    )
    assert result.status.value == "complete"
    assert result.schema_version == 2
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert len(pair.outcomes) == 3

    outcomes_by_system = {o.system: o for o in pair.outcomes}
    assert MeasuredSystem.BASELINE in outcomes_by_system
    assert MeasuredSystem.ENHANCED_DETERMINISTIC in outcomes_by_system
    assert MeasuredSystem.ENHANCED_JEV in outcomes_by_system

    jev_outcome = outcomes_by_system[MeasuredSystem.ENHANCED_JEV]
    assert jev_outcome.status == OutcomeStatus.SUCCESS
    assert len(jev_outcome.results) > 0
    jev_result = jev_outcome.results[0]

    # Verify Jev trace fields are populated
    assert jev_result.jev_map_scores is not None
    assert jev_result.jev_shortlist is not None
    assert jev_result.jev_reduce_decision is not None
    assert jev_result.jev_total_latency_ms is not None
    assert jev_result.jev_winner_margin is not None

    # Verify scorecards for all 3 systems
    sc_systems = {sc.system for sc in result.scorecards}
    assert "baseline" in sc_systems
    assert "enhanced_deterministic" in sc_systems
    assert "enhanced_jev" in sc_systems


def test_cli_evaluate_and_verify_three_way(tmp_path: Path) -> None:
    """Verifies that CLI evaluate --three-way outputs valid reports and passes verify."""
    out_dir = tmp_path / "three_way_eval_run"
    rc = cli_main(
        [
            "evaluate",
            "--offline",
            "--three-way",
            "--scenario",
            "exact-instagram-security",
            "--repetitions",
            "1",
            "--output",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "run.json").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()

    # Verify report.md contains all 3 arms
    md_content = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Baseline" in md_content
    assert "Enhanced Deterministic" in md_content
    assert "Enhanced Jev" in md_content

    # Now verify artifacts consistency and absence of secrets
    verify_rc = cli_main(["verify", "--artifacts", str(out_dir)])
    assert verify_rc == 0
