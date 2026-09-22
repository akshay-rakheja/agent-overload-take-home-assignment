"""Tests for cross-artifact consistency verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.live_lab.consistency import (
    ConsistencyReport,
    verify_artifacts_consistency,
)
from evals.live_lab.report import render_paired_run_json, render_paired_run_markdown
from server.tests.evals.live_lab.test_report import _make_dummy_run_result


def test_consistency_passes_on_matching_artifacts(tmp_path: Path) -> None:
    run = _make_dummy_run_result()
    json_str = render_paired_run_json(run)
    md_str = render_paired_run_markdown(run)

    json_file = tmp_path / "report.json"
    md_file = tmp_path / "report.md"
    json_file.write_text(json_str, encoding="utf-8")
    md_file.write_text(md_str, encoding="utf-8")

    report = verify_artifacts_consistency(
        run=run,
        artifacts={
            "report_json": json_file,
            "report_markdown": md_file,
        },
    )
    assert report.valid
    assert len(report.mismatches) == 0
    assert report.run_id == str(run.run_id)


def test_consistency_fails_on_run_id_mismatch(tmp_path: Path) -> None:
    run = _make_dummy_run_result()
    bad_json = tmp_path / "bad.json"
    data = json.loads(render_paired_run_json(run))
    data["run_id"] = "99999999-9999-4999-8999-999999999999"
    bad_json.write_text(json.dumps(data), encoding="utf-8")

    report = verify_artifacts_consistency(
        run=run,
        artifacts={"bad_json": bad_json},
    )
    assert not report.valid
    assert any("run_id mismatch" in err.lower() for err in report.mismatches)


def test_consistency_fails_on_candidate_count_exceeded(tmp_path: Path) -> None:
    run = _make_dummy_run_result()
    bad_json = tmp_path / "bad_candidates.json"
    data = json.loads(render_paired_run_json(run))
    # inject 6 candidates
    data["pairs"][0]["outcomes"][1]["results"][0]["candidates"]["value"] = [
        {"id": f"agent-{i}", "name": f"Agent {i}"} for i in range(6)
    ]
    bad_json.write_text(json.dumps(data), encoding="utf-8")

    report = verify_artifacts_consistency(
        run=run,
        artifacts={"bad_candidates": bad_json},
    )
    assert not report.valid
    assert any("candidate" in err.lower() for err in report.mismatches)
