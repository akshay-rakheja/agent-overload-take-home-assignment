"""Behavioral tests for measurements of the unchanged OpenPoke baseline."""

from __future__ import annotations

import json

from evals.baseline import (
    HISTORY_SIZES,
    ROSTER_SIZES,
    build_baseline_report,
    render_baseline_markdown,
    write_baseline_report,
)
from server.tests.fixtures.history_factory import make_history


def test_history_fixture_is_deterministic_at_deep_scale() -> None:
    first = make_history(10_000)
    second = make_history(10_000)

    assert first == second
    assert len(first) == 10_000
    assert first[0][0] == "agent_request"
    assert first[-1][2]


def test_baseline_records_breadth_and_depth_growth() -> None:
    report = build_baseline_report()

    assert [point["roster_size"] for point in report["breadth"]["scale"]] == list(ROSTER_SIZES)
    assert [point["history_entries"] for point in report["depth"]["scale"]] == list(HISTORY_SIZES)
    assert report["breadth"]["agents_injected_at_1000"] == 1_000
    assert report["depth"]["history_is_unbounded_by_default"] is True
    assert report["breadth"]["scale"][-1]["rendered_characters"] > report["breadth"]["scale"][0]["rendered_characters"]
    assert report["depth"]["scale"][-1]["rendered_characters"] > report["depth"]["scale"][0]["rendered_characters"]


def test_baseline_exposes_exact_name_and_variant_behavior() -> None:
    report = build_baseline_report()

    assert report["breadth"]["name_match_examples"] == [
        {"requested": "Alice correspondence", "reused": True},
        {"requested": "alice correspondence", "reused": False},
        {"requested": "Alice-correspondence", "reused": False},
    ]


def test_baseline_can_be_written_as_json_and_markdown(tmp_path) -> None:
    report = build_baseline_report()
    json_path = tmp_path / "baseline.json"
    markdown_path = tmp_path / "baseline.md"

    write_baseline_report(report, json_path=json_path, markdown_path=markdown_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_baseline_markdown(report)
    assert "Roster breadth" in markdown
    assert "History depth" in markdown
