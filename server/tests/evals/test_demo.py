"""Smoke test for the reviewer-facing five-minute demo."""

from __future__ import annotations

from evals.demo import render_demo


def test_demo_covers_breadth_depth_and_safe_decisions() -> None:
    output = render_demo()

    assert "1,000 identities" in output
    assert "paraphrased follow-up: reuse" in output
    assert "novel task: create_new" in output
    assert "ambiguous task: abstain" in output
    assert "candidate hard cap: 5" in output
    assert "10,000 raw history entries" in output
    assert "1,322,499" in output
    assert "4,376" in output
    assert "raw history mutation: none" in output
    assert "honest baseline failure" in output
    assert "depth evidence replay source: evals/results/hybrid_directory.json" in output
    assert "evidence evaluated commit:" in output
