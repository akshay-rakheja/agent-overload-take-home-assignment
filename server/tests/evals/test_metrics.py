"""Hand-calculated tests for routing and latency metrics."""

from __future__ import annotations

import pytest

from evals.metrics import RoutingObservation, percentile, summarize_routing


def observation(
    case_id: str,
    *,
    expected_action: str,
    expected_agent_id: str | None,
    action: str,
    agent_id: str | None,
    ranked: tuple[str, ...],
    latency_ms: float,
) -> RoutingObservation:
    return RoutingObservation(
        case_id=case_id,
        category="hand_calculated",
        split="test",
        expected_action=expected_action,
        expected_agent_id=expected_agent_id,
        action=action,
        agent_id=agent_id,
        ranked_agent_ids=ranked,
        candidate_count=len(ranked),
        prompt_characters=100,
        prompt_bytes=100,
        latency_ms=latency_ms,
    )


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile(values, 0.95) == pytest.approx(3.85)


def test_routing_metrics_match_hand_calculated_example() -> None:
    observations = [
        observation(
            "correct-reuse",
            expected_action="reuse",
            expected_agent_id="a",
            action="reuse",
            agent_id="a",
            ranked=("a",),
            latency_ms=1,
        ),
        observation(
            "duplicate-create",
            expected_action="reuse",
            expected_agent_id="b",
            action="create_new",
            agent_id=None,
            ranked=("x", "b"),
            latency_ms=2,
        ),
        observation(
            "correct-abstain",
            expected_action="abstain",
            expected_agent_id=None,
            action="abstain",
            agent_id=None,
            ranked=("c", "d"),
            latency_ms=3,
        ),
        observation(
            "wrong-reuse",
            expected_action="create_new",
            expected_agent_id=None,
            action="reuse",
            agent_id="z",
            ranked=("z",),
            latency_ms=4,
        ),
    ]

    metrics = summarize_routing(observations, top_k=5)

    assert metrics["top_k_recall"] == pytest.approx(1.0)
    assert metrics["mean_reciprocal_rank"] == pytest.approx(0.75)
    assert metrics["decision_accuracy"] == pytest.approx(0.5)
    assert metrics["wrong_agent_reuse_rate"] == pytest.approx(1 / 3)
    assert metrics["duplicate_creation_rate"] == pytest.approx(0.5)
    assert metrics["abstention_precision"] == pytest.approx(1.0)
    assert metrics["candidate_count_mean"] == pytest.approx(1.5)
    assert metrics["retrieval_latency_ms"]["p50"] == pytest.approx(2.5)
    assert metrics["retrieval_latency_ms"]["p95"] == pytest.approx(3.85)
    assert metrics["failures_by_category"] == {"hand_calculated": 2}
