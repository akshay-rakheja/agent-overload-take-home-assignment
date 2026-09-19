"""Metrics for routing quality, prompt size, and local latency."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RoutingObservation:
    case_id: str
    category: str
    split: str
    expected_action: str
    expected_agent_id: str | None
    action: str
    agent_id: str | None
    ranked_agent_ids: tuple[str, ...]
    candidate_count: int
    prompt_characters: int
    prompt_bytes: int
    latency_ms: float

    @property
    def correct(self) -> bool:
        if self.action != self.expected_action:
            return False
        if self.expected_action == "reuse":
            return self.agent_id == self.expected_agent_id
        return self.agent_id is None


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_routing(
    observations: list[RoutingObservation],
    *,
    top_k: int,
) -> dict[str, object]:
    """Summarize retrieval and final routing without conflating the two."""

    reuse = [item for item in observations if item.expected_action == "reuse"]
    top_k_hits = 0
    reciprocal_rank_total = 0.0
    for item in reuse:
        if item.expected_agent_id in item.ranked_agent_ids[:top_k]:
            top_k_hits += 1
        if item.expected_agent_id in item.ranked_agent_ids:
            reciprocal_rank_total += 1 / (item.ranked_agent_ids.index(item.expected_agent_id) + 1)

    correct = sum(item.correct for item in observations)
    unambiguous = [item for item in observations if item.expected_action != "abstain"]
    wrong_reuse = sum(
        item.action == "reuse"
        and (
            item.expected_action != "reuse"
            or item.agent_id != item.expected_agent_id
        )
        for item in unambiguous
    )
    duplicate_creation = sum(
        item.expected_action == "reuse" and item.action == "create_new"
        for item in observations
    )
    abstentions = [item for item in observations if item.action == "abstain"]
    correct_abstentions = sum(item.expected_action == "abstain" for item in abstentions)
    failures = [item for item in observations if not item.correct]
    failures_by_category = Counter(item.category for item in failures)
    latencies = [item.latency_ms for item in observations]

    return {
        "case_count": len(observations),
        "reuse_case_count": len(reuse),
        "top_k": top_k,
        "top_k_recall": _safe_rate(top_k_hits, len(reuse)),
        "mean_reciprocal_rank": _safe_rate(reciprocal_rank_total, len(reuse)),
        "decision_accuracy": _safe_rate(correct, len(observations)),
        "wrong_agent_reuse_rate": _safe_rate(wrong_reuse, len(unambiguous)),
        "duplicate_creation_rate": _safe_rate(duplicate_creation, len(reuse)),
        "abstention_precision": _safe_rate(correct_abstentions, len(abstentions)),
        "candidate_count_mean": _safe_rate(
            sum(item.candidate_count for item in observations), len(observations)
        ),
        "max_candidate_count": max((item.candidate_count for item in observations), default=0),
        "prompt_characters_mean": _safe_rate(
            sum(item.prompt_characters for item in observations), len(observations)
        ),
        "prompt_bytes_mean": _safe_rate(
            sum(item.prompt_bytes for item in observations), len(observations)
        ),
        "retrieval_latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "failures_by_category": dict(sorted(failures_by_category.items())),
        "failed_cases": [
            {
                "case_id": item.case_id,
                "category": item.category,
                "expected_action": item.expected_action,
                "expected_agent_id": item.expected_agent_id,
                "actual_action": item.action,
                "actual_agent_id": item.agent_id,
            }
            for item in failures
        ],
        "observations": [asdict(item) for item in observations],
    }

