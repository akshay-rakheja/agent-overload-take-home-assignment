"""Run the credential-free two-axis OpenPoke agent-overload evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from evals.baseline import HISTORY_SIZES, render_full_history
from evals.fixtures import make_history, make_roster
from evals.materialize import EVALUATION_NOW, fixture_to_record, materialize_case
from evals.metrics import percentile, summarize_routing
from evals.schema import load_routing_corpus
from evals.strategies import default_breadth_strategies
from server.config import Settings
from server.services.execution.directory import AgentDirectory
from server.services.execution.context_policy import ExecutionContextPolicy
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.retrieval import AgentRetriever, RetrievalQuery
from server.services.execution.routing import AgentRouter, RoutingAction


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_depth(settings: Settings) -> dict[str, Any]:
    policy = ExecutionContextPolicy(
        max_recent_episodes=settings.execution_context_max_recent_episodes,
        max_characters=settings.execution_context_max_characters,
    )
    points: list[dict[str, Any]] = []
    bounded_latencies: list[float] = []
    full_latencies: list[float] = []

    with tempfile.TemporaryDirectory(prefix="openpoke-depth-eval-") as temporary_dir:
        base_dir = Path(temporary_dir)
        for size in HISTORY_SIZES:
            log_store = ExecutionAgentLogStore(base_dir / str(size))
            primary_key = "primary-agent"
            other_key = "other-agent"
            for tag, _timestamp, payload in make_history(size):
                if tag == "agent_request":
                    log_store.record_request(primary_key, payload)
                elif tag == "agent_action":
                    log_store.record_action(primary_key, payload)
                elif tag == "tool_response":
                    log_store.record_tool_response(primary_key, "fixture", payload)
                else:
                    log_store.record_agent_response(primary_key, payload)
            log_store.record_request(other_key, "SECOND-AGENT-SENTINEL")

            primary_before = log_store.read_raw_bytes(primary_key)
            other_before = log_store.read_raw_bytes(other_key)
            entries = list(log_store.iter_entries(primary_key))

            started = perf_counter()
            full_text = render_full_history(entries)
            full_latency = (perf_counter() - started) * 1_000
            full_latencies.append(full_latency)

            started = perf_counter()
            bounded = policy.render(entries, memory_summary="Durable synthetic relationship summary.")
            bounded_latency = (perf_counter() - started) * 1_000
            bounded_latencies.append(bounded_latency)

            points.append(
                {
                    "history_entries": size,
                    "raw_history_characters": len(full_text),
                    "raw_history_bytes": len(full_text.encode("utf-8")),
                    "full_history": {
                        "prompt_characters": len(full_text),
                        "prompt_bytes": len(full_text.encode("utf-8")),
                        "included_entry_count": len(entries),
                        "latency_ms": full_latency,
                    },
                    "bounded": {
                        "prompt_characters": bounded.metrics.rendered_characters,
                        "prompt_bytes": bounded.metrics.rendered_bytes,
                        "included_episode_count": bounded.metrics.included_episode_count,
                        "omitted_entry_count": bounded.metrics.omitted_entry_count,
                        "truncated_entry_count": bounded.metrics.truncated_entry_count,
                        "summary_used": bounded.metrics.summary_used,
                        "latency_ms": bounded_latency,
                    },
                    "raw_log_unchanged": (
                        primary_before == log_store.read_raw_bytes(primary_key)
                        and other_before == log_store.read_raw_bytes(other_key)
                    ),
                    "cross_agent_contamination_failures": int(
                        "SECOND-AGENT-SENTINEL" in bounded.text
                    ),
                }
            )

    return {
        "configuration": {
            "max_recent_episodes": settings.execution_context_max_recent_episodes,
            "max_characters": settings.execution_context_max_characters,
        },
        "scale": points,
        "full_history_latency_ms": {
            "p50": percentile(full_latencies, 0.50),
            "p95": percentile(full_latencies, 0.95),
        },
        "bounded_context_latency_ms": {
            "p50": percentile(bounded_latencies, 0.50),
            "p95": percentile(bounded_latencies, 0.95),
        },
    }


def _benchmark_roster_1000(settings: Settings) -> dict[str, Any]:
    fixtures = make_roster(1_000)
    records = tuple(fixture_to_record(fixture) for fixture in fixtures)
    expected_id = records[942].agent_id
    query = RetrievalQuery("Find account-00942 follow-ups", "")
    router = AgentRouter(settings=settings)
    warmup_runs = 3
    measured_runs = 30

    with tempfile.TemporaryDirectory(prefix="openpoke-breadth-eval-") as temporary_dir:
        directory_path = Path(temporary_dir) / "roster.json"
        directory_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "agents": [record.model_dump(mode="json") for record in records],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        directory = AgentDirectory(directory_path)

        def run_once() -> tuple[float, int, int, bool]:
            started = perf_counter()
            retriever = AgentRetriever(
                directory.list_records,
                now=lambda: EVALUATION_NOW,
                settings=settings,
            )
            candidates = retriever.retrieve(query)
            decision = router.route(query, candidates)
            prompt = "\n".join(
                f"{candidate.agent_id}|{candidate.name}|{candidate.purpose}|{candidate.status.value}"
                for candidate in candidates
            )
            elapsed_ms = (perf_counter() - started) * 1_000
            correct = decision.action is RoutingAction.REUSE and decision.agent_id == expected_id
            return elapsed_ms, len(candidates), len(prompt), correct

        for _ in range(warmup_runs):
            run_once()
        observations = [run_once() for _ in range(measured_runs)]
    latencies = [observation[0] for observation in observations]
    return {
        "roster_size": len(records),
        "source": "AgentDirectory.list_records",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "candidate_count_max": max(observation[1] for observation in observations),
        "candidate_count": {
            "min": min(observation[1] for observation in observations),
            "max": max(observation[1] for observation in observations),
        },
        "prompt_characters": {
            "min": min(observation[2] for observation in observations),
            "max": max(observation[2] for observation in observations),
        },
        "correct_reuse_rate": sum(observation[3] for observation in observations) / measured_runs,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }


def _target_assessment(
    test_metrics: dict[str, Any],
    scale_benchmark: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "top_5_recall_at_least_95_percent": test_metrics["top_k_recall"] >= 0.95,
        "decision_accuracy_at_least_90_percent": test_metrics["decision_accuracy"] >= 0.90,
        "wrong_agent_reuse_at_most_2_percent": test_metrics["wrong_agent_reuse_rate"] <= 0.02,
        "candidate_count_at_most_5": test_metrics["max_candidate_count"] <= 5,
        "retrieval_p95_below_50_ms": scale_benchmark["latency_ms"]["p95"] < 50,
    }
    return {
        "checks": checks,
        "all_targets_met": all(checks.values()),
        "missed_targets": [name for name, passed in checks.items() if not passed],
        "latency_source": "scale_benchmarks.roster_1000.latency_ms.p95",
    }


def run_offline_evaluation(corpus_path: Path) -> dict[str, Any]:
    """Evaluate both overload axes without network calls or provider credentials."""

    cases = load_routing_corpus(corpus_path)
    materialized = [materialize_case(case) for case in cases]
    strategies = default_breadth_strategies()
    split_results: dict[str, dict[str, Any]] = {}

    for split in ("development", "test"):
        split_cases = [item for item in materialized if item.case.split == split]
        split_results[split] = {}
        for strategy in strategies:
            observations = [strategy.observe(item) for item in split_cases]
            split_results[split][strategy.name] = summarize_routing(observations, top_k=5)

    settings = Settings()
    hybrid_test = split_results["test"]["hybrid_directory"]
    roster_1000_benchmark = _benchmark_roster_1000(settings)
    return {
        "schema_version": 1,
        "mode": "deterministic_offline",
        "credentials_required": False,
        "implementation_commit": _git_commit(),
        "corpus_revision": _sha256(corpus_path),
        "corpus_case_count": len(cases),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "configuration": {
            "retrieval_top_k": settings.agent_retrieval_top_k,
            "retrieval_min_score": settings.agent_retrieval_min_score,
            "reuse_threshold": settings.agent_route_reuse_threshold,
            "ambiguity_margin": settings.agent_route_ambiguity_margin,
            "weights": {
                "exact_match": 0.70,
                "token_overlap_max": 0.35,
                "phrase_match": 0.12,
                "recency_max": 0.05,
                "prior_use_max": 0.04,
                "dormant_penalty": -0.03,
                "archived_penalty": -0.08,
            },
        },
        "breadth": {
            "strategies": [strategy.name for strategy in strategies],
            "splits": split_results,
            "scale_benchmarks": {"roster_1000": roster_1000_benchmark},
            "held_out_target_assessment": _target_assessment(
                hybrid_test, roster_1000_benchmark
            ),
        },
        "depth": _evaluate_depth(settings),
        "limitations": [
            "The current full-roster result is an exact-name proxy because offline code cannot reproduce an unpinned live model's choice.",
            "The harness measures deterministic routing and prompt construction, not end-to-end Gmail task success.",
            "Local latency measurements vary by machine and process load.",
            "No LLM judge is used as identity-routing ground truth.",
        ],
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(results: dict[str, Any]) -> str:
    """Render a reviewer-oriented report including targets and honest failures."""

    breadth = results["breadth"]
    depth = results["depth"]
    lines = [
        "# Agent overload evaluation",
        "",
        f"- Implementation commit: `{results['implementation_commit']}`",
        f"- Corpus revision: `{results['corpus_revision']}`",
        f"- Corpus cases: {results['corpus_case_count']} (20 development, 20 held-out)",
        "- Mode: deterministic offline; no credentials required",
        "",
        "## Held-out test results",
        "",
        "| Strategy | Top-5 recall | MRR | Decision accuracy | Wrong reuse | Duplicate creation | Max candidates | Prompt chars mean | Case-mix p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in breadth["strategies"]:
        metrics = breadth["splits"]["test"][strategy]
        lines.append(
            f"| {strategy} | {_percent(metrics['top_k_recall'])} | "
            f"{metrics['mean_reciprocal_rank']:.3f} | {_percent(metrics['decision_accuracy'])} | "
            f"{_percent(metrics['wrong_agent_reuse_rate'])} | "
            f"{_percent(metrics['duplicate_creation_rate'])} | {metrics['max_candidate_count']} | "
            f"{metrics['prompt_characters_mean']:.1f} | "
            f"{metrics['retrieval_latency_ms']['p95']:.3f} |"
        )

    benchmark = breadth["scale_benchmarks"]["roster_1000"]
    lines.extend(
        [
            "",
            "## Dedicated 1,000-record latency benchmark",
            "",
            f"Using `{benchmark['source']}` after {benchmark['warmup_runs']} warm-up runs, "
            f"{benchmark['measured_runs']} measured runs produced p50 "
            f"{benchmark['latency_ms']['p50']:.3f} ms and p95 "
            f"{benchmark['latency_ms']['p95']:.3f} ms. Candidate count remained "
            f"between {benchmark['candidate_count']['min']} and "
            f"{benchmark['candidate_count']['max']}.",
        ]
    )

    assessment = breadth["held_out_target_assessment"]
    lines.extend(["", "## Held-out target assessment", ""])
    for name, passed in assessment["checks"].items():
        lines.append(f"- {'PASS' if passed else 'MISS'} — {name.replace('_', ' ')}")

    lines.extend(
        [
            "",
            "## History-depth scaling",
            "",
            "| Raw entries | Full prompt chars | Bounded prompt chars | Included episodes | Omitted entries | Raw log unchanged |",
            "| ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for point in depth["scale"]:
        lines.append(
            f"| {point['history_entries']} | {point['full_history']['prompt_characters']} | "
            f"{point['bounded']['prompt_characters']} | "
            f"{point['bounded']['included_episode_count']} | "
            f"{point['bounded']['omitted_entry_count']} | "
            f"{'yes' if point['raw_log_unchanged'] else 'no'} |"
        )

    all_failures: list[tuple[str, dict[str, Any]]] = []
    for strategy in breadth["strategies"]:
        for failure in breadth["splits"]["test"][strategy]["failed_cases"]:
            all_failures.append((strategy, failure))
    routing_miss = all_failures[0] if all_failures else None
    abstention_error = next(
        (
            item
            for item in all_failures
            if item[1]["expected_action"] == "abstain"
            or item[1]["actual_action"] == "abstain"
        ),
        None,
    )

    lines.extend(["", "## Failure analysis", ""])
    if routing_miss:
        strategy, failure = routing_miss
        lines.append(
            f"- Routing miss: `{strategy}` on `{failure['case_id']}` expected "
            f"`{failure['expected_action']}` but produced `{failure['actual_action']}`."
        )
    if abstention_error:
        strategy, failure = abstention_error
        lines.append(
            f"- Abstention error: `{strategy}` on `{failure['case_id']}` expected "
            f"`{failure['expected_action']}` but produced `{failure['actual_action']}`."
        )
    hybrid_failures = breadth["splits"]["test"]["hybrid_directory"]["failed_cases"]
    if hybrid_failures:
        for failure in hybrid_failures:
            lines.append(
                f"- Hybrid miss: `{failure['case_id']}` ({failure['category']}) expected "
                f"`{failure['expected_action']}` but produced `{failure['actual_action']}`."
            )
    else:
        lines.append("- Hybrid directory had no held-out routing misses in this corpus revision.")
    if assessment["missed_targets"]:
        lines.append(f"- Missed targets: {', '.join(assessment['missed_targets'])}.")
    else:
        lines.append("- No proposed held-out target was missed on this corpus revision.")

    lines.extend(
        [
            "",
            "## What this does not prove",
            "",
            "This offline harness does not prove live model behavior, Gmail task success, or production latency. "
            "It isolates deterministic identity routing, prompt growth, context bounding, and local execution cost. "
            "A live model evaluation must pin the provider/model/configuration and report repeated-run variance.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(
    results: dict[str, Any],
    *,
    json_path: Path,
    report_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evals/agent_routing_cases.jsonl"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("evals/results/hybrid_directory.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/results/report.md"),
    )
    args = parser.parse_args()
    results = run_offline_evaluation(args.corpus)
    write_results(results, json_path=args.json, report_path=args.report)
    print(f"Wrote {args.json} and {args.report}")


if __name__ == "__main__":
    main()
