"""Contract tests for the complete offline evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

from evals.runner import render_report, run_offline_evaluation, write_results
from evals.materialize import EVALUATION_NOW, materialize_case
from evals.schema import load_routing_corpus
from evals.strategies import HybridDirectoryStrategy
from server.agents.interaction_agent.agent import CandidateContext, prepare_message_with_history
from server.services.execution.retrieval import AgentRetriever, RetrievalQuery
from server.services.execution.routing import AgentRouter


CORPUS_PATH = Path(__file__).resolve().parents[3] / "evals" / "agent_routing_cases.jsonl"


def test_runner_compares_all_required_strategies_and_both_axes() -> None:
    recorded_command = "python -m evals.runner --json /tmp/result.json --report /tmp/report.md"
    results = run_offline_evaluation(CORPUS_PATH, evaluator_command=recorded_command)

    assert results["mode"] == "deterministic_offline"
    assert results["credentials_required"] is False
    assert set(results["breadth"]["strategies"]) == {
        "current_full_roster_exact_name_proxy",
        "recency_only_top_five",
        "hybrid_directory",
    }
    assert set(results["breadth"]["splits"]) == {"development", "test"}
    current_proxy = results["breadth"]["splits"]["test"][
        "current_full_roster_exact_name_proxy"
    ]
    assert current_proxy["retrieval_metrics_applicable"] is False
    assert current_proxy["top_k_recall"] is None
    assert current_proxy["mean_reciprocal_rank"] is None
    assert results["breadth"]["splits"]["test"]["hybrid_directory"]["max_candidate_count"] <= 5
    benchmark = results["breadth"]["scale_benchmarks"]["roster_1000"]
    assert benchmark["roster_size"] == 1_000
    assert benchmark["source"] == "AgentDirectory.list_records"
    assert benchmark["measured_runs"] >= 20
    assert benchmark["candidate_count_max"] <= 5
    assert benchmark["latency_ms"]["p95"] >= 0
    assert (
        results["breadth"]["test_partition_target_assessment"]["latency_source"]
        == "scale_benchmarks.roster_1000.latency_ms.p95"
    )
    assert "held_out_target_assessment" not in results["breadth"]
    assert results["corpus"]["sha256"] == results["corpus_revision"]
    assert set(results["corpus"]["partitions"]) == {"development", "test"}
    assert results["corpus"]["partitions"]["development"]["case_count"] == 20
    assert results["corpus"]["partitions"]["test"]["case_count"] == 20
    assert results["reproducibility"]["random_seed"] is None
    assert results["reproducibility"]["evaluator_command"] == recorded_command
    assert [point["history_entries"] for point in results["depth"]["scale"]] == [
        10,
        100,
        1_000,
        10_000,
    ]
    assert results["depth"]["scale"][-1]["bounded"]["prompt_characters"] <= 12_000
    assert results["depth"]["scale"][-1]["full_history"]["prompt_characters"] > 1_000_000
    assert all(point["raw_log_unchanged"] for point in results["depth"]["scale"])
    assert all(
        point["cross_agent_contamination_failures"] == 0
        for point in results["depth"]["scale"]
    )


def test_hybrid_strategy_uses_production_candidate_renderer() -> None:
    case = next(
        case
        for case in load_routing_corpus(CORPUS_PATH)
        if case.case_id == "test-paraphrase-recruiting"
    )
    materialized = materialize_case(case)
    query = RetrievalQuery(case.query, case.conversation_context)
    candidates = AgentRetriever(
        materialized.records,
        now=lambda: EVALUATION_NOW,
    ).retrieve(query)
    decision = AgentRouter().route(query, candidates)
    production_message = prepare_message_with_history(
        case.query,
        case.conversation_context,
        candidate_context=CandidateContext(tuple(candidates), decision),
    )[0]["content"]

    result = HybridDirectoryStrategy().run(materialized)

    assert result.prompt_text in production_message
    assert result.prompt_text.startswith("<agent_candidates ")


def test_results_write_as_json_and_honest_markdown(tmp_path) -> None:
    results = run_offline_evaluation(CORPUS_PATH)
    json_path = tmp_path / "hybrid_directory.json"
    report_path = tmp_path / "report.md"

    write_results(results, json_path=json_path, report_path=report_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["corpus_revision"]
    report = report_path.read_text(encoding="utf-8")
    assert report == render_report(results)
    assert "Checked-in test partition results" in report
    assert "| current_full_roster_exact_name_proxy | n/a | n/a |" in report
    assert "Exact evaluator command" in report
    assert "Failure analysis" in report
    assert "What this does not prove" in report
    assert "live model" in report.lower()
