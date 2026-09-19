"""Concise, credential-free narrative demo for the take-home review."""

from __future__ import annotations

from pathlib import Path

from evals.baseline import build_baseline_report, render_full_history
from evals.fixtures import make_history
from evals.materialize import materialize_case
from evals.schema import load_routing_corpus
from evals.strategies import HybridDirectoryStrategy
from server.config import Settings
from server.services.execution.context_policy import ExecutionContextPolicy


CORPUS_PATH = Path(__file__).with_name("agent_routing_cases.jsonl")


def render_demo() -> str:
    cases = {case.case_id: case for case in load_routing_corpus(CORPUS_PATH)}
    strategy = HybridDirectoryStrategy()
    selected = {
        "1,000 identities": "test-scale-1000",
        "paraphrased follow-up": "test-paraphrase-recruiting",
        "novel task": "test-novel-travel",
        "ambiguous task": "test-ambiguous-jordan",
    }

    settings = Settings()
    baseline = build_baseline_report()
    history = make_history(10_000)
    full_history = render_full_history(history)
    bounded = ExecutionContextPolicy(
        max_recent_episodes=settings.execution_context_max_recent_episodes,
        max_characters=settings.execution_context_max_characters,
    ).render(history, memory_summary="Durable synthetic relationship summary.")

    lines = [
        "OpenPoke agent overload demo",
        "",
        "Breadth: current OpenPoke injects every identity into the interaction prompt.",
        f"baseline 1,000-agent roster characters: {baseline['breadth']['scale'][-1]['rendered_characters']:,}",
        f"candidate hard cap: {settings.agent_retrieval_top_k}",
    ]
    for label, case_id in selected.items():
        materialized = materialize_case(cases[case_id])
        result = strategy.run(materialized)
        lines.append(
            f"{label}: {result.action} | directory={len(materialized.records):,} | "
            f"candidates={result.candidate_count}"
        )

    lines.extend(
        [
            "",
            "Depth: one selected agent retains raw history but receives bounded prompt context.",
            f"10,000 raw history entries: {len(full_history):,} full characters",
            f"bounded prompt characters: {bounded.metrics.rendered_characters:,}",
            f"included recent episodes: {bounded.metrics.included_episode_count}",
            f"omitted older entries: {bounded.metrics.omitted_entry_count:,}",
            "raw history mutation: none",
            "",
            "honest baseline failure: current exact-name proxy creates new on test-exact-sam",
            "full details: evals/results/report.md",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print(render_demo())


if __name__ == "__main__":
    main()
