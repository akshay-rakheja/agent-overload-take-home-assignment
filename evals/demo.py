"""Concise, credential-free narrative demo for the take-home review."""

from __future__ import annotations

import json
from pathlib import Path

from evals.baseline import build_baseline_report
from evals.materialize import materialize_case
from evals.schema import load_routing_corpus
from evals.strategies import HybridDirectoryStrategy
from server.config import MAX_AGENT_CANDIDATES


CORPUS_PATH = Path(__file__).with_name("agent_routing_cases.jsonl")
RESULTS_PATH = Path(__file__).with_name("results") / "hybrid_directory.json"


def render_demo() -> str:
    cases = {case.case_id: case for case in load_routing_corpus(CORPUS_PATH)}
    strategy = HybridDirectoryStrategy()
    selected = {
        "1,000 identities": "test-scale-1000",
        "paraphrased follow-up": "test-paraphrase-recruiting",
        "novel task": "test-novel-travel",
        "ambiguous task": "test-ambiguous-jordan",
    }

    baseline = build_baseline_report()
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    depth_point = results["depth"]["scale"][-1]
    source_path = RESULTS_PATH.relative_to(Path(__file__).resolve().parent.parent).as_posix()

    lines = [
        "OpenPoke agent overload demo",
        "",
        "Breadth: current OpenPoke injects every identity into the interaction prompt.",
        f"baseline 1,000-agent roster characters: {baseline['breadth']['scale'][-1]['rendered_characters']:,}",
        f"candidate hard cap: {MAX_AGENT_CANDIDATES}",
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
            f"depth evidence replay source: {source_path}",
            f"evidence evaluated commit: {results['implementation_commit']}",
            f"10,000 raw history entries: {depth_point['full_history']['prompt_characters']:,} full characters",
            f"bounded prompt characters: {depth_point['bounded']['prompt_characters']:,}",
            f"included recent episodes: {depth_point['bounded']['included_episode_count']}",
            f"omitted older entries: {depth_point['bounded']['omitted_entry_count']:,}",
            "raw history mutation: none"
            if depth_point["raw_log_unchanged"]
            else "raw history mutation: detected",
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
