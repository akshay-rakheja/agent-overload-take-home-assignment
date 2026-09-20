"""Measure the unchanged OpenPoke full-roster and full-history behavior."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Iterable

from evals.fixtures import HistoryEntry, make_history, make_roster


ROSTER_SIZES = (10, 100, 500, 1_000)
HISTORY_SIZES = (10, 100, 1_000, 10_000)


def _measure(rendered: str) -> dict[str, int]:
    return {
        "rendered_characters": len(rendered),
        "rendered_bytes": len(rendered.encode("utf-8")),
    }


def render_full_roster(names: Iterable[str]) -> str:
    """Mirror the current interaction-agent roster representation."""

    return "\n".join(
        f'<agent name="{escape(name or "agent", quote=True)}" />'
        for name in names
    )


def render_full_history(entries: Iterable[HistoryEntry]) -> str:
    """Mirror the current log transcript included in an execution prompt."""

    parts: list[str] = []
    for tag, timestamp, payload in entries:
        parts.append(f'<{tag} timestamp="{timestamp}">{escape(payload, quote=False)}</{tag}>')
    return "\n".join(parts)


def build_baseline_report() -> dict[str, object]:
    """Return directly measurable properties of the current implementation."""

    breadth_scale: list[dict[str, int]] = []
    for size in ROSTER_SIZES:
        roster = make_roster(size)
        rendered = render_full_roster(agent.name for agent in roster)
        breadth_scale.append({"roster_size": size, "agents_injected": size, **_measure(rendered)})

    depth_scale: list[dict[str, int]] = []
    for size in HISTORY_SIZES:
        rendered = render_full_history(make_history(size))
        depth_scale.append({"history_entries": size, "entries_loaded": size, **_measure(rendered)})

    return {
        "schema_version": 1,
        "measurement_scope": "deterministic_offline_current_behavior",
        "breadth": {
            "implementation": "full_roster_prompt_injection",
            "agents_injected_at_1000": 1_000,
            "reuse_match": "exact_case_sensitive_name",
            "naming_variants_can_create_duplicates": True,
            "name_match_examples": [
                {"requested": requested, "reused": requested in {"Alice correspondence"}}
                for requested in (
                    "Alice correspondence",
                    "alice correspondence",
                    "Alice-correspondence",
                )
            ],
            "scale": breadth_scale,
        },
        "depth": {
            "implementation": "full_execution_log_rehydration",
            "history_is_unbounded_by_default": True,
            "default_conversation_limit": None,
            "scale": depth_scale,
        },
        "limitations": [
            "This baseline measures deterministic prompt growth, not live-model routing accuracy.",
            "No external LLM, Gmail account, or API credentials are used.",
        ],
    }


def render_baseline_markdown(report: dict[str, object]) -> str:
    """Render a concise reviewer-readable view of baseline JSON."""

    breadth = report["breadth"]
    depth = report["depth"]
    assert isinstance(breadth, dict) and isinstance(depth, dict)

    lines = [
        "# Current OpenPoke baseline",
        "",
        "These are deterministic prompt-size measurements, not claims about live-model accuracy.",
        "",
        "## Roster breadth",
        "",
        "| Agents | Injected | Characters | Bytes |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for point in breadth["scale"]:
        lines.append(
            f"| {point['roster_size']} | {point['agents_injected']} | "
            f"{point['rendered_characters']} | {point['rendered_bytes']} |"
        )

    lines.extend(
        [
            "",
            "## History depth",
            "",
            "| Entries | Loaded | Characters | Bytes |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for point in depth["scale"]:
        lines.append(
            f"| {point['history_entries']} | {point['entries_loaded']} | "
            f"{point['rendered_characters']} | {point['rendered_bytes']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_baseline_report(
    report: dict[str, object],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Persist equivalent machine- and human-readable baseline reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_baseline_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("evals/results/baseline.json"))
    parser.add_argument("--markdown", type=Path, default=Path("evals/results/baseline.md"))
    args = parser.parse_args()
    write_baseline_report(build_baseline_report(), json_path=args.json, markdown_path=args.markdown)
    print(f"Wrote {args.json} and {args.markdown}")


if __name__ == "__main__":
    main()
