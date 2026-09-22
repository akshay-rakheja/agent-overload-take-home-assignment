"""Deterministic report rendering for Evaluation Lab paired runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from server.services.evaluation_lab.orchestrator import PairedRunResult


def render_paired_run_json(run: PairedRunResult) -> str:
    """Deterministic JSON serialization with sorted keys and 2-space indent."""
    data = json.loads(run.model_dump_json())
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_obs(val: Any) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, dict):
        avail = val.get("availability")
        if avail in ("unavailable", "not_applicable"):
            reason = val.get("reason")
            return f"Unavailable ({reason})" if reason else "Unavailable"
        if avail == "inferred":
            return f"{val.get('value')} (inferred)"
        return str(val.get("value", ""))
    return str(val)


def render_paired_run_markdown(run: PairedRunResult) -> str:
    """Generate deterministic Markdown summary report for a paired run."""
    models = sorted({
        outcome.model_id
        for pair in run.pairs
        for outcome in pair.outcomes
        if outcome.model_id
    })
    model_str = ", ".join(models) if models else "default"
    max_reps = max((pair.scheduled.repetition for pair in run.pairs), default=1)

    lines: list[str] = [
        "# Evaluation Lab Paired Run Report",
        "",
        "## Run Metadata",
        "",
        f"- **Run ID:** `{run.run_id}`",
        f"- **Status:** `{run.status.value}`",
        f"- **Execution Mode:** `{run.execution_mode.value}`",
        f"- **Model:** `{model_str}`",
        f"- **Created At:** `{run.created_at.isoformat()}`",
        f"- **Scenarios:** `{', '.join(run.request.scenario_ids)}`",
        f"- **Repetitions:** {max_reps}",
        "",
        "## Paired Outcomes",
        "",
        "| Scenario | Rep | System | Status | Mode | Candidates | Prompt Exposure | Final Response / Grade | Input Tokens | Output Tokens | Total Tokens |",
        "|---|---:|---|---|---|---|---|---|---:|---:|---:|",
    ]

    for pair in run.pairs:
        scen_id = pair.scheduled.scenario_id
        rep = pair.scheduled.repetition
        for outcome in pair.outcomes:
            sys_name = outcome.system.capitalize()
            status = outcome.status
            if outcome.results:
                for res in outcome.results:
                    mode = res.mode.value or res.mode.availability.value
                    if res.candidates and res.candidates.value:
                        cand_str = f"{len(res.candidates.value)} candidates"
                    else:
                        cand_str = res.candidates.availability.value if res.candidates else "none"

                    exp_str = "None"
                    if res.prompt_exposure and res.prompt_exposure.value:
                        if isinstance(res.prompt_exposure.value, dict):
                            count = res.prompt_exposure.value.get("exposed_name_count", 0)
                            chars = res.prompt_exposure.value.get("prompt_characters", 0)
                            exp_str = f"{count} names ({chars} chars)"
                        else:
                            exp_str = str(res.prompt_exposure.value)

                    resp_str = res.final_response.value or res.final_response.availability.value
                    if len(resp_str) > 40:
                        resp_str = resp_str[:37] + "..."

                    in_tok = res.usage.input_tokens.value if res.usage and res.usage.input_tokens else None
                    out_tok = res.usage.output_tokens.value if res.usage and res.usage.output_tokens else None
                    tot_tok = res.usage.total_tokens.value if res.usage and res.usage.total_tokens else None

                    in_tok_str = f"{in_tok:,}" if in_tok is not None else "-"
                    out_tok_str = f"{out_tok:,}" if out_tok is not None else "-"
                    tot_tok_str = f"{tot_tok:,}" if tot_tok is not None else "-"

                    lines.append(
                        f"| `{scen_id}` | {rep} | {sys_name} | `{status}` | `{mode}` | {cand_str} | {exp_str} | {resp_str} | {in_tok_str} | {out_tok_str} | {tot_tok_str} |"
                    )
            else:
                lines.append(
                    f"| `{scen_id}` | {rep} | {sys_name} | `{status}` | - | - | - | - | - | - | - |"
                )

    lines.extend([
        "",
        "## Scorecards",
        "",
    ])

    if run.scorecards:
        lines.extend([
            "| Scenario | Repetition | Passed | Reason / Detail |",
            "|---|---:|---|---|",
        ])
        for sc in run.scorecards:
            pass_str = "PASS" if sc.passed else "FAIL"
            detail = f"{len(sc.turns)} turns" if sc.turns else "no turns"
            lines.append(f"| `{sc.scenario_id}` | {sc.repetition or 1} | **{pass_str}** | {detail} |")
    else:
        lines.append("_No scorecards recorded for this run._")

    lines.append("")
    return "\n".join(lines)


def write_reports(run: PairedRunResult, output_dir: Path) -> tuple[Path, Path]:
    """Write report.json and report.md deterministically to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"

    json_path.write_text(render_paired_run_json(run), encoding="utf-8")
    md_path.write_text(render_paired_run_markdown(run), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "render_paired_run_json",
    "render_paired_run_markdown",
    "write_reports",
]
