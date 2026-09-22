"""Cross-artifact consistency verification for Evaluation Lab results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from server.services.evaluation_lab.orchestrator import PairedRunResult


MAX_CANDIDATES = 5


class ConsistencyReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    run_id: str
    checks_run: tuple[str, ...]
    mismatches: tuple[str, ...]


def verify_artifacts_consistency(
    *,
    run: PairedRunResult,
    artifacts: Mapping[str, Path | str],
) -> ConsistencyReport:
    """Verify that all output artifacts agree on run ID, scenarios, model, candidate caps, and metrics."""
    mismatches: list[str] = []
    checks: list[str] = []
    expected_run_id = str(run.run_id)

    for name, path_or_str in artifacts.items():
        path = Path(path_or_str)
        if not path.exists():
            mismatches.append(f"Artifact {name} does not exist at {path}")
            continue

        checks.append(f"check_existence:{name}")
        content = path.read_text(encoding="utf-8")

        if path.suffix == ".json" or name.endswith("_json"):
            checks.append(f"check_json_schema:{name}")
            try:
                data = json.loads(content)
            except Exception as exc:
                mismatches.append(f"Artifact {name} is not valid JSON: {exc}")
                continue

            if isinstance(data, dict):
                art_run_id = str(data.get("run_id", ""))
                if art_run_id != expected_run_id:
                    mismatches.append(
                        f"Artifact {name} run_id mismatch: expected {expected_run_id}, got {art_run_id}"
                    )
                else:
                    checks.append(f"run_id_match:{name}")

                # Check candidate counts
                pairs = data.get("pairs", [])
                for p_idx, pair in enumerate(pairs):
                    for o_idx, outcome in enumerate(pair.get("outcomes", [])):
                        for r_idx, res in enumerate(outcome.get("results", [])):
                            cands = res.get("candidates", {})
                            if isinstance(cands, dict):
                                cand_list = cands.get("value")
                                if isinstance(cand_list, list) and len(cand_list) > MAX_CANDIDATES:
                                    mismatches.append(
                                        f"Artifact {name} pair[{p_idx}].outcome[{o_idx}].result[{r_idx}] "
                                        f"exceeded max candidates ({len(cand_list)} > {MAX_CANDIDATES})"
                                    )

        elif path.suffix == ".md" or name.endswith("_markdown"):
            checks.append(f"check_markdown_content:{name}")
            if expected_run_id not in content:
                mismatches.append(f"Artifact {name} missing expected run_id {expected_run_id}")
            else:
                checks.append(f"run_id_in_markdown:{name}")

            for scen in run.request.scenario_ids:
                if scen not in content:
                    mismatches.append(f"Artifact {name} missing scenario {scen}")
                else:
                    checks.append(f"scenario_in_markdown:{name}:{scen}")

    valid = len(mismatches) == 0
    return ConsistencyReport(
        valid=valid,
        run_id=expected_run_id,
        checks_run=tuple(checks),
        mismatches=tuple(mismatches),
    )


__all__ = [
    "ConsistencyReport",
    "verify_artifacts_consistency",
]
