"""Backward-compatible persisted paired-run contracts."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from server.services.evaluation_lab.run_store import RunStore


def test_schema_v1_run_store_reads_scorecards_written_before_pair_association(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[4]
    fixture = json.loads(
        (repository / "web" / "tests" / "fixtures" / "backend.json").read_text(
            encoding="utf-8"
        )
    )
    legacy = fixture["evidence_run"]
    for scorecard in legacy["scorecards"]:
        scorecard.pop("passed")
        for turn in scorecard["turns"]:
            turn.pop("passed")
        scorecard.pop("pair_id")
        scorecard.pop("repetition")

    root = tmp_path / ".lab" / "runs"
    root.mkdir(parents=True)
    run_id = UUID(legacy["run_id"])
    (root / f"{run_id}.json").write_text(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    restored = RunStore(root).get(run_id)

    assert restored is not None
    assert all(scorecard.pair_id is None for scorecard in restored.scorecards)
    assert all(scorecard.repetition is None for scorecard in restored.scorecards)
