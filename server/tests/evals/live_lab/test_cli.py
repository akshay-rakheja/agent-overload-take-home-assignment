"""Tests for evals.live_lab.cli commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.live_lab.cli import main
from server.tests.evals.live_lab.test_environment import _all_required_vars, _write_canonical_env


def test_cli_preflight_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    canonical = tmp_path / ".openpoke-lab.env"
    _write_canonical_env(canonical, _all_required_vars(), mode=0o600)

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".env").symlink_to(canonical)

    with patch("evals.live_lab.cli.check_environment") as mock_check:
        from evals.live_lab.environment import EnvironmentReport
        mock_check.return_value = EnvironmentReport(
            valid=True,
            canonical_env_path=str(canonical),
            canonical_env_sha256="abc",
            canonical_env_mode="0600",
            present_variables=(),
            missing_variables=(),
            worktrees_checked=(),
            errors=(),
        )
        code = main(["preflight", "--canonical-env", str(canonical)])
        assert code == 0
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert data["valid"] is True


def test_cli_evaluate_and_report_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "eval_out"

    # 1. Evaluate
    code = main([
        "evaluate",
        "--offline",
        "--scenario",
        "scenario-exact-named-reuse",
        "--repetitions",
        "1",
        "--output",
        str(out_dir),
    ])
    assert code == 0
    assert (out_dir / "run.json").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()

    # 2. Report
    report_out_dir = tmp_path / "report_out"
    code = main([
        "report",
        "--run-file",
        str(out_dir / "run.json"),
        "--output",
        str(report_out_dir),
    ])
    assert code == 0
    assert (report_out_dir / "report.json").exists()
    assert (report_out_dir / "report.md").exists()

    # 3. Verify
    capsys.readouterr()
    code = main([
        "verify",
        "--artifacts",
        str(out_dir),
    ])
    assert code == 0
    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["valid"] is True
    assert data["secret_scan_failures"] == 0
