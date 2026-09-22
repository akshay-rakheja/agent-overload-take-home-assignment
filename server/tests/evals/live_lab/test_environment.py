"""Tests for local lab environment safety and inspection."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from evals.live_lab.environment import (
    REQUIRED_ENV_VARS,
    EnvironmentReport,
    check_environment,
    hash_file,
    parse_variable_names,
)


def _write_canonical_env(path: Path, vars_dict: dict[str, str], mode: int = 0o600) -> None:
    content = "\n".join(f"{k}={v}" for k, v in vars_dict.items()) + "\n"
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def _all_required_vars() -> dict[str, str]:
    return {var: f"secret-val-for-{var}" for var in REQUIRED_ENV_VARS}


def test_parse_variable_names_extracts_names_only(tmp_path: Path) -> None:
    env_file = tmp_path / ".test.env"
    env_file.write_text(
        "# Comment line\n"
        "FOO=bar_secret\n"
        "\n"
        "BAZ='qux_secret'\n"
        "SPACED = value_secret \n",
        encoding="utf-8",
    )
    names = parse_variable_names(env_file)
    assert names == {"FOO", "BAZ", "SPACED"}


def test_hash_file_deterministic(tmp_path: Path) -> None:
    f1 = tmp_path / "f1.env"
    f2 = tmp_path / "f2.env"
    f1.write_text("TEST=1\n", encoding="utf-8")
    f2.write_text("TEST=1\n", encoding="utf-8")
    assert hash_file(f1) == hash_file(f2)
    assert len(hash_file(f1)) == 64


def test_canonical_env_missing_fails(tmp_path: Path) -> None:
    canonical = tmp_path / ".missing.env"
    report = check_environment(canonical_env=canonical, worktrees=[])
    assert not report.valid
    assert any("absent" in err.lower() or "missing" in err.lower() for err in report.errors)


def test_canonical_env_wrong_mode_fails(tmp_path: Path) -> None:
    canonical = tmp_path / ".openpoke-lab.env"
    _write_canonical_env(canonical, _all_required_vars(), mode=0o644)
    report = check_environment(canonical_env=canonical, worktrees=[])
    assert not report.valid
    assert any("0600" in err for err in report.errors)


def test_canonical_env_is_symlink_fails(tmp_path: Path) -> None:
    real = tmp_path / "real.env"
    _write_canonical_env(real, _all_required_vars(), mode=0o600)
    canonical = tmp_path / "symlink.env"
    canonical.symlink_to(real)
    report = check_environment(canonical_env=canonical, worktrees=[])
    assert not report.valid
    assert any("symlink" in err.lower() for err in report.errors)


def test_missing_required_variable_names_fails(tmp_path: Path) -> None:
    canonical = tmp_path / ".openpoke-lab.env"
    vars_dict = _all_required_vars()
    del vars_dict["OPENROUTER_API_KEY"]
    del vars_dict["OPENPOKE_AGENT_RETRIEVAL_TOP_K"]
    _write_canonical_env(canonical, vars_dict, mode=0o600)

    report = check_environment(canonical_env=canonical, worktrees=[])
    assert not report.valid
    assert "OPENROUTER_API_KEY" in report.missing_variables
    assert "OPENPOKE_AGENT_RETRIEVAL_TOP_K" in report.missing_variables
    # Ensure no secret values appear in report or errors
    for err in report.errors:
        assert "secret-val" not in err


def test_worktree_physical_copy_fails(tmp_path: Path) -> None:
    canonical = tmp_path / ".openpoke-lab.env"
    _write_canonical_env(canonical, _all_required_vars(), mode=0o600)

    worktree = tmp_path / "wt"
    worktree.mkdir()
    wt_env = worktree / ".env"
    _write_canonical_env(wt_env, _all_required_vars(), mode=0o600)  # physical copy, not symlink

    report = check_environment(canonical_env=canonical, worktrees=[worktree])
    assert not report.valid
    assert any("physical credential copy" in err.lower() or "not a symlink" in err.lower() for err in report.errors)


def test_worktree_wrong_symlink_target_fails(tmp_path: Path) -> None:
    canonical = tmp_path / ".openpoke-lab.env"
    _write_canonical_env(canonical, _all_required_vars(), mode=0o600)

    wrong_target = tmp_path / "wrong.env"
    wrong_target.write_text("OTHER=1\n", encoding="utf-8")

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".env").symlink_to(wrong_target)

    report = check_environment(canonical_env=canonical, worktrees=[worktree])
    assert not report.valid
    assert any("target" in err.lower() or "mismatch" in err.lower() for err in report.errors)


def test_environment_success_with_symlinked_worktree(tmp_path: Path) -> None:
    canonical = tmp_path / ".openpoke-lab.env"
    _write_canonical_env(canonical, _all_required_vars(), mode=0o600)

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".env").symlink_to(canonical)

    report = check_environment(canonical_env=canonical, worktrees=[worktree], verify_git_ignored=False)
    assert report.valid
    assert len(report.errors) == 0
    assert report.canonical_env_mode == "0600"
    assert len(report.canonical_env_sha256) == 64
    assert set(report.present_variables) == set(REQUIRED_ENV_VARS)
    assert len(report.missing_variables) == 0
    # No secret values disclosed in any report field
    serialized = report.model_dump_json()
    assert "secret-val" not in serialized
