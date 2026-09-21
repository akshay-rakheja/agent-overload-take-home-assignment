from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from evals.live_lab import revisions


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=Live Lab",
        "-c",
        "user.email=live-lab@example.invalid",
        "commit",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "baseline"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "server").mkdir()
    (repo / "server" / "config.py").write_text("BASE = True\n", encoding="utf-8")
    base = _commit(repo, "historical base")
    return repo, base


def _configure_overlay(monkeypatch, repo: Path, base: str, head: str) -> str:
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", f"{base}..{head}"],
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256(diff).hexdigest()
    monkeypatch.setattr(revisions, "APPROVED_OVERLAY_HEAD", head)
    monkeypatch.setattr(revisions, "APPROVED_OVERLAY_DIFF_SHA256", digest)
    return digest


def test_exact_base_without_overlay_is_accepted(tmp_path: Path, monkeypatch) -> None:
    repo, base = _repo(tmp_path)
    monkeypatch.setattr(revisions, "PROTECTED_ORIGINAL_REPO", tmp_path / "absent")

    result = revisions.verify_baseline_revision(
        repo,
        expected_base=base,
        allowed_paths=frozenset({"server/config.py"}),
    )

    assert result.base_sha == base
    assert result.overlay_head_sha == base
    assert result.changed_paths == ()
    assert result.overlay_diff_sha256 == hashlib.sha256(b"").hexdigest()


def test_exact_approved_overlay_is_accepted(tmp_path: Path, monkeypatch) -> None:
    repo, base = _repo(tmp_path)
    (repo / "server" / "config.py").write_text("BASE = True\nLAB = True\n", encoding="utf-8")
    head = _commit(repo, "approved overlay")
    digest = _configure_overlay(monkeypatch, repo, base, head)
    monkeypatch.setattr(revisions, "PROTECTED_ORIGINAL_REPO", tmp_path / "absent")

    result = revisions.verify_baseline_revision(
        repo,
        expected_base=base,
        allowed_paths=frozenset({"server/config.py"}),
    )

    assert result.overlay_head_sha == head
    assert result.overlay_diff_sha256 == digest
    assert result.changed_paths == ("server/config.py",)


def test_wrong_base_is_rejected(tmp_path: Path, monkeypatch) -> None:
    repo, base = _repo(tmp_path)
    monkeypatch.setattr(revisions, "PROTECTED_ORIGINAL_REPO", tmp_path / "absent")

    with pytest.raises(ValueError, match="base"):
        revisions.verify_baseline_revision(
            repo,
            expected_base="0" * 40,
            allowed_paths=frozenset({"server/config.py"}),
        )


def test_extra_overlay_path_is_rejected(tmp_path: Path, monkeypatch) -> None:
    repo, base = _repo(tmp_path)
    (repo / "unexpected.py").write_text("surprise = True\n", encoding="utf-8")
    head = _commit(repo, "unexpected overlay")
    _configure_overlay(monkeypatch, repo, base, head)
    monkeypatch.setattr(revisions, "PROTECTED_ORIGINAL_REPO", tmp_path / "absent")

    with pytest.raises(ValueError, match="allowed"):
        revisions.verify_baseline_revision(
            repo,
            expected_base=base,
            allowed_paths=frozenset({"server/config.py"}),
        )


def test_dirty_tracked_worktree_is_rejected(tmp_path: Path, monkeypatch) -> None:
    repo, base = _repo(tmp_path)
    monkeypatch.setattr(revisions, "PROTECTED_ORIGINAL_REPO", tmp_path / "absent")
    (repo / "server" / "config.py").write_text("dirty = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dirty tracked"):
        revisions.verify_baseline_revision(
            repo,
            expected_base=base,
            allowed_paths=frozenset({"server/config.py"}),
        )


def test_protected_original_must_remain_clean_at_base(tmp_path: Path, monkeypatch) -> None:
    repo, base = _repo(tmp_path)
    original = tmp_path / "openpoke"
    _git(tmp_path, "clone", "-q", str(repo), str(original))
    monkeypatch.setattr(revisions, "PROTECTED_ORIGINAL_REPO", original)
    (original / "server" / "config.py").write_text("dirty = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protected original"):
        revisions.verify_baseline_revision(
            repo,
            expected_base=base,
            allowed_paths=frozenset({"server/config.py"}),
        )
