"""Strict Git identity checks for the read-only historical worktree."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict


HISTORICAL_BASE_SHA = "5b5f635935a64ab37884c025d70abb0ed731c094"
APPROVED_OVERLAY_HEAD = "68434a5a2345073d74aeffa0dd7e28cb96479cc7"
APPROVED_OVERLAY_DIFF_SHA256 = (
    "50297a63b77e8ddebbbd32a49383ca912c55c88566457d09fe654115af5ccfbd"
)
APPROVED_OVERLAY_PATHS = frozenset(
    {
        "server/agents/execution_agent/runtime.py",
        "server/agents/execution_agent/tools/registry.py",
        "server/agents/interaction_agent/tools.py",
        "server/config.py",
        "server/requirements-dev.txt",
        "server/services/evaluation_lab/__init__.py",
        "server/services/evaluation_lab/policy.py",
        "server/services/gmail/client.py",
        "server/tests/agents/execution_agent/test_lab_tool_policy.py",
        "server/tests/compatibility/test_historical_source_guard.py",
        "server/tests/fixtures/composio_link_responses.json",
        "server/tests/services/evaluation_lab/test_policy.py",
        "server/tests/services/gmail/test_client_contract.py",
    }
)
PROTECTED_ORIGINAL_REPO = Path(__file__).resolve().parents[3] / "openpoke"


class BaselineRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_sha: str
    overlay_head_sha: str
    overlay_diff_sha256: str
    changed_paths: tuple[str, ...]


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=not binary,
    )
    if completed.returncode:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise ValueError(f"git verification failed: {stderr.strip()}")
    return completed.stdout


def _tracked_status(repo: Path) -> str:
    return str(_git(repo, "status", "--porcelain", "--untracked-files=no")).strip()


def _verify_protected_original(expected_base: str) -> None:
    original = PROTECTED_ORIGINAL_REPO
    if not original.exists():
        return
    try:
        head = str(_git(original, "rev-parse", "HEAD")).strip()
        status = _tracked_status(original)
    except ValueError as exc:
        raise ValueError(f"protected original verification failed: {exc}") from exc
    if head != expected_base or status:
        raise ValueError(
            "protected original mismatch: expected clean historical base "
            f"{expected_base}, got head={head} dirty={bool(status)}"
        )


def verify_baseline_revision(
    repo: Path,
    *,
    expected_base: str,
    allowed_paths: frozenset[str],
) -> BaselineRevision:
    """Accept only the pinned base or its one recorded overlay."""

    root = repo.resolve(strict=True)
    if not (root / ".git").exists() and not str(_git(root, "rev-parse", "--git-dir")).strip():
        raise ValueError(f"baseline repo is not a Git checkout: {root}")
    if _tracked_status(root):
        raise ValueError("baseline worktree has dirty tracked state")

    head = str(_git(root, "rev-parse", "HEAD")).strip()
    try:
        resolved_base = str(_git(root, "rev-parse", f"{expected_base}^{{commit}}")).strip()
    except ValueError as exc:
        raise ValueError(f"expected base is unavailable: {expected_base}") from exc
    if resolved_base != expected_base:
        raise ValueError(f"expected base must be a full exact SHA: {expected_base}")
    merge_base = str(_git(root, "merge-base", expected_base, head)).strip()
    if merge_base != expected_base:
        raise ValueError(f"HEAD does not descend from expected base {expected_base}")

    changed = tuple(
        sorted(
            path
            for path in str(
                _git(root, "diff", "--name-only", f"{expected_base}..{head}")
            ).splitlines()
            if path
        )
    )
    unexpected = sorted(set(changed) - set(allowed_paths))
    if unexpected:
        raise ValueError(f"overlay contains paths outside allowed set: {unexpected}")

    diff = bytes(_git(root, "diff", "--binary", f"{expected_base}..{head}", binary=True))
    digest = hashlib.sha256(diff).hexdigest()
    if head == expected_base:
        if changed:
            raise ValueError("base checkout unexpectedly contains an overlay")
    elif head != APPROVED_OVERLAY_HEAD:
        raise ValueError(f"unapproved overlay head: {head}")
    elif digest != APPROVED_OVERLAY_DIFF_SHA256:
        raise ValueError(
            "approved overlay checksum mismatch: "
            f"expected {APPROVED_OVERLAY_DIFF_SHA256}, got {digest}"
        )

    _verify_protected_original(expected_base)
    return BaselineRevision(
        base_sha=expected_base,
        overlay_head_sha=head,
        overlay_diff_sha256=digest,
        changed_paths=changed,
    )


__all__ = [
    "APPROVED_OVERLAY_DIFF_SHA256",
    "APPROVED_OVERLAY_HEAD",
    "APPROVED_OVERLAY_PATHS",
    "BaselineRevision",
    "HISTORICAL_BASE_SHA",
    "verify_baseline_revision",
]
