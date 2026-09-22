"""Strict environment inspection and canonical secret-file boundary checks."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict


REQUIRED_ENV_VARS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "COMPOSIO_API_KEY",
    "COMPOSIO_GMAIL_AUTH_CONFIG_ID",
    "OPENPOKE_LAB_ENABLED",
    "OPENPOKE_LAB_COMPOSIO_USER_ID",
    "OPENPOKE_AGENT_RETRIEVAL_TOP_K",
    "OPENPOKE_AGENT_RETRIEVAL_MIN_SCORE",
    "OPENPOKE_AGENT_ROUTE_REUSE_THRESHOLD",
    "OPENPOKE_AGENT_ROUTE_AMBIGUITY_MARGIN",
    "OPENPOKE_AGENT_ROUTING_CONTEXT_MAX_CHARACTERS",
    "OPENPOKE_EXECUTION_CONTEXT_MAX_RECENT_EPISODES",
)

DEFAULT_CANONICAL_ENV_PATH = Path(
    "/Users/akshayrakheja/Documents/general-magic-take-home/.openpoke-lab.env"
)


class EnvironmentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    canonical_env_path: str
    canonical_env_sha256: str
    canonical_env_mode: str
    present_variables: tuple[str, ...]
    missing_variables: tuple[str, ...]
    worktrees_checked: tuple[str, ...]
    errors: tuple[str, ...]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def parse_variable_names(path: Path) -> set[str]:
    names: set[str] = set()
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return names

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key:
                names.add(key)
    return names


def _is_git_ignored(repo: Path, relative_file: str) -> bool:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", relative_file],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def check_environment(
    *,
    canonical_env: Path = DEFAULT_CANONICAL_ENV_PATH,
    worktrees: Sequence[Path] = (),
    required_vars: Sequence[str] = REQUIRED_ENV_VARS,
    verify_git_ignored: bool = True,
) -> EnvironmentReport:
    errors: list[str] = []
    canonical_path_str = str(canonical_env)
    canonical_sha256 = ""
    canonical_mode_str = ""
    present_vars: set[str] = set()
    missing_vars: list[str] = []
    worktrees_checked: list[str] = []

    # 1. Canonical file inspection
    if not canonical_env.exists():
        errors.append(f"Canonical environment file is absent: {canonical_path_str}")
    elif canonical_env.is_symlink():
        errors.append(f"Canonical environment file must be a regular file, not a symlink: {canonical_path_str}")
    else:
        st = canonical_env.stat()
        mode_val = stat.S_IMODE(st.st_mode)
        canonical_mode_str = oct(mode_val)[2:].zfill(4)
        if mode_val != 0o600:
            errors.append(
                f"Canonical environment file mode must be 0600 (got {canonical_mode_str}): {canonical_path_str}"
            )
        try:
            canonical_sha256 = hash_file(canonical_env)
            present_vars = parse_variable_names(canonical_env)
        except Exception as exc:
            errors.append(f"Failed to read canonical environment file: {exc}")

    # Check required variables
    for req in required_vars:
        if req not in present_vars:
            missing_vars.append(req)
    if missing_vars:
        errors.append(f"Missing required variable names in canonical environment: {missing_vars}")

    # 2. Worktree symlink and isolation inspection
    canonical_resolved = canonical_env.resolve() if canonical_env.exists() else None

    for wt in worktrees:
        wt_path = Path(wt)
        worktrees_checked.append(str(wt_path))
        env_link = wt_path / ".env"
        if not env_link.exists() and not env_link.is_symlink():
            errors.append(f"Worktree .env is missing: {env_link}")
            continue

        if not env_link.is_symlink():
            errors.append(f"Physical credential copy detected in worktree (must be symlink): {env_link}")
            continue

        try:
            target_resolved = env_link.resolve()
            if canonical_resolved is not None and target_resolved != canonical_resolved:
                errors.append(
                    f"Worktree .env symlink target mismatch: points to {target_resolved}, expected {canonical_resolved}"
                )
        except Exception as exc:
            errors.append(f"Could not resolve worktree .env symlink {env_link}: {exc}")

        if verify_git_ignored and wt_path.exists() and (wt_path / ".git").exists():
            if not _is_git_ignored(wt_path, ".env"):
                errors.append(f"Worktree .env must be gitignored: {env_link}")

    valid = len(errors) == 0
    return EnvironmentReport(
        valid=valid,
        canonical_env_path=canonical_path_str,
        canonical_env_sha256=canonical_sha256,
        canonical_env_mode=canonical_mode_str,
        present_variables=tuple(sorted(present_vars)),
        missing_variables=tuple(sorted(missing_vars)),
        worktrees_checked=tuple(worktrees_checked),
        errors=tuple(errors),
    )


__all__ = [
    "DEFAULT_CANONICAL_ENV_PATH",
    "EnvironmentReport",
    "REQUIRED_ENV_VARS",
    "check_environment",
    "hash_file",
    "parse_variable_names",
]
