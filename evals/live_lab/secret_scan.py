"""Secret and unredacted mailbox scanning for Evaluation Lab artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict


class SecretScanFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    snippet: str
    line_number: int | None = None


class SecretScanReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    file_path: str | None = None
    findings: tuple[SecretScanFinding, ...]


_RULES = [
    ("openrouter_api_key", re.compile(r"(?i)OPENROUTER_API_KEY\s*[:=]\s*['\"]?(sk-[a-zA-Z0-9_\-]{10,}|[a-zA-Z0-9_\-]{20,})")),
    ("composio_api_key", re.compile(r"(?i)COMPOSIO_API_KEY\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{16,})")),
    ("bearer_token", re.compile(r"(?i)Authorization\s*:\s*Bearer\s+(?!\[REDACTED\])([a-zA-Z0-9_\-\.]{16,})")),
    ("oauth_code", re.compile(r"(?i)(?:[?&]code=|oauth_code\s*[:=]\s*['\"]?)(?!\[REDACTED\]|fixture-code)([a-zA-Z0-9_\-]{16,})")),
    ("oauth_token", re.compile(r"(?i)(?:access_token|refresh_token)\s*[:=]\s*['\"]?(?!\[REDACTED\]|fixture-token)([a-zA-Z0-9_\-]{16,})")),
    ("private_email", re.compile(r"\b[a-zA-Z0-9_.+-]+@(?!example\.com|openpoke\.local|test\.com)[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")),
]

_MACHINE_PATH_RULE = ("machine_path", re.compile(r"/(?:Users|home)/[a-zA-Z0-9_\-]+/"))


def scan_text(text: str, filename: str = "", check_machine_paths: bool = False) -> list[SecretScanFinding]:
    findings: list[SecretScanFinding] = []
    lines = text.splitlines()

    rules = list(_RULES)
    if check_machine_paths:
        rules.append(_MACHINE_PATH_RULE)

    for line_idx, line in enumerate(lines, start=1):
        # Ignore comments or fixture declarations if they are manifest references
        if "fact-dr-appointment" in line or "fact-router-creds" in line:
            # Clean safe fixture facts
            pass

        for rule_name, pattern in rules:
            for match in pattern.finditer(line):
                matched_str = match.group(0)
                # Redact matched text in the snippet
                snippet = f"Line {line_idx}: {rule_name} detected"
                findings.append(
                    SecretScanFinding(
                        rule=rule_name,
                        snippet=snippet,
                        line_number=line_idx,
                    )
                )

    return findings


def scan_file(path: Path, check_machine_paths: bool = False) -> SecretScanReport:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return SecretScanReport(valid=True, file_path=str(path), findings=())
    except Exception as exc:
        return SecretScanReport(
            valid=False,
            file_path=str(path),
            findings=(
                SecretScanFinding(rule="unreadable_file", snippet=str(exc), line_number=None),
            ),
        )

    findings = scan_text(content, filename=path.name, check_machine_paths=check_machine_paths)
    return SecretScanReport(
        valid=len(findings) == 0,
        file_path=str(path),
        findings=tuple(findings),
    )


def scan_directory(root: Path, check_machine_paths: bool = False, excludes: Sequence[str] = ()) -> list[SecretScanReport]:
    reports: list[SecretScanReport] = []
    for path in root.rglob("*"):
        if path.is_file():
            if any(exc in str(path) for exc in excludes):
                continue
            rep = scan_file(path, check_machine_paths=check_machine_paths)
            if not rep.valid:
                reports.append(rep)
    return reports


__all__ = [
    "SecretScanFinding",
    "SecretScanReport",
    "scan_directory",
    "scan_file",
    "scan_text",
]
