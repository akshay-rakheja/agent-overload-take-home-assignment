"""Tests for artifact secret scanning and mailbox redaction validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.live_lab.secret_scan import (
    SecretScanFinding,
    SecretScanReport,
    scan_file,
    scan_text,
)


def test_clean_sanitized_text_passes() -> None:
    text = (
        "# Evaluation Lab Report\n"
        "Model: openai/gpt-4.1-mini\n"
        "Fact ID: fact-dr-appointment\n"
        "Manifest SHA-256: 3004313f83fe284050d53c299c8fc4063ea1ec1f44a80e15967bf0225139a0fa\n"
        "Headers: Authorization: [REDACTED]\n"
        "Status: completed\n"
    )
    findings = scan_text(text)
    assert len(findings) == 0


def test_detects_openrouter_api_key() -> None:
    text = "OPENROUTER_API_KEY=sk-or-v1-abcdef1234567890abcdef1234567890\n"
    findings = scan_text(text)
    assert len(findings) >= 1
    assert any("openrouter_api_key" in f.rule.lower() or "api_key" in f.rule.lower() for f in findings)


def test_detects_bearer_token() -> None:
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdef\n"
    findings = scan_text(text)
    assert len(findings) >= 1
    assert any("bearer" in f.rule.lower() for f in findings)


def test_detects_oauth_access_token_and_code() -> None:
    text1 = "https://example.com/oauth/callback?code=secret_auth_code_12345678\n"
    findings1 = scan_text(text1)
    assert len(findings1) >= 1
    assert any("oauth" in f.rule.lower() or "code" in f.rule.lower() for f in findings1)

    text2 = "access_token = 'secret_access_token_abcdef123456'\n"
    findings2 = scan_text(text2)
    assert len(findings2) >= 1


def test_detects_private_email_address() -> None:
    text = "Message from real-user-mailbox@gmail.com regarding private appointment\n"
    findings = scan_text(text)
    assert len(findings) >= 1
    assert any("email" in f.rule.lower() for f in findings)


def test_ignores_mock_domains_and_fabrications() -> None:
    text = (
        "Contact: agent@openpoke.local or test@example.com\n"
        "Fact: fact-dr-appointment\n"
        "Fabricated: True\n"
        "Fixture: fixture-provider\n"
    )
    findings = scan_text(text)
    assert len(findings) == 0


def test_detects_machine_specific_home_path_when_checked() -> None:
    text = "Log written to /Users/akshayrakheja/Documents/secret_path.log\n"
    findings = scan_text(text, check_machine_paths=True)
    assert len(findings) >= 1
    assert any("machine_path" in f.rule.lower() for f in findings)


def test_scan_file(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("All good with [REDACTED] tokens.\n", encoding="utf-8")
    report = scan_file(f)
    assert report.valid
    assert len(report.findings) == 0
