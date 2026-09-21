"""Strict contracts for fabricated self-envelope Gmail fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from evals.live_lab.fixture_email import (
    SUBJECT_PREFIX,
    build_fact_manifest,
    fixture_fact_ids,
    render_fixture_messages,
    write_pre_send_manifest,
)


def test_renderer_has_exactly_eight_stable_self_envelopes() -> None:
    first = render_fixture_messages("run_A7k29mQ4")
    second = render_fixture_messages("run_A7k29mQ4")

    assert first == second
    assert len(first) == 8
    assert len({message.template_id for message in first}) == 8
    assert len({message.subject for message in first}) == 8
    assert all(message.envelope_from == "self" for message in first)
    assert all(message.envelope_to == "self" for message in first)
    assert all(message.subject.startswith(f"{SUBJECT_PREFIX} run_A7k29mQ4 ") for message in first)
    assert fixture_fact_ids() == {
        "SEC-7419",
        "ENG-2284",
        "NF-3207",
        "VF-20481",
        "MS-8820",
        "CW-8117",
        "ARC-1042",
        "AMB-6063",
    }


def test_renderer_contains_every_fabricated_fact_and_no_banned_content() -> None:
    messages = render_fixture_messages("opaque_64uN2pR8")
    rendered = "\n".join(
        f"{message.subject}\n{message.body}\n{message.envelope_from}\n{message.envelope_to}"
        for message in messages
    )

    for expected in (
        "2026-09-18 04:12 UTC",
        "Lisbon",
        "Pixel 10",
        "indigo-orbit",
        "183 likes",
        "27 comments",
        "Aurora Loop",
        "Prism Cut 2.4",
        "2026-10-07",
        "Storyboard Lock",
        "Pro Render Monthly",
        "CAD 47.80",
        "2026-09-19",
        "Temporal Layers",
        "2026-10-11 17:30 UTC",
        "GLASS-52",
        "CAD 312.40",
        "2026-10-15",
        "PO-4406",
        "Cedar Comet",
        "2026-08-29",
        "9f2c7a",
    ):
        assert expected in rendered

    assert not re.search(
        r"[\w.+-]+@[\w.-]+|https?://|oauth|bearer|api[_ -]?key|access[_ -]?token|client[_ -]?secret",
        rendered,
        re.I,
    )


@pytest.mark.parametrize(
    "run_id",
    ["short", "contains spaces", "user@example.com", "https://fixture.invalid", "a" * 65],
)
def test_renderer_rejects_nonopaque_or_banned_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        render_fixture_messages(run_id)


def test_sanitized_fact_manifest_has_a_reproducible_self_digest(tmp_path: Path) -> None:
    messages = render_fixture_messages("manifest_7Yp4kD2x")
    first = build_fact_manifest(messages)
    second = build_fact_manifest(messages)

    assert first == second
    assert set(first.facts) == fixture_fact_ids()
    assert all(item.subject_sha256 not in {message.subject for message in messages} for item in first.facts.values())
    unsigned = first.model_dump(mode="json", exclude={"manifest_sha256"})
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert first.manifest_sha256 == hashlib.sha256(canonical).hexdigest()

    destination = tmp_path / ".lab" / "pre-send" / "facts.json"
    written = write_pre_send_manifest(destination, messages)
    assert written == first
    assert json.loads(destination.read_text(encoding="utf-8"))["manifest_sha256"] == first.manifest_sha256


def test_pre_send_manifest_refuses_tracked_destinations(tmp_path: Path) -> None:
    messages = render_fixture_messages("manifest_7Yp4kD2x")

    with pytest.raises(ValueError, match="ignored .lab"):
        write_pre_send_manifest(tmp_path / "facts.json", messages)
