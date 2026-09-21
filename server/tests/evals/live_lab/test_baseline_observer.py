from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from evals.live_lab import baseline_observer as observer_module
from evals.live_lab.baseline_observer import infer_baseline_outcome
from evals.live_lab.baseline_observer import run_baseline_turn
from evals.live_lab.raw_observation import BaselineTurnRequest, ObservedToolCall, StateFingerprint


def _state(roster, journals):
    payloads = {path: str(content).encode() for path, content in journals.items()}
    return StateFingerprint(
        roster=tuple(roster),
        journal_sha256={path: hashlib.sha256(payload).hexdigest() for path, payload in payloads.items()},
        journal_bytes={path: len(payload) for path, payload in payloads.items()},
        journal_contents=payloads,
        journal_complete={path: True for path in payloads},
    )


def _dispatch(name: str) -> ObservedToolCall:
    return ObservedToolCall(name="send_message_to_agent", arguments={"agent_name": name, "instructions": "do it"})


def test_existing_name_and_one_appended_journal_infers_reuse() -> None:
    before = _state(["Known"], {"known.log": "a"})
    after = _state(["Known"], {"known.log": "ab"})

    result = infer_baseline_outcome(before, after, [_dispatch("Known")])

    assert result.action == "reuse"
    assert result.name == "Known"
    assert "journal" in result.reason


def test_one_new_roster_name_infers_create() -> None:
    before = _state(["Known"], {"known.log": "a"})
    after = _state(["Known", "New Workflow"], {"known.log": "a", "new-workflow.log": "b"})

    result = infer_baseline_outcome(before, after, [_dispatch("New Workflow")])

    assert result.action == "create_new"
    assert result.name == "New Workflow"


def test_no_dispatch_is_abstain() -> None:
    state = _state(["Known"], {"known.log": "a"})

    result = infer_baseline_outcome(state, state, [])

    assert result.action == "abstain"
    assert result.name is None


def test_multiple_changed_journals_are_unobservable() -> None:
    before = _state(["A", "B"], {"a.log": "1", "b.log": "1"})
    after = _state(["A", "B"], {"a.log": "12", "b.log": "12"})

    result = infer_baseline_outcome(before, after, [_dispatch("A")])

    assert result.action == "unobservable"
    assert result.name is None
    assert "multiple" in result.reason


def test_truncated_journal_is_failed_observation() -> None:
    before = StateFingerprint(
        roster=("Known",),
        journal_sha256={"known.log": "a"},
        journal_bytes={"known.log": 100},
        journal_contents={"known.log": b"a" * 100},
    )
    after = StateFingerprint(
        roster=("Known",),
        journal_sha256={"known.log": "b"},
        journal_bytes={"known.log": 20},
        journal_contents={"known.log": b"a" * 20},
    )

    result = infer_baseline_outcome(before, after, [_dispatch("Known")])

    assert result.action == "unobservable"
    assert result.failed
    assert "truncated" in result.reason


def test_malformed_dispatch_arguments_are_failed_observation() -> None:
    state = _state(["Known"], {"known.log": "a"})
    call = ObservedToolCall(
        name="send_message_to_agent",
        arguments={},
        malformed=True,
    )

    result = infer_baseline_outcome(state, state, [call])

    assert result.action == "unobservable"
    assert result.failed
    assert "malformed" in result.reason


def test_partial_journal_append_is_failed_observation() -> None:
    before = StateFingerprint(
        roster=("Known",),
        journal_sha256={"known.log": "a"},
        journal_bytes={"known.log": 100},
        journal_complete={"known.log": True},
        journal_contents={"known.log": b"a" * 100},
    )
    after = StateFingerprint(
        roster=("Known",),
        journal_sha256={"known.log": "b"},
        journal_bytes={"known.log": 120},
        journal_complete={"known.log": False},
        journal_contents={"known.log": b"a" * 100 + b"partial-entry"},
    )

    result = infer_baseline_outcome(before, after, [_dispatch("Known")])

    assert result.action == "unobservable"
    assert result.failed
    assert "partial" in result.reason


def test_dispatch_to_a_with_only_b_journal_change_is_unobservable() -> None:
    before = _state(["Agent A", "Agent B"], {"agent-a.log": "a", "agent-b.log": "b"})
    after = _state(["Agent A", "Agent B"], {"agent-a.log": "a", "agent-b.log": "b-appended"})

    result = infer_baseline_outcome(before, after, [_dispatch("Agent A")])

    assert result.action == "unobservable"
    assert result.name is None
    assert "selected" in result.reason or "unrelated" in result.reason


def test_no_dispatch_with_truncated_journal_is_failed_not_abstain() -> None:
    before = _state(["Known"], {"known.log": "complete"})
    after = _state(["Known"], {"known.log": "com"})

    result = infer_baseline_outcome(before, after, [])

    assert result.action == "unobservable"
    assert result.failed
    assert "truncated" in result.reason


def test_slug_collision_cannot_support_unique_name_inference() -> None:
    before = _state(["A B", "A-B"], {"a-b.log": "before"})
    after = _state(["A B", "A-B"], {"a-b.log": "before-after"})

    result = infer_baseline_outcome(before, after, [_dispatch("A B")])

    assert result.action == "unobservable"
    assert result.name is None
    assert "unique" in result.reason or "collision" in result.reason


def test_size_growth_without_append_only_prefix_is_failed() -> None:
    before = _state(["Known"], {"known.log": "original"})
    after = _state(["Known"], {"known.log": "rewritten-original"})

    result = infer_baseline_outcome(before, after, [_dispatch("Known")])

    assert result.action == "unobservable"
    assert result.failed
    assert "append-only" in result.reason


def test_corrupt_roster_returns_failed_observation_instead_of_raising(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    execution_dir = data_dir / "execution_agents"
    execution_dir.mkdir(parents=True)
    (execution_dir / "roster.json").write_text("{corrupt", encoding="utf-8")
    request = BaselineTurnRequest(
        run_id="corrupt-roster",
        data_dir=str(data_dir),
        event_path=str(tmp_path / "events.jsonl"),
        user_message="hello",
        timeout_seconds=0.1,
    )

    observation = asyncio.run(run_baseline_turn(request))

    assert observation.inferred_action == "unobservable"
    assert observation.final_response is None
    assert any(error.code == "state_read_error" for error in observation.errors)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://baseline.invalid")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, *, fail_post=False, corrupt_after: Path | None = None, **kwargs):
        self.fail_post = fail_post
        self.corrupt_after = corrupt_after
        self.get_count = 0
        self.post_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url):
        self.get_count += 1
        messages = [] if self.get_count == 1 else [{"role": "assistant", "content": "done"}]
        return _FakeResponse({"messages": messages})

    async def post(self, url, json):
        self.post_count += 1
        if self.fail_post:
            raise httpx.ConnectError("connection failed")
        if self.corrupt_after is not None:
            self.corrupt_after.write_text("{partial", encoding="utf-8")
        return _FakeResponse({}, status_code=202)


def _observer_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    execution_dir = data_dir / "execution_agents"
    execution_dir.mkdir(parents=True)
    (execution_dir / "roster.json").write_text('["Known"]\n', encoding="utf-8")
    (execution_dir / "known.log").write_text(
        "<agent_request>before</agent_request>\n",
        encoding="utf-8",
    )
    event_path = tmp_path / "run" / "events.jsonl"
    event_path.parent.mkdir()
    (event_path.parent / "process_context.json").write_text(
        json.dumps({"process_nonce": "fixture-process"}),
        encoding="utf-8",
    )
    return data_dir, event_path


def test_http_connection_error_returns_sanitized_failed_observation(tmp_path: Path, monkeypatch) -> None:
    data_dir, event_path = _observer_fixture(tmp_path)
    monkeypatch.setattr(
        observer_module.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeClient(fail_post=True),
    )
    request = BaselineTurnRequest(
        run_id="http-error",
        data_dir=str(data_dir),
        event_path=str(event_path),
        user_message="hello",
        timeout_seconds=0.1,
    )

    observation = asyncio.run(run_baseline_turn(request))

    assert observation.inferred_action == "unobservable"
    assert any(error.code == "http_error" for error in observation.errors)
    assert all("connection failed" not in error.message for error in observation.errors)


def test_corrupt_event_line_cannot_support_success(tmp_path: Path, monkeypatch) -> None:
    data_dir, event_path = _observer_fixture(tmp_path)
    event_path.write_text("{corrupt-event\n", encoding="utf-8")
    monkeypatch.setattr(
        observer_module.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeClient(),
    )
    request = BaselineTurnRequest(
        run_id="corrupt-event",
        data_dir=str(data_dir),
        event_path=str(event_path),
        user_message="hello",
        timeout_seconds=0.1,
    )

    observation = asyncio.run(run_baseline_turn(request))

    assert observation.inferred_action == "unobservable"
    assert any(error.code == "event_corruption" for error in observation.errors)


def test_partial_roster_after_submission_returns_failed_observation(tmp_path: Path, monkeypatch) -> None:
    data_dir, event_path = _observer_fixture(tmp_path)
    roster_path = data_dir / "execution_agents" / "roster.json"
    monkeypatch.setattr(
        observer_module.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeClient(corrupt_after=roster_path),
    )
    request = BaselineTurnRequest(
        run_id="partial-roster",
        data_dir=str(data_dir),
        event_path=str(event_path),
        user_message="hello",
        timeout_seconds=0.1,
    )

    observation = asyncio.run(run_baseline_turn(request))

    assert observation.inferred_action == "unobservable"
    assert any(error.code == "state_read_error" for error in observation.errors)
