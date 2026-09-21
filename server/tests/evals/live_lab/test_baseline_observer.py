from __future__ import annotations

from evals.live_lab.baseline_observer import infer_baseline_outcome
from evals.live_lab.raw_observation import ObservedToolCall, StateFingerprint


def _state(roster, journals):
    return StateFingerprint(roster=tuple(roster), journal_sha256=dict(journals), journal_bytes={k: 10 for k in journals})


def _dispatch(name: str) -> ObservedToolCall:
    return ObservedToolCall(name="send_message_to_agent", arguments={"agent_name": name, "instructions": "do it"})


def test_existing_name_and_one_appended_journal_infers_reuse() -> None:
    before = _state(["Known"], {"known.log": "a"})
    after = _state(["Known"], {"known.log": "b"})

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
    after = _state(["A", "B"], {"a.log": "2", "b.log": "2"})

    result = infer_baseline_outcome(before, after, [_dispatch("A")])

    assert result.action == "unobservable"
    assert result.name is None
    assert "multiple" in result.reason


def test_truncated_journal_is_failed_observation() -> None:
    before = StateFingerprint(roster=("Known",), journal_sha256={"known.log": "a"}, journal_bytes={"known.log": 100})
    after = StateFingerprint(roster=("Known",), journal_sha256={"known.log": "b"}, journal_bytes={"known.log": 20})

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
    )
    after = StateFingerprint(
        roster=("Known",),
        journal_sha256={"known.log": "b"},
        journal_bytes={"known.log": 120},
        journal_complete={"known.log": False},
    )

    result = infer_baseline_outcome(before, after, [_dispatch("Known")])

    assert result.action == "unobservable"
    assert result.failed
    assert "partial" in result.reason
