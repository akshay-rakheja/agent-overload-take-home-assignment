"""External observation of the historical backend's visible state changes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from .raw_observation import (
    BaselineInference,
    BaselineObservation,
    BaselineTurnRequest,
    ObservedError,
    ObservedToolCall,
    RawModelCall,
    StateFingerprint,
)


_COMPLETE_JOURNAL_LINE = re.compile(rb"^<([a-z_]+)(?: [^>]*)?>.*</\1>$")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def snapshot_baseline_state(data_dir: Path) -> StateFingerprint:
    execution_dir = data_dir / "execution_agents"
    roster_path = execution_dir / "roster.json"
    raw_roster = json.loads(roster_path.read_text(encoding="utf-8"))
    if not isinstance(raw_roster, list) or not all(isinstance(name, str) for name in raw_roster):
        raise ValueError("historical roster must be a list of names")
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    complete: dict[str, bool] = {}
    for journal in sorted(execution_dir.glob("*.log")):
        payload = journal.read_bytes()
        hashes[journal.name] = _sha256(payload)
        sizes[journal.name] = len(payload)
        lines = payload.splitlines()
        complete[journal.name] = payload.endswith(b"\n") and all(
            _COMPLETE_JOURNAL_LINE.match(line) is not None for line in lines
        )
    return StateFingerprint(
        roster=tuple(raw_roster),
        journal_sha256=hashes,
        journal_bytes=sizes,
        journal_complete=complete,
    )


def infer_baseline_outcome(
    before: StateFingerprint,
    after: StateFingerprint,
    tool_calls: Sequence[ObservedToolCall],
) -> BaselineInference:
    """Infer only what roster/journal deltas and explicit dispatch calls prove."""

    dispatches = [call for call in tool_calls if call.name == "send_message_to_agent"]
    if not dispatches:
        return BaselineInference(action="abstain", name=None, reason="no dispatch tool call observed")
    if any(call.malformed for call in dispatches):
        return BaselineInference(
            action="unobservable",
            name=None,
            reason="dispatch contained malformed tool arguments",
            failed=True,
        )
    names = [call.arguments.get("agent_name") for call in dispatches]
    if len(dispatches) != 1 or not isinstance(names[0], str) or not names[0]:
        return BaselineInference(
            action="unobservable",
            name=None,
            reason="dispatch name is not uniquely observable",
            failed=True,
        )

    truncated = [
        path
        for path, old_size in before.journal_bytes.items()
        if after.journal_bytes.get(path, 0) < old_size
    ]
    incomplete = [
        path
        for path, is_complete in after.journal_complete.items()
        if not is_complete
    ]
    if truncated or incomplete:
        detail = "truncated" if truncated else "partial"
        return BaselineInference(
            action="unobservable",
            name=None,
            reason=f"failed observation: {detail} journal write",
            failed=True,
        )

    changed = sorted(
        path
        for path in set(before.journal_sha256) | set(after.journal_sha256)
        if before.journal_sha256.get(path) != after.journal_sha256.get(path)
    )
    added_names = [name for name in after.roster if name not in before.roster]
    removed_names = [name for name in before.roster if name not in after.roster]
    selected_name = names[0]
    if len(changed) > 1:
        return BaselineInference(
            action="unobservable",
            name=None,
            reason="multiple plausible journals changed",
        )
    if removed_names or len(added_names) > 1:
        return BaselineInference(
            action="unobservable",
            name=None,
            reason="roster mutation is not a single baseline creation",
            failed=True,
        )
    if added_names:
        if added_names == [selected_name] and len(changed) == 1:
            return BaselineInference(
                action="create_new",
                name=selected_name,
                reason="one dispatched name was added to roster with one journal append",
            )
        return BaselineInference(
            action="unobservable",
            name=None,
            reason="new roster name does not match explicit dispatch evidence",
        )
    if selected_name in before.roster and len(changed) == 1:
        return BaselineInference(
            action="reuse",
            name=selected_name,
            reason="existing dispatched name has one appended journal",
        )
    return BaselineInference(
        action="unobservable",
        name=None,
        reason="dispatch did not produce a unique roster and journal delta",
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _tool_calls(events: Sequence[dict[str, Any]]) -> list[ObservedToolCall]:
    return [
        ObservedToolCall.model_validate(event["tool_call"])
        for event in events
        if event.get("kind") == "tool_call" and isinstance(event.get("tool_call"), dict)
    ]


def _model_calls(events: Sequence[dict[str, Any]]) -> list[RawModelCall]:
    return [
        RawModelCall.model_validate(event["model_call"])
        for event in events
        if event.get("kind") == "model_call" and isinstance(event.get("model_call"), dict)
    ]


async def _history(client: httpx.AsyncClient, base_url: str) -> list[dict[str, Any]]:
    response = await client.get(f"{base_url.rstrip('/')}/chat/history")
    response.raise_for_status()
    payload = response.json()
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    return [message for message in messages if isinstance(message, dict)]


async def run_baseline_turn(request: BaselineTurnRequest) -> BaselineObservation:
    """Submit one real baseline turn, then externally infer its outcome."""

    data_dir = Path(request.data_dir)
    event_path = Path(request.event_path)
    before = snapshot_baseline_state(data_dir)
    initial_events = len(_read_events(event_path))
    errors: list[ObservedError] = []
    final_response: str | None = None
    timed_out = False

    async with httpx.AsyncClient(timeout=min(request.timeout_seconds, 10.0)) as client:
        history_before = await _history(client, request.base_url)
        assistant_count = sum(message.get("role") == "assistant" for message in history_before)
        response = await client.post(
            f"{request.base_url.rstrip('/')}/chat/send",
            json={"messages": [{"role": "user", "content": request.user_message}], "stream": False},
        )
        response.raise_for_status()
        deadline = time.monotonic() + request.timeout_seconds
        while time.monotonic() < deadline:
            current = await _history(client, request.base_url)
            replies = [message for message in current if message.get("role") == "assistant"]
            if len(replies) > assistant_count:
                final_response = str(replies[-1].get("content") or "")
                break
            await asyncio.sleep(request.poll_interval_seconds)
        else:
            timed_out = True
            errors.append(
                ObservedError(
                    phase="interaction",
                    code="timeout",
                    message="baseline turn did not produce a response before the deadline",
                )
            )
        if timed_out and request.late_grace_seconds:
            late_deadline = time.monotonic() + request.late_grace_seconds
            while time.monotonic() < late_deadline:
                current = await _history(client, request.base_url)
                replies = [message for message in current if message.get("role") == "assistant"]
                if len(replies) > assistant_count:
                    final_response = str(replies[-1].get("content") or "")
                    errors.append(
                        ObservedError(
                            phase="interaction",
                            code="late_response",
                            message="response arrived after the observer timeout",
                            late=True,
                        )
                    )
                    break
                await asyncio.sleep(request.poll_interval_seconds)

    after = snapshot_baseline_state(data_dir)
    events = _read_events(event_path)[initial_events:]
    calls = _tool_calls(events)
    model_calls = _model_calls(events)
    for model_call in model_calls:
        if model_call.error_type:
            errors.append(
                ObservedError(
                    phase=model_call.component,
                    code="model_error",
                    message=f"{model_call.component} model call failed: {model_call.error_type}",
                )
            )
    inference = infer_baseline_outcome(before, after, calls)
    if inference.failed:
        errors.append(
            ObservedError(
                phase="state",
                code="failed_observation",
                message=inference.reason,
                partial="partial" in inference.reason,
            )
        )
    if errors:
        inference = BaselineInference(
            action="unobservable",
            name=None,
            reason="baseline outcome is inconclusive because observer errors were preserved",
            failed=True,
        )
    prompt_event = next(
        (event for event in events if event.get("kind") == "interaction_prompt"),
        {},
    )
    return BaselineObservation(
        run_id=request.run_id,
        prompt_xml_sha256=str(prompt_event.get("prompt_sha256") or ""),
        prompt_characters=int(prompt_event.get("prompt_characters") or 0),
        exposed_names=tuple(str(name) for name in prompt_event.get("exposed_names", [])),
        roster_before=before.roster,
        roster_after=after.roster,
        journal_hashes_before=before.journal_sha256,
        journal_hashes_after=after.journal_sha256,
        inferred_action=inference.action,
        inferred_name=inference.name,
        inference_reason=inference.reason,
        final_response=final_response,
        raw_model_calls=tuple(model_calls),
        errors=tuple(errors),
    )


__all__ = ["infer_baseline_outcome", "run_baseline_turn", "snapshot_baseline_state"]
