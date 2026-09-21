"""External observation of the historical backend's visible state changes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
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
    contents: dict[str, bytes] = {}
    for journal in sorted(execution_dir.glob("*.log")):
        payload = journal.read_bytes()
        hashes[journal.name] = _sha256(payload)
        sizes[journal.name] = len(payload)
        lines = payload.splitlines()
        complete[journal.name] = payload.endswith(b"\n") and all(
            _COMPLETE_JOURNAL_LINE.match(line) is not None for line in lines
        )
        contents[journal.name] = payload
    return StateFingerprint(
        roster=tuple(raw_roster),
        journal_sha256=hashes,
        journal_bytes=sizes,
        journal_complete=complete,
        journal_contents=contents,
    )


def _historical_slug(name: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in name.strip()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "agent"


def infer_baseline_outcome(
    before: StateFingerprint,
    after: StateFingerprint,
    tool_calls: Sequence[ObservedToolCall],
) -> BaselineInference:
    """Infer only what roster/journal deltas and explicit dispatch calls prove."""

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
    missing = sorted(set(before.journal_sha256) - set(after.journal_sha256))
    if missing:
        return BaselineInference(
            action="unobservable",
            name=None,
            reason="failed observation: journal removed or truncated",
            failed=True,
        )
    for path in changed:
        if path in before.journal_sha256 and path not in before.journal_contents:
            return BaselineInference(
                action="unobservable",
                name=None,
                reason="failed observation: prior journal bytes unavailable for append-only proof",
                failed=True,
            )
        if after.journal_complete.get(path) is not True:
            return BaselineInference(
                action="unobservable",
                name=None,
                reason="failed observation: journal completeness evidence unavailable",
                failed=True,
            )
        old_payload = before.journal_contents.get(path, b"")
        new_payload = after.journal_contents.get(path)
        if new_payload is None:
            return BaselineInference(
                action="unobservable",
                name=None,
                reason="failed observation: append-only journal evidence unavailable",
                failed=True,
            )
        if old_payload and not new_payload.startswith(old_payload):
            return BaselineInference(
                action="unobservable",
                name=None,
                reason="failed observation: journal change was not append-only",
                failed=True,
            )

    added_names = [name for name in after.roster if name not in before.roster]
    removed_names = [name for name in before.roster if name not in after.roster]
    roster_changed = before.roster != after.roster
    dispatches = [call for call in tool_calls if call.name == "send_message_to_agent"]
    if not dispatches:
        if changed or roster_changed:
            return BaselineInference(
                action="unobservable",
                name=None,
                reason="state mutated without an observed dispatch",
                failed=True,
            )
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
        expected_path = f"{_historical_slug(selected_name)}.log"
        owners = [name for name in after.roster if _historical_slug(name) == _historical_slug(selected_name)]
        if len(owners) != 1:
            return BaselineInference(
                action="unobservable",
                name=None,
                reason="selected name does not map to a unique historical journal",
            )
        if (
            after.roster == (*before.roster, selected_name)
            and added_names == [selected_name]
            and changed == [expected_path]
        ):
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
    expected_path = f"{_historical_slug(selected_name)}.log"
    owners = [name for name in before.roster if _historical_slug(name) == _historical_slug(selected_name)]
    if len(owners) != 1:
        return BaselineInference(
            action="unobservable",
            name=None,
            reason="selected name does not map to a unique historical journal",
        )
    if (
        before.roster == after.roster
        and before.roster.count(selected_name) == 1
        and changed == [expected_path]
    ):
        return BaselineInference(
            action="reuse",
            name=selected_name,
            reason="existing dispatched name has one appended journal",
        )
    return BaselineInference(
        action="unobservable",
        name=None,
        reason="dispatch changed an unrelated journal or lacked selected journal evidence",
    )


def _failed_observation(
    request: BaselineTurnRequest,
    *,
    code: str,
    phase: str,
    message: str,
    before: StateFingerprint | None = None,
    after: StateFingerprint | None = None,
    errors: Sequence[ObservedError] = (),
) -> BaselineObservation:
    before = before or StateFingerprint(roster=(), journal_sha256={}, journal_bytes={})
    after = after or before
    return BaselineObservation(
        run_id=request.run_id,
        prompt_xml_sha256="",
        prompt_characters=0,
        exposed_names=(),
        roster_before=before.roster,
        roster_after=after.roster,
        journal_hashes_before=before.journal_sha256,
        journal_hashes_after=after.journal_sha256,
        inferred_action="unobservable",
        inferred_name=None,
        inference_reason="baseline observation failed before trustworthy evidence was available",
        final_response=None,
        raw_model_calls=(),
        errors=(*errors, ObservedError(phase=phase, code=code, message=message)),
    )


def _read_events(path: Path) -> tuple[list[dict[str, Any]], list[ObservedError]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], []
    except (OSError, UnicodeError) as exc:
        return [], [
            ObservedError(
                phase="events",
                code="event_read_error",
                message=f"unable to read observation events: {type(exc).__name__}",
            )
        ]
    events: list[dict[str, Any]] = []
    errors: list[ObservedError] = []
    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeError):
            errors.append(
                ObservedError(
                    phase="events",
                    code="event_corruption",
                    message=f"observation event line {line_number} is corrupt",
                    partial=True,
                )
            )
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            errors.append(
                ObservedError(
                    phase="events",
                    code="event_corruption",
                    message=f"observation event line {line_number} is not an object",
                    partial=True,
                )
            )
    return events, errors


def _tool_calls(
    events: Sequence[dict[str, Any]],
) -> tuple[list[ObservedToolCall], list[ObservedError]]:
    calls: list[ObservedToolCall] = []
    errors: list[ObservedError] = []
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        try:
            calls.append(ObservedToolCall.model_validate(event.get("tool_call")))
        except Exception as exc:
            errors.append(
                ObservedError(
                    phase="events",
                    code="event_corruption",
                    message=f"tool-call evidence is invalid: {type(exc).__name__}",
                    partial=True,
                )
            )
    return calls, errors


def _model_calls(
    events: Sequence[dict[str, Any]],
) -> tuple[list[RawModelCall], list[ObservedError]]:
    calls: list[RawModelCall] = []
    errors: list[ObservedError] = []
    for event in events:
        if event.get("kind") != "model_call":
            continue
        try:
            calls.append(RawModelCall.model_validate(event.get("model_call")))
        except Exception as exc:
            errors.append(
                ObservedError(
                    phase="events",
                    code="event_corruption",
                    message=f"model-call evidence is invalid: {type(exc).__name__}",
                    partial=True,
                )
            )
    return calls, errors


async def _history(client: httpx.AsyncClient, base_url: str) -> list[dict[str, Any]]:
    response = await client.get(f"{base_url.rstrip('/')}/chat/history")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError("historical history response has no message list")
    messages = payload["messages"]
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("historical history contains an invalid message")
    return messages


_TURN_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


def _process_nonce(run_dir: Path) -> str:
    payload = json.loads((run_dir / "process_context.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("process_nonce"), str):
        raise ValueError("process context has no nonce")
    return payload["process_nonce"]


def _taint_matches(run_dir: Path, process_nonce: str) -> bool:
    try:
        payload = json.loads((run_dir / "tainted.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True
    return not isinstance(payload, dict) or payload.get("process_nonce") == process_nonce


def _mark_tainted(run_dir: Path, process_nonce: str, run_id: str, reason: str) -> None:
    payload = {
        "process_nonce": process_nonce,
        "reason": reason,
        "run_id": run_id,
    }
    target = run_dir / "tainted.json"
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _claim_turn(run_dir: Path, process_nonce: str, run_id: str) -> str:
    token = secrets.token_hex(16)
    payload = json.dumps(
        {"owner_token": token, "process_nonce": process_nonce, "run_id": run_id},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    descriptor = os.open(
        run_dir / "active_turn.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return token


def _release_turn(run_dir: Path, owner_token: str) -> None:
    target = run_dir / "active_turn.json"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return
    if isinstance(payload, dict) and payload.get("owner_token") == owner_token:
        target.unlink(missing_ok=True)


async def run_baseline_turn(request: BaselineTurnRequest) -> BaselineObservation:
    """Submit one real baseline turn, then externally infer its outcome."""

    data_dir = Path(request.data_dir)
    event_path = Path(request.event_path)
    run_dir = event_path.parent
    try:
        before = snapshot_baseline_state(data_dir)
    except Exception as exc:
        return _failed_observation(
            request,
            code="state_read_error",
            phase="snapshot_before",
            message=f"unable to read baseline state: {type(exc).__name__}",
        )
    try:
        process_nonce = _process_nonce(run_dir)
    except Exception as exc:
        return _failed_observation(
            request,
            code="process_context_error",
            phase="ownership",
            message=f"unable to verify baseline process context: {type(exc).__name__}",
            before=before,
        )

    loop = asyncio.get_running_loop()
    lock_key = (id(loop), str(event_path.resolve()))
    turn_lock = _TURN_LOCKS.setdefault(lock_key, asyncio.Lock())
    async with turn_lock:
        if _taint_matches(run_dir, process_nonce):
            return _failed_observation(
                request,
                code="tainted_process",
                phase="ownership",
                message="baseline process requires a fresh verified launch after an inconclusive turn",
                before=before,
            )
        _, initial_event_errors = _read_events(event_path)
        if initial_event_errors:
            _mark_tainted(run_dir, process_nonce, request.run_id, "preexisting event corruption")
            return _failed_observation(
                request,
                code="event_corruption",
                phase="events",
                message="preexisting observation evidence is corrupt",
                before=before,
                errors=initial_event_errors[:-1],
            )
        try:
            owner_token = _claim_turn(run_dir, process_nonce, request.run_id)
        except FileExistsError:
            return _failed_observation(
                request,
                code="concurrent_turn",
                phase="ownership",
                message="another measured turn owns this baseline process",
                before=before,
            )
        except Exception as exc:
            return _failed_observation(
                request,
                code="turn_claim_error",
                phase="ownership",
                message=f"unable to claim measured turn: {type(exc).__name__}",
                before=before,
            )

        errors: list[ObservedError] = []
        final_response: str | None = None
        timed_out = False
        submitted = False
        try:
            try:
                async with httpx.AsyncClient(timeout=min(request.timeout_seconds, 10.0)) as client:
                    history_before = await _history(client, request.base_url)
                    assistant_count = sum(
                        message.get("role") == "assistant" for message in history_before
                    )
                    response = await client.post(
                        f"{request.base_url.rstrip('/')}/chat/send",
                        json={
                            "messages": [{"role": "user", "content": request.user_message}],
                            "stream": False,
                        },
                    )
                    submitted = True
                    response.raise_for_status()
                    deadline = time.monotonic() + request.timeout_seconds
                    while time.monotonic() < deadline:
                        current = await _history(client, request.base_url)
                        replies = [
                            message for message in current if message.get("role") == "assistant"
                        ]
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
                            replies = [
                                message
                                for message in current
                                if message.get("role") == "assistant"
                            ]
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
            except Exception as exc:
                errors.append(
                    ObservedError(
                        phase="http",
                        code="http_error",
                        message=f"baseline HTTP operation failed: {type(exc).__name__}",
                    )
                )

            try:
                after = snapshot_baseline_state(data_dir)
            except Exception as exc:
                errors.append(
                    ObservedError(
                        phase="snapshot_after",
                        code="state_read_error",
                        message=f"unable to read baseline state: {type(exc).__name__}",
                        partial=True,
                    )
                )
                after = before

            all_events, event_errors = _read_events(event_path)
            errors.extend(event_errors)
            events = [event for event in all_events if event.get("owner_token") == owner_token]
            calls, call_errors = _tool_calls(events)
            model_calls, model_errors = _model_calls(events)
            errors.extend(call_errors)
            errors.extend(model_errors)
            for model_call in model_calls:
                if model_call.error_type:
                    errors.append(
                        ObservedError(
                            phase=model_call.component,
                            code="model_error",
                            message=(
                                f"{model_call.component} model call failed: "
                                f"{model_call.error_type}"
                            ),
                        )
                    )
            prompt_event = next(
                (event for event in events if event.get("kind") == "interaction_prompt"),
                {},
            )
            if submitted and not events:
                errors.append(
                    ObservedError(
                        phase="events",
                        code="missing_evidence",
                        message="submitted baseline turn produced no owned observation evidence",
                    )
                )
            elif submitted and not prompt_event:
                errors.append(
                    ObservedError(
                        phase="events",
                        code="missing_evidence",
                        message="submitted baseline turn has no owned prompt evidence",
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
            observation = BaselineObservation(
                run_id=request.run_id,
                prompt_xml_sha256=str(prompt_event.get("prompt_sha256") or ""),
                prompt_characters=int(prompt_event.get("prompt_characters") or 0),
                exposed_names=tuple(
                    str(name) for name in prompt_event.get("exposed_names", [])
                ),
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
            if errors or inference.action == "unobservable":
                _mark_tainted(
                    run_dir,
                    process_nonce,
                    request.run_id,
                    errors[0].code if errors else "unobservable",
                )
            else:
                _release_turn(run_dir, owner_token)
            return observation
        except Exception as exc:
            _mark_tainted(run_dir, process_nonce, request.run_id, "observer_error")
            return _failed_observation(
                request,
                code="observer_error",
                phase="observer",
                message=f"baseline observer failed safely: {type(exc).__name__}",
                before=before,
            )


__all__ = ["infer_baseline_outcome", "run_baseline_turn", "snapshot_baseline_state"]
