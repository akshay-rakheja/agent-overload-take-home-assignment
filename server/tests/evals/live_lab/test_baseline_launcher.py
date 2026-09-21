from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import sys
from pathlib import Path

import pytest

from evals.live_lab.baseline_launcher import ObservationSink, wrap_async_call, wrap_sync_call
from evals.live_lab.baseline_observer import run_baseline_turn
from evals.live_lab.fixtures import build_fixture_manifest, materialize_baseline
from evals.live_lab.processes import ManagedProcess
from evals.live_lab.raw_observation import BaselineTurnRequest


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASELINE_WORKTREE = PROJECT_ROOT.parent / "openpoke-evaluation-baseline"


class _FailingSink:
    def __init__(self, *, fail_store: bool = False, fail_append: bool = False) -> None:
        self.fail_store = fail_store
        self.fail_append = fail_append

    def store_private(self, value):
        if self.fail_store:
            raise OSError("private storage unavailable")
        return "0" * 64

    def append(self, event):
        if self.fail_append:
            raise OSError("event storage unavailable")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fake_response_queue(agent_name: str) -> list[dict]:
    return [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "dispatch-1",
                                "type": "function",
                                "function": {
                                    "name": "send_message_to_agent",
                                    "arguments": json.dumps(
                                        {"agent_name": agent_name, "instructions": "inspect fixture"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "fixture execution complete"}}]},
        {"choices": [{"message": {"content": "fixture turn complete"}}]},
    ]


_FAKE_SERVER = r'''
import json, socket, sys, time
host, port, queue_path = "127.0.0.1", int(sys.argv[1]), sys.argv[2]
queue = json.loads(open(queue_path, encoding="utf-8").read())
last_response = queue[-1]
index = 0
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((host, port))
server.listen()
while True:
    connection, _ = server.accept()
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    first = head.split(b"\r\n", 1)[0]
    length = 0
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    while len(body) < length:
        body += connection.recv(65536)
    if first.startswith(b"GET "):
        payload = b"{}"
        status = b"204 No Content"
    else:
        request_payload = json.loads(body[:length] or b"{}")
        system_text = ""
        non_system_text = ""
        for message in request_payload.get("messages", []):
            if message.get("role") == "system":
                system_text = str(message.get("content", ""))
            else:
                non_system_text += str(message.get("content", ""))
        if "Agent Name:" in system_text:
            component = "execution"
        elif "<new_agent_message>" in non_system_text:
            component = "interaction_agent"
        else:
            component = "interaction_user"
        match = next(
            (item for item in queue if item.get("_component") in (None, component)),
            last_response,
        )
        response = match
        if not response.get("_repeat") and response in queue:
            queue.remove(response)
        time.sleep(float(response.get("_delay", 0)))
        status = str(response.get("_status", "200 OK")).encode()
        payload = json.dumps(response.get("_body", response)).encode()
    headers = (
        b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
        + b"Content-Length: " + str(len(payload)).encode() + b"\r\nConnection: close\r\n\r\n"
    )
    try:
        connection.sendall(headers + payload)
    except OSError:
        pass
    connection.close()
'''


def _process_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "127.0.0.1,localhost",
    }


def _run_real_turn(
    tmp_path: Path,
    *,
    size: int,
    selected_name: str,
    responses: list[dict] | None = None,
    timeout_seconds: float = 10,
    late_grace_seconds: float = 0,
    execution_timeout_seconds: float | None = None,
    run_second_turn: bool = False,
):
    manifest = build_fixture_manifest(seed=1313, roster_size=size)
    data_dir = tmp_path / "fixture" / "server" / "data"
    materialize_baseline(manifest, data_dir)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    queue_path = tmp_path / "responses.json"
    queue_path.write_text(
        json.dumps(responses or _fake_response_queue(selected_name)),
        encoding="utf-8",
    )
    fake_port = _free_port()
    readiness_nonce = secrets.token_hex(16)
    fake = ManagedProcess(
        argv=[sys.executable, "-c", _FAKE_SERVER, str(fake_port), str(queue_path)],
        env=_process_env(),
        cwd=tmp_path,
        pid_file=tmp_path / "fake.pid",
        stdout_path=tmp_path / "fake.out",
        stderr_path=tmp_path / "fake.err",
    )
    baseline = ManagedProcess(
        argv=[
            sys.executable,
            "-m",
            "evals.live_lab.baseline_launcher",
            "--worktree",
            str(BASELINE_WORKTREE),
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--fake-base-url",
            f"http://127.0.0.1:{fake_port}/v1",
            "--readiness-nonce",
            readiness_nonce,
        ]
        + (
            ["--execution-timeout-seconds", str(execution_timeout_seconds)]
            if execution_timeout_seconds is not None
            else []
        ),
        env=_process_env(),
        cwd=PROJECT_ROOT,
        pid_file=tmp_path / "baseline.pid",
        stdout_path=tmp_path / "baseline.out",
        stderr_path=tmp_path / "baseline.err",
        private_dir=run_dir / "private" / "process-streams",
        readiness_host="127.0.0.1",
        readiness_port=8001,
        readiness_nonce=readiness_nonce,
    )
    fake.start()
    fake.wait_ready(f"http://127.0.0.1:{fake_port}/health", timeout=5)
    baseline.start()
    try:
        # A cold checkout can spend over a minute importing individual historical
        # modules from the host filesystem. This is one bounded launch, not a retry.
        baseline.wait_ready(
            f"http://127.0.0.1:8001/__live_lab_ready__/{readiness_nonce}",
            timeout=120,
        )
        observation = asyncio.run(
            run_baseline_turn(
                BaselineTurnRequest(
                    run_id=f"size-{size}",
                    data_dir=str(data_dir),
                    event_path=str(run_dir / "events.jsonl"),
                    user_message="inspect the fixture",
                    timeout_seconds=timeout_seconds,
                    late_grace_seconds=late_grace_seconds,
                )
            )
        )
        if run_second_turn:
            second = asyncio.run(
                run_baseline_turn(
                    BaselineTurnRequest(
                        run_id=f"size-{size}-second",
                        data_dir=str(data_dir),
                        event_path=str(run_dir / "events.jsonl"),
                        user_message="second turn must not consume late work",
                        timeout_seconds=0.2,
                    )
                )
            )
            return manifest, observation, second, run_dir
        return manifest, observation, run_dir
    finally:
        baseline.stop(timeout=5)
        fake.stop(timeout=5)


def test_sync_wrapper_calls_original_once_and_returns_same_object(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    returned = [{"role": "user", "content": "private mailbox text"}]
    calls = 0

    def original(value):
        nonlocal calls
        calls += 1
        return returned

    wrapped = wrap_sync_call("interaction_prompt", original, sink)

    assert wrapped("input") is returned
    assert calls == 1
    assert "private mailbox text" not in (tmp_path / "events.jsonl").read_text()
    assert any((tmp_path / "private").iterdir())


def test_async_wrapper_calls_original_once_and_reraises_same_error(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    error = RuntimeError("context limit: private mailbox body")
    calls = 0

    async def original(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    wrapped = wrap_async_call("execution_model", original, sink)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(wrapped(model="fake-model", messages=[]))

    assert caught.value is error
    assert calls == 1
    exported = (tmp_path / "events.jsonl").read_text()
    assert "private mailbox body" not in exported
    assert "error_sha256" in exported


def test_tool_wrapper_exports_name_but_hashes_sensitive_arguments(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    wrapped = wrap_sync_call("interaction_tool", lambda *_: object(), sink)

    wrapped(
        "send_message_to_agent",
        {"agent_name": "Known", "instructions": "private mailbox body"},
    )
    event = sink.read_events()[0]

    assert event["tool_call"]["arguments"]["agent_name"] == "Known"
    assert event["tool_call"]["arguments"]["instructions"]["characters"] == 20
    assert "private mailbox body" not in (tmp_path / "events.jsonl").read_text()


def test_interaction_prompt_event_has_full_roster_exposure(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    names = tuple(f"Agent {index:04d}" for index in range(1000))
    prompt = "<active_agents>\n" + "\n".join(f'<agent name="{name}" />' for name in names) + "\n</active_agents>"
    wrapped = wrap_sync_call("interaction_prompt", lambda *_: [{"role": "user", "content": prompt}], sink)

    wrapped("turn", "history")
    events = sink.read_events()

    assert events[0]["exposed_names"] == list(names)
    assert events[0]["prompt_characters"] == len(prompt)
    assert events[0]["prompt_sha256"]


def test_execution_prompt_records_unbounded_history_size(tmp_path: Path) -> None:
    sink = ObservationSink(tmp_path / "events.jsonl", tmp_path / "private")
    short = wrap_sync_call("execution_prompt", lambda: "system\n# Execution History\nold", sink)
    long = wrap_sync_call("execution_prompt", lambda: "system\n# Execution History\n" + "entry\n" * 10_000, sink)

    short()
    long()
    events = sink.read_events()

    assert events[1]["prompt_characters"] > events[0]["prompt_characters"]
    assert events[1]["prompt_characters"] > 50_000


def test_async_pre_call_observation_failure_does_not_prevent_original() -> None:
    returned = object()
    calls = 0

    async def original(**kwargs):
        nonlocal calls
        calls += 1
        return returned

    wrapped = wrap_async_call("interaction_model", original, _FailingSink(fail_store=True))

    assert asyncio.run(wrapped(model="fake", messages=[])) is returned
    assert calls == 1


def test_async_success_observation_failure_does_not_replace_return() -> None:
    returned = {"choices": []}
    wrapped = wrap_async_call(
        "interaction_model",
        lambda **kwargs: asyncio.sleep(0, result=returned),
        _FailingSink(fail_append=True),
    )

    assert asyncio.run(wrapped(model="fake", messages=[])) is returned


def test_async_observation_failure_does_not_replace_original_exception() -> None:
    original_error = RuntimeError("original failure")

    async def original(**kwargs):
        raise original_error

    wrapped = wrap_async_call("interaction_model", original, _FailingSink(fail_store=True))

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(wrapped(model="fake", messages=[]))
    assert caught.value is original_error


def test_sync_observation_failure_does_not_replace_return_or_exception() -> None:
    returned = object()
    assert wrap_sync_call(
        "interaction_prompt",
        lambda: returned,
        _FailingSink(fail_store=True),
    )() is returned

    original_error = RuntimeError("sync original failure")

    def fail():
        raise original_error

    with pytest.raises(RuntimeError) as caught:
        wrap_sync_call("interaction_prompt", fail, _FailingSink(fail_append=True))()
    assert caught.value is original_error


@pytest.mark.parametrize(
    ("size", "selection", "expected_action"),
    [
        (10, "AI Video Newsletter Curator", "reuse"),
        (100, "Calendar Deadline Tracker", "create_new"),
        (500, "Instagram Security Monitor", "reuse"),
        (1000, "Instagram Engagement Digest", "reuse"),
    ],
)
def test_real_historical_path_exposes_full_roster_and_infers_action(
    tmp_path: Path,
    size: int,
    selection: str,
    expected_action: str,
) -> None:
    manifest, observation, _ = _run_real_turn(
        tmp_path,
        size=size,
        selected_name=selection,
    )

    assert observation.exposed_names == tuple(agent.name for agent in manifest.agents)
    assert observation.inferred_action == expected_action
    assert observation.inferred_name == selection
    assert observation.final_response in {"fixture execution complete", "fixture turn complete"}
    assert not observation.errors


def test_real_historical_execution_prompt_keeps_complete_seeded_history(tmp_path: Path) -> None:
    _, observation, run_dir = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
    )
    events = ObservationSink(run_dir / "events.jsonl", run_dir / "private").read_events()
    execution_prompt = next(event for event in events if event["kind"] == "execution_prompt")
    private_prompt = (run_dir / "private" / f"{execution_prompt['prompt_sha256']}.json").read_text()

    assert observation.inferred_action == "reuse"
    assert execution_prompt["prompt_characters"] > 500_000
    assert "DEPTH-START-SENTINEL" in private_prompt
    assert "DEPTH-END-SENTINEL" in private_prompt


def test_real_historical_context_limit_is_preserved_as_failure(tmp_path: Path) -> None:
    responses = [
        {
            "_status": "400 Bad Request",
            "_body": {"error": "context length exceeded"},
        }
    ]

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=0.3,
    )

    assert observation.inferred_action == "unobservable"
    assert observation.final_response is None
    assert {error.code for error in observation.errors} == {"timeout", "model_error"}
    assert observation.raw_model_calls[0].error_type == "OpenRouterError"


def test_real_historical_late_response_is_not_promoted_to_success(tmp_path: Path) -> None:
    responses = _fake_response_queue("AI Video Newsletter Curator")
    responses[0] = {"_delay": 0.35, "_body": responses[0]}

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=0.1,
        late_grace_seconds=2,
    )

    assert observation.inferred_action == "unobservable"
    assert observation.final_response is not None
    assert {error.code for error in observation.errors} >= {"timeout", "late_response"}


def test_late_failed_turn_taints_context_and_cannot_satisfy_next_turn(tmp_path: Path) -> None:
    responses = _fake_response_queue("AI Video Newsletter Curator")
    responses[0] = {"_delay": 0.35, "_body": responses[0]}

    _, first, second, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=0.1,
        late_grace_seconds=2,
        run_second_turn=True,
    )

    assert first.inferred_action == "unobservable"
    assert any(error.code == "late_response" for error in first.errors)
    assert second.inferred_action == "unobservable"
    assert second.final_response is None
    assert any(error.code == "tainted_process" for error in second.errors)


def test_real_historical_malformed_tool_args_remain_inconclusive(tmp_path: Path) -> None:
    responses = _fake_response_queue("AI Video Newsletter Curator")
    responses[0]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{bad-json"
    responses[1] = {"choices": [{"message": {"content": "Please clarify the workflow."}}]}

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
    )

    assert observation.inferred_action == "unobservable"
    assert observation.inferred_name is None
    assert any(error.code == "failed_observation" for error in observation.errors)


def test_real_historical_execution_timeout_is_preserved_as_failure(tmp_path: Path) -> None:
    dispatch = _fake_response_queue("AI Video Newsletter Curator")[0]
    wait_response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "wait-1",
                            "type": "function",
                            "function": {
                                "name": "wait",
                                "arguments": json.dumps({"reason": "await execution"}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    responses = [
        {"_component": "interaction_user", "_body": dispatch},
        {
            "_component": "interaction_user",
            "_repeat": True,
            "_body": wait_response,
        },
        {
            "_component": "execution",
            "_delay": 0.5,
            "_body": {"choices": [{"message": {"content": "too late"}}]},
        },
        {
            "_component": "interaction_agent",
            "_body": {"choices": [{"message": {"content": "execution timeout preserved"}}]},
        },
    ]

    _, observation, _ = _run_real_turn(
        tmp_path,
        size=10,
        selected_name="AI Video Newsletter Curator",
        responses=responses,
        timeout_seconds=3,
        execution_timeout_seconds=0.1,
    )

    assert observation.inferred_action == "unobservable"
    assert any(
        error.phase == "execution" and error.code == "model_error"
        for error in observation.errors
    )
