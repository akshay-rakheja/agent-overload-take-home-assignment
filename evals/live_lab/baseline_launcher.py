"""Launch the pinned historical app with in-memory observation wrappers."""

from __future__ import annotations

import argparse
import functools
import hashlib
import html
import importlib
import inspect
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .revisions import (
    APPROVED_OVERLAY_PATHS,
    HISTORICAL_BASE_SHA,
    verify_baseline_revision,
)


_AGENT_NAME = re.compile(r'<agent\s+name="([^"]*)"\s*/>')
_COMPONENTS = {
    "interaction_model": "interaction",
    "execution_model": "execution",
    "email_search_model": "email_search",
    "summarizer_model": "summarizer",
    "classifier_model": "classifier",
}


def _canonical(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        rendered = repr(value)
    return rendered.encode("utf-8", errors="replace")


class ObservationSink:
    """Write sanitized events and hashed private payloads to separate paths."""

    def __init__(
        self,
        event_path: Path,
        private_dir: Path,
        *,
        owner_path: Path | None = None,
        process_nonce: str | None = None,
    ) -> None:
        self.event_path = Path(event_path)
        self.private_dir = Path(private_dir)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.owner_path = Path(owner_path) if owner_path is not None else None
        self.process_nonce = process_nonce
        self._lock = threading.Lock()

    def store_private(self, value: Any) -> str:
        payload = _canonical(value)
        digest = hashlib.sha256(payload).hexdigest()
        target = self.private_dir / f"{digest}.json"
        if not target.exists():
            target.write_bytes(payload)
        return digest

    def append(self, event: Mapping[str, Any]) -> None:
        public_event = dict(event)
        if self.process_nonce is not None:
            public_event["process_nonce"] = self.process_nonce
        if self.owner_path is not None:
            try:
                owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
                owner = None
            if (
                isinstance(owner, dict)
                and owner.get("process_nonce") == self.process_nonce
                and isinstance(owner.get("owner_token"), str)
            ):
                public_event["owner_token"] = owner["owner_token"]
        line = json.dumps(public_event, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        try:
            lines = self.event_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines]


def _prompt_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        final = value[-1]
        if isinstance(final, dict) and isinstance(final.get("content"), str):
            return final["content"]
    return None


def _public_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
    sink: ObservationSink,
) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in arguments.items():
        if tool_name == "send_message_to_agent" and key == "agent_name" and isinstance(value, str):
            public[key] = value
        elif isinstance(value, str):
            public[key] = {
                "characters": len(value),
                "sha256": sink.store_private(value),
            }
        elif value is None or isinstance(value, (bool, int, float)):
            public[key] = value
        else:
            payload = _canonical(value)
            public[key] = {
                "bytes": len(payload),
                "sha256": sink.store_private(value),
            }
    return public


def _observation_failure(sink: ObservationSink, stage: str, exc: BaseException) -> None:
    """Best-effort metadata only; observation must never affect baseline control flow."""

    try:
        sink.append(
            {
                "kind": "observation_failure",
                "stage": stage,
                "error_type": type(exc).__name__,
            }
        )
    except BaseException:
        pass


def _best_effort_observe(
    sink: ObservationSink,
    stage: str,
    operation: Callable[[], Any],
) -> Any | None:
    try:
        return operation()
    except BaseException as exc:
        _observation_failure(sink, stage, exc)
        return None


def _sync_event(component: str, result: Any, elapsed_ms: float, sink: ObservationSink) -> dict[str, Any]:
    prompt = _prompt_text(result)
    digest = sink.store_private(result)
    if component == "interaction_prompt" and prompt is not None:
        return {
            "kind": "interaction_prompt",
            "elapsed_ms": elapsed_ms,
            "prompt_sha256": digest,
            "prompt_characters": len(prompt),
            "exposed_names": [html.unescape(name) for name in _AGENT_NAME.findall(prompt)],
        }
    if component == "active_roster" and prompt is not None:
        return {
            "kind": "active_roster",
            "elapsed_ms": elapsed_ms,
            "roster_sha256": digest,
            "roster_characters": len(prompt),
            "exposed_names": [html.unescape(name) for name in _AGENT_NAME.findall(prompt)],
        }
    if component == "execution_prompt" and prompt is not None:
        return {
            "kind": "execution_prompt",
            "elapsed_ms": elapsed_ms,
            "prompt_sha256": digest,
            "prompt_characters": len(prompt),
        }
    return {"kind": component, "elapsed_ms": elapsed_ms, "result_sha256": digest}


def wrap_sync_call(component: str, original: Callable[..., Any], sink: ObservationSink):
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            def observe_error() -> None:
                error_digest = sink.store_private(
                    {"error_type": type(exc).__name__, "message": str(exc)}
                )
                sink.append(
                    {
                        "kind": component,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "error_type": type(exc).__name__,
                        "error_sha256": error_digest,
                    }
                )

            _best_effort_observe(sink, f"{component}:exception", observe_error)
            raise
        elapsed = (time.perf_counter() - started) * 1000

        def observe_success() -> None:
            if component == "interaction_tool":
                name = args[0] if args else kwargs.get("name", "")
                arguments = args[1] if len(args) > 1 else kwargs.get("arguments", {})
                malformed = False
                raw_digest = sink.store_private(arguments)
                if isinstance(arguments, str):
                    try:
                        normalized = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        normalized = {}
                        malformed = True
                elif isinstance(arguments, dict):
                    normalized = arguments
                else:
                    normalized = {}
                    malformed = True
                sink.append(
                    {
                        "kind": "tool_call",
                        "elapsed_ms": elapsed,
                        "tool_call": {
                            "name": str(name),
                            "arguments": _public_tool_arguments(str(name), normalized, sink),
                            "raw_arguments_sha256": raw_digest,
                            "malformed": malformed,
                        },
                    }
                )
            else:
                sink.append(_sync_event(component, result, elapsed, sink))

        _best_effort_observe(sink, f"{component}:success", observe_success)
        return result

    return wrapped


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _response_tool_call_count(response: Any) -> int | None:
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
        return None
    count = 0
    for choice in response["choices"]:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            return None
        tool_calls = choice["message"].get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            return None
        count += len(tool_calls)
    return count


def wrap_async_call(component: str, original: Callable[..., Any], sink: ObservationSink):
    @functools.wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        request_digest = _best_effort_observe(
            sink,
            f"{component}:request",
            lambda: sink.store_private({"args": args, "kwargs": kwargs}),
        )
        if not isinstance(request_digest, str):
            request_digest = ""
        model_component = _COMPONENTS.get(component)
        try:
            result = original(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except BaseException as exc:
            elapsed = (time.perf_counter() - started) * 1000
            def observe_error() -> None:
                error_digest = sink.store_private(
                    {"error_type": type(exc).__name__, "message": str(exc)}
                )
                if model_component:
                    sink.append(
                        {
                            "kind": "model_call",
                            "model_call": {
                                "component": model_component,
                                "model": kwargs.get("model"),
                                "elapsed_ms": elapsed,
                                "request_sha256": request_digest,
                                "message_count": len(kwargs.get("messages") or []),
                                "tool_names": _tool_names(kwargs.get("tools")),
                                "error_type": type(exc).__name__,
                            },
                            "error_sha256": error_digest,
                        }
                    )
                else:
                    sink.append(
                        {
                            "kind": component,
                            "elapsed_ms": elapsed,
                            "error_type": type(exc).__name__,
                            "error_sha256": error_digest,
                        }
                    )

            _best_effort_observe(sink, f"{component}:exception", observe_error)
            raise
        elapsed = (time.perf_counter() - started) * 1000

        def observe_success() -> None:
            response_digest = sink.store_private(result)
            if model_component:
                choices = result.get("choices") if isinstance(result, dict) else None
                sink.append(
                    {
                        "kind": "model_call",
                        "model_call": {
                            "component": model_component,
                            "model": kwargs.get("model"),
                            "elapsed_ms": elapsed,
                            "request_sha256": request_digest,
                            "response_sha256": response_digest,
                            "message_count": len(kwargs.get("messages") or []),
                            "tool_names": _tool_names(kwargs.get("tools")),
                            "response_choice_count": len(choices) if isinstance(choices, list) else 0,
                            "response_tool_call_count": _response_tool_call_count(result),
                        },
                    }
                )
            else:
                sink.append(
                    {
                        "kind": component,
                        "elapsed_ms": elapsed,
                        "request_sha256": request_digest,
                        "response_sha256": response_digest,
                    }
                )

        _best_effort_observe(sink, f"{component}:success", observe_success)
        return result

    return wrapped


def _purge_server_modules() -> None:
    for name in tuple(sys.modules):
        if name == "server" or name.startswith("server."):
            del sys.modules[name]


def _configure_historical_state(data_dir: Path, model_id: str) -> None:
    config = importlib.import_module("server.config")
    settings = config.Settings(
        server_host="127.0.0.1",
        openrouter_api_key="live-lab-fake-key",
        lab_enabled=True,
        lab_composio_user_id="live-lab-local-user",
        interaction_agent_model=model_id,
        execution_agent_model=model_id,
        execution_agent_search_model=model_id,
        summarizer_model=model_id,
        email_classifier_model=model_id,
        conversation_summary_threshold=0,
    )
    config.get_settings = lambda: settings

    memory_module = importlib.import_module(
        "server.services.conversation.summarization.working_memory_log"
    )
    memory_module._working_memory_log = memory_module.WorkingMemoryLog(
        data_dir / "conversation" / "poke_working_memory.log"
    )
    conversation_module = importlib.import_module("server.services.conversation.log")
    conversation_module._conversation_log = conversation_module.ConversationLog(
        data_dir / "conversation" / "poke_conversation.log"
    )
    roster_module = importlib.import_module("server.services.execution.roster")
    roster_module._agent_roster = roster_module.AgentRoster(
        data_dir / "execution_agents" / "roster.json"
    )
    logs_module = importlib.import_module("server.services.execution.log_store")
    logs_module._execution_agent_logs = logs_module.ExecutionAgentLogStore(
        data_dir / "execution_agents"
    )
    timezone_module = importlib.import_module("server.services.timezone_store")
    timezone_module._timezone_store = timezone_module.TimezoneStore(data_dir / "timezone.txt")


def _transport(fake_base_url: str):
    client_module = importlib.import_module("server.openrouter_client.client")
    original = client_module.request_chat_completion

    async def request_chat_completion(**kwargs: Any) -> Any:
        kwargs["base_url"] = fake_base_url
        return await original(**kwargs)

    return request_chat_completion


def install_baseline_wrappers(
    sink: ObservationSink,
    *,
    fake_base_url: str,
    execution_timeout_seconds: float = 90.0,
) -> None:
    """Patch all imported call sites before importing ``server.app``."""

    interaction_agent = importlib.import_module("server.agents.interaction_agent.agent")
    interaction_runtime = importlib.import_module("server.agents.interaction_agent.runtime")
    interaction_tools = importlib.import_module("server.agents.interaction_agent.tools")
    execution_agent = importlib.import_module("server.agents.execution_agent.agent")
    execution_runtime = importlib.import_module("server.agents.execution_agent.runtime")
    email_search = importlib.import_module("server.agents.execution_agent.tasks.search_email.tool")
    summarizer = importlib.import_module("server.services.conversation.summarization.summarizer")
    classifier = importlib.import_module("server.services.gmail.importance_classifier")
    interaction_tools._EXECUTION_BATCH_MANAGER.timeout_seconds = execution_timeout_seconds

    active_wrapper = wrap_sync_call(
        "active_roster", interaction_agent._render_active_agents, sink
    )
    interaction_agent._render_active_agents = active_wrapper
    prompt_wrapper = wrap_sync_call(
        "interaction_prompt", interaction_agent.prepare_message_with_history, sink
    )
    interaction_agent.prepare_message_with_history = prompt_wrapper
    interaction_runtime.prepare_message_with_history = prompt_wrapper

    original_parse_tool_calls = interaction_runtime.InteractionAgentRuntime._parse_tool_calls

    @functools.wraps(original_parse_tool_calls)
    def parse_tool_calls_with_rejections(runtime: Any, raw_tool_calls: Any) -> Any:
        parsed = original_parse_tool_calls(runtime, raw_tool_calls)

        def observe_rejections() -> None:
            for tool_call in parsed:
                arguments = getattr(tool_call, "arguments", {})
                if isinstance(arguments, dict) and "__invalid_arguments__" in arguments:
                    sink.append(
                        {
                            "kind": "tool_call",
                            "tool_call": {
                                "name": str(getattr(tool_call, "name", "")),
                                "arguments": {},
                                "raw_arguments_sha256": sink.store_private(raw_tool_calls),
                                "malformed": True,
                            },
                        }
                    )

        _best_effort_observe(sink, "interaction_tool_parse:rejected", observe_rejections)
        return parsed

    interaction_runtime.InteractionAgentRuntime._parse_tool_calls = parse_tool_calls_with_rejections

    execution_prompt = wrap_sync_call(
        "execution_prompt", execution_agent.ExecutionAgent.build_system_prompt_with_history, sink
    )
    execution_agent.ExecutionAgent.build_system_prompt_with_history = execution_prompt
    tool_wrapper = wrap_sync_call("interaction_tool", interaction_tools.handle_tool_call, sink)
    interaction_tools.handle_tool_call = tool_wrapper
    interaction_runtime.handle_tool_call = tool_wrapper

    transport = _transport(fake_base_url)
    interaction_runtime.request_chat_completion = wrap_async_call(
        "interaction_model", transport, sink
    )
    execution_runtime.request_chat_completion = wrap_async_call(
        "execution_model", transport, sink
    )
    email_search.request_chat_completion = wrap_async_call("email_search_model", transport, sink)
    summarizer.request_chat_completion = wrap_async_call("summarizer_model", transport, sink)
    classifier.request_chat_completion = wrap_async_call("classifier_model", transport, sink)


def build_historical_app(
    *,
    worktree: Path,
    data_dir: Path,
    run_dir: Path,
    fake_base_url: str,
    model_id: str,
    process_nonce: str,
    execution_timeout_seconds: float = 90.0,
):
    """Import the baseline package only after process-local hooks are ready."""

    _purge_server_modules()
    root = str(worktree.resolve(strict=True))
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parents[2]]
    sys.path.insert(0, root)
    _configure_historical_state(data_dir.resolve(strict=True), model_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    context_path = run_dir / "process_context.json"
    previous_nonce: str | None = None
    try:
        previous_context = json.loads(context_path.read_text(encoding="utf-8"))
        if isinstance(previous_context, dict) and isinstance(
            previous_context.get("process_nonce"), str
        ):
            previous_nonce = previous_context["process_nonce"]
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        pass
    if previous_nonce != process_nonce:
        (run_dir / "active_turn.json").unlink(missing_ok=True)
        (run_dir / "tainted.json").unlink(missing_ok=True)
    context_path.write_text(
        json.dumps({"process_nonce": process_nonce}, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    sink = ObservationSink(
        run_dir / "events.jsonl",
        run_dir / "private",
        owner_path=run_dir / "active_turn.json",
        process_nonce=process_nonce,
    )
    install_baseline_wrappers(
        sink,
        fake_base_url=fake_base_url,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    app_module = importlib.import_module("server.app")
    app_module.app.router.on_startup.clear()
    app_module.app.router.on_shutdown.clear()

    async def live_lab_ready() -> dict[str, str]:
        return {"nonce": process_nonce}

    app_module.app.add_api_route(
        f"/__live_lab_ready__/{process_nonce}",
        live_lab_ready,
        methods=["GET"],
        include_in_schema=False,
    )
    return app_module.app


def _file_hashes(data_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not data_dir.exists():
        return hashes
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            hashes[path.relative_to(data_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _preflight(args: argparse.Namespace) -> int:
    worktree = Path(args.worktree)
    revision = verify_baseline_revision(
        worktree,
        expected_base=HISTORICAL_BASE_SHA,
        allowed_paths=APPROVED_OVERLAY_PATHS,
    )
    data_dir = Path(args.data_dir) if args.data_dir else worktree / "server" / "data"
    roster_path = data_dir / "execution_agents" / "roster.json"
    roster_count = 0
    if roster_path.is_file():
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        roster_count = len(roster) if isinstance(roster, list) else 0
    model_id = args.model_id
    print(
        json.dumps(
            {
                "base_sha": revision.base_sha,
                "fixture_hashes": _file_hashes(data_dir),
                "model_ids": {
                    "classifier": model_id,
                    "email_search": model_id,
                    "execution": model_id,
                    "interaction": model_id,
                    "summarizer": model_id,
                },
                "overlay_diff_sha256": revision.overlay_diff_sha256,
                "overlay_head_sha": revision.overlay_head_sha,
                "port": args.port,
                "roster_count": roster_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--data-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--fake-base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--model-id", default="live-lab/fake-chat-completions")
    parser.add_argument("--execution-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--readiness-nonce")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.host != "127.0.0.1" or args.port != 8001:
        raise ValueError("historical live-lab server is pinned to 127.0.0.1:8001")
    if args.preflight_only:
        return _preflight(args)
    if not args.data_dir or not args.run_dir:
        raise ValueError("--data-dir and --run-dir are required unless --preflight-only is used")
    if not args.readiness_nonce:
        raise ValueError("--readiness-nonce is required for a measured historical launch")
    verify_baseline_revision(
        Path(args.worktree),
        expected_base=HISTORICAL_BASE_SHA,
        allowed_paths=APPROVED_OVERLAY_PATHS,
    )
    app = build_historical_app(
        worktree=Path(args.worktree),
        data_dir=Path(args.data_dir),
        run_dir=Path(args.run_dir),
        fake_base_url=args.fake_base_url,
        model_id=args.model_id,
        process_nonce=args.readiness_nonce,
        execution_timeout_seconds=args.execution_timeout_seconds,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ObservationSink",
    "build_historical_app",
    "install_baseline_wrappers",
    "main",
    "wrap_async_call",
    "wrap_sync_call",
]
