import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Any, Dict, List, Literal, Optional, Set

from .agent import CandidateContext, build_candidate_context, build_system_prompt, prepare_message_with_history
from .tools import DispatchContext, ToolResult, get_tool_schemas, handle_tool_call
from ...config import ModelRole, get_settings
from ...services.conversation import get_conversation_log, get_working_memory_log
from ...services.evaluation_lab.models import TraceEventKind
from ...services.evaluation_lab.trace import emit_trace, trace_timing
from ...services.evaluation_lab.usage import PhaseName, monotonic_phase
from ...services.execution import (
    ActivityCard,
    ActivityCardGenerator,
    AgentCandidate,
    AgentDirectory,
    AgentStatus,
    FakeJevClient,
    JevRouter,
    RoutingAction,
    RoutingDecision,
    TypeSafeJevClient,
    get_agent_directory,
)
from ...services.execution.inspector import (
    finalize_turn_inspection,
    record_agent_callback_response,
    record_dispatch_result,
    record_tool_dispatch,
    record_turn_candidate_context,
)
from ...openrouter_client import request_chat_completion
from ...logging_config import logger


@dataclass
class InteractionResult:
    """Result from the interaction agent."""

    success: bool
    response: str
    error: Optional[str] = None
    execution_agents_used: int = 0


@dataclass
class _ToolCall:
    """Parsed tool invocation from an LLM response."""

    identifier: Optional[str]
    name: str
    arguments: Dict[str, Any]


@dataclass
class _LoopSummary:
    """Aggregate information produced by the interaction loop."""

    last_assistant_text: str = ""
    user_messages: List[str] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    execution_agents: Set[str] = field(default_factory=set)


class InteractionAgentRuntime:
    """Manages the interaction agent's request processing."""

    MAX_TOOL_ITERATIONS = 8
    routing_mode: Literal["deterministic", "jev"] = "deterministic"
    system_name: str = "enhanced_deterministic"
    conversation_log: Any = None

    # Initialize interaction agent runtime with settings and service dependencies
    def __init__(
        self,
        *,
        directory: AgentDirectory | None = None,
        routing_mode: Literal["deterministic", "jev"] = "deterministic",
        system_name: str = "enhanced_deterministic",
    ) -> None:
        settings = get_settings()
        self.api_key = settings.openrouter_api_key
        self.model_config = settings.model_call_config(ModelRole.INTERACTION)
        self.model = self.model_config.model_id
        self.settings = settings
        self.routing_mode = routing_mode
        self.system_name = system_name
        self.conversation_log = get_conversation_log(system_name)
        self.working_memory_log = get_working_memory_log()
        self.tool_schemas = get_tool_schemas()
        self.agent_directory = directory or get_agent_directory(system_name)
        self.dispatch_context = DispatchContext(system_name=system_name)

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
            )

    # Main entry point for processing user messages through the LLM interaction loop
    async def execute(self, user_message: str) -> InteractionResult:
        """Handle a user-authored message."""

        try:
            transcript_before = self._load_conversation_transcript()
            self.conversation_log.record_user_message(user_message)

            system_prompt = build_system_prompt()
            routing_mode = getattr(self, "routing_mode", "deterministic")
            system_name = getattr(self, "system_name", "enhanced_deterministic")
            if routing_mode == "jev":
                messages = await self._prepare_turn_messages_jev(
                    user_message, transcript_before, message_type="user"
                )
            else:
                messages = self._prepare_turn_messages(
                    user_message, transcript_before, message_type="user"
                )

            logger.info("Processing user message through interaction agent")
            summary = await self._run_interaction_loop(system_prompt, messages)

            final_response = self._finalize_response(summary)

            if final_response and not summary.user_messages:
                self.conversation_log.record_reply(final_response)

            finalize_turn_inspection(user_message, final_response or "", system=system_name)

            return InteractionResult(
                success=True,
                response=final_response,
                execution_agents_used=len(summary.execution_agents),
            )

        except Exception as exc:
            emit_trace(
                TraceEventKind.ERROR,
                {
                    "boundary": "interaction_runtime",
                    "stage": "execute",
                    "error_type": type(exc).__name__,
                },
            )
            logger.error("Interaction agent failed", extra={"error": str(exc)})
            error_message = f"⚠️ Error: {exc}"
            self.conversation_log.record_reply(error_message)
            return InteractionResult(
                success=False,
                response=error_message,
                error=str(exc),
            )

    # Handle incoming messages from execution agents and generate appropriate responses
    async def handle_agent_message(self, agent_message: str) -> InteractionResult:
        """Process a status update emitted by an execution agent."""

        try:
            transcript_before = self._load_conversation_transcript()
            self.conversation_log.record_agent_message(agent_message)

            system_prompt = build_system_prompt()
            routing_mode = getattr(self, "routing_mode", "deterministic")
            system_name = getattr(self, "system_name", "enhanced_deterministic")
            if routing_mode == "jev":
                messages = await self._prepare_turn_messages_jev(
                    agent_message, transcript_before, message_type="agent"
                )
            else:
                messages = self._prepare_turn_messages(
                    agent_message, transcript_before, message_type="agent"
                )

            logger.info("Processing execution agent results")
            summary = await self._run_interaction_loop(system_prompt, messages)

            final_response = self._finalize_response(summary)

            if final_response and not summary.user_messages:
                self.conversation_log.record_reply(final_response)

            record_agent_callback_response(agent_message, final_response or "", system=system_name)

            return InteractionResult(
                success=True,
                response=final_response,
                execution_agents_used=len(summary.execution_agents),
            )

        except Exception as exc:
            emit_trace(
                TraceEventKind.ERROR,
                {
                    "boundary": "interaction_runtime",
                    "stage": "handle_agent_message",
                    "error_type": type(exc).__name__,
                },
            )
            logger.error("Interaction agent (agent message) failed", extra={"error": str(exc)})
            error_message = f"⚠️ Error: {exc}"
            self.conversation_log.record_reply(error_message)
            return InteractionResult(
                success=False,
                response=error_message,
                error=str(exc),
            )

    def _prepare_turn_messages(
        self,
        latest_text: str,
        transcript: str,
        *,
        message_type: str,
    ) -> List[Dict[str, str]]:
        """Bind the deterministic routing decision to this turn's tool permissions."""
        routing_mode = getattr(self, "routing_mode", "deterministic")
        system_name = getattr(self, "system_name", "enhanced_deterministic")
        if routing_mode == "jev":
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(
                    self._prepare_turn_messages_jev(
                        latest_text, transcript, message_type=message_type
                    )
                )
            else:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        self._prepare_turn_messages_jev(
                            latest_text, transcript, message_type=message_type
                        ),
                    ).result()

        candidate_context = build_candidate_context(
            latest_text,
            transcript,
            directory=self.agent_directory,
        )
        if message_type == "user":
            record_turn_candidate_context(candidate_context, system=system_name)
        allowed_ids = (
            frozenset({candidate_context.decision.agent_id})
            if candidate_context.decision.action is RoutingAction.REUSE
            and candidate_context.decision.agent_id is not None
            else frozenset()
        )
        self.dispatch_context = DispatchContext(
            routing_action=candidate_context.decision.action,
            allowed_agent_ids=allowed_ids,
            system_name=system_name,
        )
        emit_trace(
            TraceEventKind.ROUTING_DECISION,
            {
                "action": candidate_context.decision.action.value,
                "agent_id": (
                    str(candidate_context.decision.agent_id)
                    if candidate_context.decision.agent_id is not None
                    else None
                ),
                "confidence": candidate_context.decision.confidence,
                "reasons": candidate_context.decision.reasons,
                "recommendation": (
                    str(candidate_context.decision.agent_id)
                    if candidate_context.decision.agent_id is not None
                    else candidate_context.decision.action.value
                ),
            },
        )
        emit_trace(
            TraceEventKind.AUTHORIZATION,
            {
                "routing_action": (
                    self.dispatch_context.routing_action.value
                    if self.dispatch_context.routing_action is not None
                    else None
                ),
                "authorized_ids": sorted(
                    str(agent_id) for agent_id in self.dispatch_context.allowed_agent_ids
                ),
            },
        )
        return prepare_message_with_history(
            latest_text,
            transcript,
            message_type=message_type,
            directory=self.agent_directory,
            candidate_context=candidate_context,
        )

    async def _prepare_turn_messages_jev(
        self,
        latest_text: str,
        transcript: str,
        *,
        message_type: str,
    ) -> List[Dict[str, str]]:
        """Bind the TypeSafe Jev Map/Reduce routing decision to this turn's tool permissions."""
        records = self.agent_directory.list_records()
        cards: list[ActivityCard] = []

        from ...services.execution import get_execution_agent_logs
        log_store = get_execution_agent_logs()

        for r in records:
            gen = ActivityCardGenerator(
                agent_id=r.agent_id,
                name=r.name,
                purpose=r.purpose,
                status=r.status.value if hasattr(r.status, "value") else str(r.status),
                created_at=r.created_at or datetime.now(timezone.utc),
                last_used_at=r.last_used_at or datetime.now(timezone.utc),
            )
            journal_entries = []
            try:
                raw_bytes = b""
                if hasattr(log_store, "read_raw_bytes"):
                    raw_bytes = log_store.read_raw_bytes(str(r.agent_id))
                    if not raw_bytes and getattr(r, "legacy_storage_key", None):
                        raw_bytes = log_store.read_raw_bytes(r.legacy_storage_key)
                    if not raw_bytes:
                        raw_bytes = log_store.read_raw_bytes(r.name)
                elif hasattr(log_store, "read_log"):
                    raw_log = log_store.read_log(r.name)
                    raw_bytes = raw_log.encode("utf-8", errors="replace") if isinstance(raw_log, str) else b""

                raw_log = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else ""
                if raw_log:
                    import re
                    req_matches = list(re.finditer(r"<agent_request[^>]*>(.*?)</agent_request>", raw_log, re.DOTALL))
                    for m in req_matches[-3:]:
                        txt = m.group(1).strip()
                        if txt:
                            journal_entries.append({
                                "agent_id": str(r.agent_id),
                                "episode_id": str(r.agent_id),
                                "occurred_at": (r.last_used_at or datetime.now(timezone.utc)).isoformat(),
                                "request_intent": txt[:200],
                                "disposition": "completed",
                            })
            except Exception:
                pass

            card = gen.generate_from_records(journal_entries, durable_summary=r.purpose)
            cards.append(card)

        # Instantiate Jev client (live TypeSafe client preferred if configured, fallback to Fake)
        try:
            client = TypeSafeJevClient()
        except Exception:
            client = FakeJevClient()

        router = JevRouter(
            client=client,
            max_concurrency=25,
            min_map_threshold=0.35,
            ambiguity_margin=0.05,
        )
        with monotonic_phase(PhaseName.RETRIEVAL_ROUTING):
            jev_ctx = await router.route(latest_text, cards)

        # Build AgentCandidate list from Jev scoring
        records_by_id = {r.agent_id: r for r in records}
        candidates: list[AgentCandidate] = []
        for s in (jev_ctx.shortlist or jev_ctx.map_scores):
            rec = records_by_id.get(s.agent_id) or (self.agent_directory.get(s.agent_id) if hasattr(self.agent_directory, "get") else None)
            cand_name = rec.name if rec else "agent"
            cand_purpose = rec.purpose if rec else ""
            cand_status = rec.status if rec else AgentStatus.HOT
            candidates.append(
                AgentCandidate(
                    agent_id=s.agent_id,
                    name=cand_name,
                    purpose=cand_purpose,
                    status=cand_status,
                    score=s.composite_score,
                    reasons=(
                        f"TypeSafe Jev score: {s.composite_score:.2f} (affinity: {s.affinity_score:.2f}, continuity: {s.continuity_score:.2f}, risk: {s.risk_score:.2f})",
                    ),
                    score_components={
                        "composite": s.composite_score,
                        "affinity": s.affinity_score,
                        "continuity": s.continuity_score,
                        "risk": s.risk_score,
                    },
                )
            )

        decision = RoutingDecision(
            action=jev_ctx.decision.action,
            agent_id=jev_ctx.decision.recommended_agent_id,
            confidence=jev_ctx.decision.confidence,
            reasons=(jev_ctx.decision.rationale,),
        )
        candidate_context = CandidateContext(candidates=tuple(candidates), decision=decision)

        if message_type == "user":
            record_turn_candidate_context(
                candidate_context,
                system=self.system_name,
                jev_context=jev_ctx,
            )

        allowed_ids = (
            frozenset({jev_ctx.decision.recommended_agent_id})
            if jev_ctx.decision.action is RoutingAction.REUSE
            and jev_ctx.decision.recommended_agent_id is not None
            else frozenset()
        )
        self.dispatch_context = DispatchContext(
            routing_action=jev_ctx.decision.action,
            allowed_agent_ids=allowed_ids,
            system_name=self.system_name,
        )

        emit_trace(
            TraceEventKind.ROUTING_DECISION,
            {
                "action": jev_ctx.decision.action.value,
                "agent_id": str(jev_ctx.decision.recommended_agent_id) if jev_ctx.decision.recommended_agent_id else None,
                "confidence": jev_ctx.decision.confidence,
                "reasons": [jev_ctx.decision.rationale],
                "recommendation": str(jev_ctx.decision.recommended_agent_id) if jev_ctx.decision.recommended_agent_id else jev_ctx.decision.action.value,
                "system": "enhanced_jev",
            },
        )
        emit_trace(
            TraceEventKind.AUTHORIZATION,
            {
                "routing_action": self.dispatch_context.routing_action.value if self.dispatch_context.routing_action else None,
                "authorized_ids": sorted(str(aid) for aid in self.dispatch_context.allowed_agent_ids),
                "system": "enhanced_jev",
            },
        )

        return prepare_message_with_history(
            latest_text,
            transcript,
            message_type=message_type,
            directory=self.agent_directory,
            candidate_context=candidate_context,
        )

    # Core interaction loop that handles LLM calls and tool executions until completion
    async def _run_interaction_loop(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> _LoopSummary:
        """Iteratively query the LLM until it issues a final response."""

        summary = _LoopSummary()

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            response = await self._make_llm_call(system_prompt, messages)
            assistant_message = self._extract_assistant_message(response)

            assistant_content = (assistant_message.get("content") or "").strip()
            if assistant_content:
                summary.last_assistant_text = assistant_content

            raw_tool_calls = assistant_message.get("tool_calls") or []
            parsed_tool_calls = self._parse_tool_calls(raw_tool_calls)

            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.get("content", "") or "",
            }
            if raw_tool_calls:
                assistant_entry["tool_calls"] = raw_tool_calls
            messages.append(assistant_entry)

            if not parsed_tool_calls:
                break

            for tool_call in parsed_tool_calls:
                summary.tool_names.append(tool_call.name)

                if tool_call.name == "send_message_to_agent":
                    agent_reference = (
                        tool_call.arguments.get("agent_id")
                        or tool_call.arguments.get("agent_name")
                    )
                    if isinstance(agent_reference, str) and agent_reference:
                        summary.execution_agents.add(agent_reference)

                system_name = getattr(self, "system_name", "enhanced_deterministic")
                record_tool_dispatch(tool_call.name, tool_call.arguments, system=system_name)
                result = self._execute_tool(tool_call)
                if tool_call.name == "send_message_to_agent":
                    record_dispatch_result(tool_call.arguments, result, system=system_name)

                if result.user_message:
                    summary.user_messages.append(result.user_message)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.identifier or tool_call.name,
                    "content": self._format_tool_result(tool_call, result),
                }
                messages.append(tool_message)
        else:
            if not summary.user_messages and not summary.last_assistant_text:
                raise RuntimeError("Reached tool iteration limit without final response")
            logger.warning("Reached tool iteration limit after delivering user messages")

        if not summary.user_messages and not summary.last_assistant_text:
            logger.warning("Interaction loop exited without assistant content")

        return summary

    # Load conversation history, preferring summarized version if available
    def _load_conversation_transcript(self) -> str:
        if self.settings.summarization_enabled:
            rendered = self.working_memory_log.render_transcript()
            if rendered.strip():
                return rendered
        return self.conversation_log.load_transcript()

    # Execute API call to OpenRouter with system prompt, messages, and tool schemas
    async def _make_llm_call(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make an LLM call via OpenRouter."""

        emit_trace(
            TraceEventKind.MODEL_CALL,
            {
                "runtime": "interaction",
                "stage": "started",
                "model": self.model,
                "tool_schema_count": len(self.tool_schemas),
            },
        )
        with monotonic_phase(
            PhaseName.INTERACTION_MODEL, clock=monotonic_ns
        ) as timing:
            logger.debug(
                "Interaction agent calling LLM",
                extra={"model": self.model, "tools": len(self.tool_schemas)},
            )
            try:
                model_policy = (
                    {"config": self.model_config}
                    if hasattr(self, "model_config")
                    else {"model": self.model}
                )
                response = await request_chat_completion(
                    role=ModelRole.INTERACTION,
                    messages=messages,
                    system=system_prompt,
                    api_key=self.api_key,
                    tools=self.tool_schemas,
                    **model_policy,
                )
            except Exception as exc:
                finished = timing.finish()
                emit_trace(
                    TraceEventKind.MODEL_CALL,
                    {
                        "runtime": "interaction",
                        "stage": "failed",
                        "model": self.model,
                        "started_monotonic_ns": timing.started_monotonic_ns,
                        "finished_monotonic_ns": finished,
                        "elapsed_ns": timing.elapsed_ns,
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            finished = timing.finish()
            emit_trace(
                TraceEventKind.MODEL_CALL,
                {
                    "runtime": "interaction",
                    "stage": "completed",
                    "model": self.model,
                    "started_monotonic_ns": timing.started_monotonic_ns,
                    "finished_monotonic_ns": finished,
                    "elapsed_ns": timing.elapsed_ns,
                },
            )
            return response

    # Extract the assistant's message from the OpenRouter API response structure
    def _extract_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Return the assistant message from the raw response payload."""

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM response did not include an assistant message")
        return message

    # Convert raw LLM tool calls into structured _ToolCall objects with validation
    def _parse_tool_calls(self, raw_tool_calls: List[Dict[str, Any]]) -> List[_ToolCall]:
        """Normalize tool call payloads from the LLM."""

        parsed: List[_ToolCall] = []
        for raw in raw_tool_calls:
            function_block = raw.get("function") or {}
            name = function_block.get("name")
            if not isinstance(name, str) or not name:
                logger.warning("Skipping tool call without name", extra={"tool": raw})
                continue

            arguments, error = self._parse_tool_arguments(function_block.get("arguments"))
            if error:
                logger.warning("Tool call arguments invalid", extra={"tool": name, "error": error})
                parsed.append(
                    _ToolCall(
                        identifier=raw.get("id"),
                        name=name,
                        arguments={"__invalid_arguments__": error},
                    )
                )
                continue

            parsed.append(
                _ToolCall(identifier=raw.get("id"), name=name, arguments=arguments)
            )

        return parsed

    # Parse and validate tool arguments from various formats (dict, JSON string, etc.)
    def _parse_tool_arguments(
        self, raw_arguments: Any
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Convert tool arguments into a dictionary, reporting errors."""

        if raw_arguments is None:
            return {}, None

        if isinstance(raw_arguments, dict):
            return raw_arguments, None

        if isinstance(raw_arguments, str):
            if not raw_arguments.strip():
                return {}, None
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                return {}, f"invalid json: {exc}"
            if isinstance(parsed, dict):
                return parsed, None
            return {}, "decoded arguments were not an object"

        return {}, f"unsupported argument type: {type(raw_arguments).__name__}"

    # Execute tool calls with error handling and logging, returning standardized results
    def _execute_tool(self, tool_call: _ToolCall) -> ToolResult:
        """Execute a tool call and convert low-level errors into structured results."""

        if "__invalid_arguments__" in tool_call.arguments:
            error = tool_call.arguments["__invalid_arguments__"]
            emit_trace(
                TraceEventKind.TOOL_CALL,
                {
                    "runtime": "interaction",
                    "tool_name": tool_call.name,
                    "stage": "rejected",
                    "success": False,
                    "reason": "invalid_arguments",
                },
            )
            self._log_tool_invocation(tool_call, stage="rejected", detail={"error": error})
            return ToolResult(success=False, payload={"error": error})

        emit_trace(
            TraceEventKind.TOOL_CALL,
            {
                "runtime": "interaction",
                "tool_name": tool_call.name,
                "stage": "started",
            },
        )
        with trace_timing(monotonic_ns) as timing:
            try:
                self._log_tool_invocation(tool_call, stage="start")
                call_kwargs = {"dispatch_context": getattr(self, "dispatch_context", None)}
                import inspect
                sig = inspect.signature(handle_tool_call)
                if "conversation_log" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    call_kwargs["conversation_log"] = getattr(self, "conversation_log", None)
                if "directory" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    call_kwargs["directory"] = getattr(self, "agent_directory", None)
                result = handle_tool_call(
                    tool_call.name,
                    tool_call.arguments,
                    **call_kwargs,
                )
            except Exception as exc:  # pragma: no cover - defensive
                finished = timing.finish()
                emit_trace(
                    TraceEventKind.TOOL_CALL,
                    {
                        "runtime": "interaction",
                        "tool_name": tool_call.name,
                        "stage": "failed",
                        "success": False,
                        "started_monotonic_ns": timing.started_monotonic_ns,
                        "finished_monotonic_ns": finished,
                        "elapsed_ns": timing.elapsed_ns,
                        "error_type": type(exc).__name__,
                    },
                )
                logger.error(
                    "Tool execution crashed",
                    extra={"tool": tool_call.name, "error": str(exc)},
                )
                self._log_tool_invocation(
                    tool_call,
                    stage="error",
                    detail={"error": str(exc)},
                )
                return ToolResult(success=False, payload={"error": str(exc)})

            if not isinstance(result, ToolResult):
                logger.warning(
                    "Tool did not return ToolResult; coercing",
                    extra={"tool": tool_call.name},
                )
                wrapped = ToolResult(success=True, payload=result)
                finished = timing.finish()
                emit_trace(
                    TraceEventKind.TOOL_CALL,
                    {
                        "runtime": "interaction",
                        "tool_name": tool_call.name,
                        "stage": "completed",
                        "success": wrapped.success,
                        "result": wrapped.payload,
                        "started_monotonic_ns": timing.started_monotonic_ns,
                        "finished_monotonic_ns": finished,
                        "elapsed_ns": timing.elapsed_ns,
                    },
                )
                self._log_tool_invocation(tool_call, stage="done", result=wrapped)
                return wrapped

            status = "success" if result.success else "error"
            logger.debug(
                "Tool executed",
                extra={
                    "tool": tool_call.name,
                    "status": status,
                },
            )
            self._log_tool_invocation(tool_call, stage="done", result=result)
            finished = timing.finish()
            trace_stage = "completed" if result.success else "rejected"
            emit_trace(
                TraceEventKind.TOOL_CALL,
                {
                    "runtime": "interaction",
                    "tool_name": tool_call.name,
                    "stage": trace_stage,
                    "success": result.success,
                    "result": result.payload,
                    "started_monotonic_ns": timing.started_monotonic_ns,
                    "finished_monotonic_ns": finished,
                    "elapsed_ns": timing.elapsed_ns,
                },
            )
            return result

    # Format tool execution results into JSON for LLM consumption
    def _format_tool_result(self, tool_call: _ToolCall, result: ToolResult) -> str:
        """Render a tool execution result back to the LLM."""

        payload: Dict[str, Any] = {
            "tool": tool_call.name,
            "status": "success" if result.success else "error",
            "arguments": {
                key: value
                for key, value in tool_call.arguments.items()
                if key != "__invalid_arguments__"
            },
        }

        if result.payload is not None:
            key = "result" if result.success else "error"
            payload[key] = result.payload

        return self._safe_json_dump(payload)

    # Safely serialize objects to JSON with fallback to string representation
    def _safe_json_dump(self, payload: Any) -> str:
        """Serialize payload to JSON, falling back to repr on failure."""

        try:
            return json.dumps(payload, default=str)
        except TypeError:
            return repr(payload)

    # Log tool execution stages (start, done, error) with structured metadata
    def _log_tool_invocation(
        self,
        tool_call: _ToolCall,
        *,
        stage: str,
        result: Optional[ToolResult] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit structured logs for tool lifecycle events."""

        cleaned_args = {
            key: value
            for key, value in tool_call.arguments.items()
            if key != "__invalid_arguments__"
        }

        log_payload: Dict[str, Any] = {
            "tool": tool_call.name,
            "stage": stage,
            "arguments": cleaned_args,
        }

        if result is not None:
            log_payload["success"] = result.success
            if result.payload is not None:
                log_payload["payload"] = result.payload

        if detail:
            log_payload.update(detail)

        if stage == "done":
            logger.info(f"Tool '{tool_call.name}' completed")
        elif stage in {"error", "rejected"}:
            logger.warning(f"Tool '{tool_call.name}' {stage}")
        else:
            logger.debug(f"Tool '{tool_call.name}' {stage}")

    # Determine final user-facing response from interaction loop summary
    def _finalize_response(self, summary: _LoopSummary) -> str:
        """Decide what text should be exposed to the user as the final reply."""

        if summary.user_messages:
            response = summary.user_messages[-1]
        else:
            response = summary.last_assistant_text

        emit_trace(TraceEventKind.FINAL_RESPONSE, {"response": response})
        return response
