"""Tool definitions for interaction agent."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from ...config import get_settings
from ...logging_config import logger
from ...services.conversation import get_conversation_log
from ...services.evaluation_lab import LabToolPolicy, lab_tool_rejection
from ...services.evaluation_lab.models import Availability, TraceEventKind
from ...services.evaluation_lab.trace import emit_trace, trace_active
from ...services.execution import (
    AgentDirectory,
    ExecutionAgentLogStore,
    UnknownAgentError,
    get_agent_directory,
    get_execution_agent_logs,
    normalize_agent_text,
    RoutingAction,
)
from ..execution_agent.batch_manager import ExecutionBatchManager


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False


@dataclass
class DispatchContext:
    """Per-interaction routing authorization and creation idempotency state."""

    routing_action: RoutingAction | None = None
    allowed_agent_ids: frozenset[UUID] = frozenset()
    created_agent_ids: dict[tuple[str, str], UUID] = field(default_factory=dict)
    system_name: str = "enhanced_deterministic"

# Tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message_to_agent",
            "description": "Reuse a listed execution agent by stable ID, or create a new one with a name and purpose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Stable ID from agent_candidates. Required when reusing an existing agent."
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "Human-readable name for a new agent. Do not send this when reusing by ID."
                    },
                    "agent_purpose": {
                        "type": "string",
                        "description": "Concise routing purpose for a new agent. Required with agent_name."
                    },
                    "creation_intent_id": {
                        "type": "string",
                        "description": "Optional stable token for one create-new decision. Reuse it when retrying the same decision; use a different token only to intentionally create another identity in this turn."
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                },
                "required": ["instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Record an email draft so the user can review the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email for the draft.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject for the draft.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text).",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

_EXECUTION_BATCH_MANAGER = ExecutionBatchManager()


@dataclass(frozen=True)
class _DirectoryCountObservation:
    value: int | None
    availability: Availability
    reason: str | None = None


def _directory_count(directory: AgentDirectory) -> _DirectoryCountObservation:
    path = getattr(directory, "_path", None)
    if path is None or not hasattr(path, "read_text"):
        return _DirectoryCountObservation(
            value=None,
            availability=Availability.UNAVAILABLE,
            reason="directory snapshot unavailable",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _DirectoryCountObservation(
            value=None,
            availability=Availability.UNAVAILABLE,
            reason="directory snapshot missing",
        )
    except Exception:
        return _DirectoryCountObservation(
            value=None,
            availability=Availability.UNAVAILABLE,
            reason="directory snapshot unreadable",
        )

    if isinstance(payload, list):
        return _DirectoryCountObservation(
            value=len(payload), availability=Availability.AVAILABLE
        )
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and isinstance(payload.get("agents"), list)
    ):
        return _DirectoryCountObservation(
            value=len(payload["agents"]), availability=Availability.AVAILABLE
        )
    return _DirectoryCountObservation(
        value=None,
        availability=Availability.UNAVAILABLE,
        reason="directory snapshot shape unavailable",
    )


def _journal_sha256(log_store: ExecutionAgentLogStore, storage_key: str) -> str | None:
    try:
        return hashlib.sha256(log_store.read_raw_bytes(storage_key)).hexdigest()
    except Exception:
        return None


def _tool_result_code(result: ToolResult) -> str | None:
    if not isinstance(result.payload, dict):
        return None
    direct = result.payload.get("code")
    if isinstance(direct, str):
        return direct
    error = result.payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    return None


# Create or reuse execution agent and dispatch instructions asynchronously
def send_message_to_agent(
    instructions: str,
    agent_id: str | None = None,
    agent_name: str | None = None,
    agent_purpose: str | None = None,
    creation_intent_id: str | None = None,
    *,
    dispatch_context: DispatchContext | None = None,
    directory: AgentDirectory | None = None,
    log_store: ExecutionAgentLogStore | None = None,
    batch_manager: ExecutionBatchManager | None = None,
) -> ToolResult:
    """Dispatch by stable ID, or idempotently create a new identity for this turn."""

    context = dispatch_context or DispatchContext()
    resolved_directory = directory or (get_agent_directory(context.system_name) if context else get_agent_directory())
    resolved_logs = log_store or get_execution_agent_logs()
    resolved_batch_manager = batch_manager or _EXECUTION_BATCH_MANAGER
    tracing = trace_active()
    directory_count_before = (
        _directory_count(resolved_directory)
        if tracing
        else _DirectoryCountObservation(
            value=None,
            availability=Availability.NOT_APPLICABLE,
            reason="trace inactive",
        )
    )
    idempotent_creation = False

    emit_trace(
        TraceEventKind.DISPATCH_ATTEMPT,
        {
            "reference_type": "reuse" if agent_id else "create",
            "requested_agent_id": agent_id,
            "routing_action": (
                context.routing_action.value
                if context.routing_action is not None
                else None
            ),
            "authorized_ids": sorted(
                str(allowed_id) for allowed_id in context.allowed_agent_ids
            ),
            "creation_intent_supplied": bool(creation_intent_id),
            "directory_count_before": directory_count_before.value,
            "directory_count_before_availability": directory_count_before.availability.value,
            "directory_count_before_reason": directory_count_before.reason,
        },
    )

    def finish(
        result: ToolResult,
        *,
        record: Any = None,
        created: bool = False,
        journal_sha256_before: str | None = None,
        journal_sha256_after: str | None = None,
    ) -> ToolResult:
        directory_count_after = (
            _directory_count(resolved_directory)
            if tracing
            else _DirectoryCountObservation(
                value=None,
                availability=Availability.NOT_APPLICABLE,
                reason="trace inactive",
            )
        )
        payload: dict[str, Any] = {
            "status": "accepted" if result.success else "rejected",
            "success": result.success,
            "code": _tool_result_code(result),
            "directory_count_before": directory_count_before.value,
            "directory_count_before_availability": directory_count_before.availability.value,
            "directory_count_before_reason": directory_count_before.reason,
            "directory_count_after": directory_count_after.value,
            "directory_count_after_availability": directory_count_after.availability.value,
            "directory_count_after_reason": directory_count_after.reason,
        }
        if record is not None:
            stable_id = str(record.agent_id)
            payload.update(
                {
                    "selected_agent_id": stable_id,
                    "new_agent_created": created,
                    "idempotent_creation": idempotent_creation,
                    "journal_sha256_before": journal_sha256_before,
                    "journal_sha256_after": journal_sha256_after,
                }
            )
        emit_trace(TraceEventKind.DISPATCH_RESULT, payload)

        if result.success and record is not None:
            selected = {
                "agent_id": str(record.agent_id),
                "name": record.name,
                "status": record.status.value,
            }
            identity: dict[str, Any] = {
                "selected": selected,
                "delta": {
                    "directory_count_before": directory_count_before.value,
                    "directory_count_before_availability": directory_count_before.availability.value,
                    "directory_count_before_reason": directory_count_before.reason,
                    "directory_count_after": directory_count_after.value,
                    "directory_count_after_availability": directory_count_after.availability.value,
                    "directory_count_after_reason": directory_count_after.reason,
                    "journal_sha256_before": journal_sha256_before,
                    "journal_sha256_after": journal_sha256_after,
                },
                "idempotent_creation": idempotent_creation,
            }
            if created:
                identity["created"] = selected
            emit_trace(TraceEventKind.IDENTITY, identity)
        return result

    def routing_rejection(message: str) -> ToolResult:
        return ToolResult(
            success=False,
            payload={"code": "routing_not_authorized", "message": message},
        )

    if agent_id and (agent_name or agent_purpose or creation_intent_id):
        return finish(ToolResult(
            success=False,
            payload={
                "code": "invalid_agent_reference",
                "message": "Provide agent_id for reuse or agent_name and agent_purpose for creation, not both.",
            },
        ))

    is_new = False
    pending_creation = False
    if agent_id:
        if context.routing_action is not None and context.routing_action is not RoutingAction.REUSE:
            return finish(routing_rejection("The routing policy did not authorize agent reuse for this turn."))
        try:
            parsed_agent_id = UUID(agent_id)
        except (TypeError, ValueError, AttributeError):
            parsed_agent_id = None
        if (
            context.routing_action is RoutingAction.REUSE
            and parsed_agent_id not in context.allowed_agent_ids
        ):
            return finish(routing_rejection("Only the recommended execution agent may be reused for this turn."))
        try:
            record = resolved_directory.require(agent_id)
        except UnknownAgentError:
            return finish(ToolResult(
                success=False,
                payload={
                    "code": "unknown_agent_id",
                    "message": "The selected execution agent no longer exists. Refresh candidates.",
                },
            ))
    else:
        if context.routing_action is not None and context.routing_action is not RoutingAction.CREATE_NEW:
            return finish(routing_rejection("The routing policy did not authorize agent creation for this turn."))
        if not agent_name or not agent_purpose:
            return finish(ToolResult(
                success=False,
                payload={
                    "code": "missing_new_agent_metadata",
                    "message": "Creating an execution agent requires both agent_name and agent_purpose.",
                },
            ))
        normalized_intent = normalize_agent_text(creation_intent_id or "")
        creation_key = (
            "intent" if normalized_intent else "name",
            normalized_intent or normalize_agent_text(agent_name),
        )
        existing_id = context.created_agent_ids.get(creation_key)
        if existing_id is not None:
            idempotent_creation = True
            record = resolved_directory.require(existing_id)
        else:
            pending_creation = True

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return finish(ToolResult(success=False, payload={"error": "No event loop available"}))

    if pending_creation:
        record = resolved_directory.create(
            name=agent_name,
            purpose=agent_purpose,
            aliases=(agent_name,),
        )
        context.created_agent_ids[creation_key] = record.agent_id
        is_new = True

    journal_before = (
        _journal_sha256(resolved_logs, str(record.agent_id)) if tracing else None
    )
    record = resolved_directory.mark_used(record.agent_id)
    stable_id = str(record.agent_id)
    resolved_logs.record_request(stable_id, instructions)
    journal_after = _journal_sha256(resolved_logs, stable_id) if tracing else None

    action = "Created" if is_new else "Reused"
    logger.info(f"{action} agent: {record.name} ({stable_id})")

    async def _execute_async() -> None:
        try:
            execution_kwargs: dict[str, Any] = {"agent_id": stable_id}
            if record.legacy_storage_key:
                execution_kwargs["legacy_storage_key"] = record.legacy_storage_key
            import inspect
            if hasattr(resolved_batch_manager, "execute_agent"):
                sig = inspect.signature(resolved_batch_manager.execute_agent)
                if "system_name" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    execution_kwargs["system_name"] = getattr(context, "system_name", "enhanced_deterministic")
            result = await resolved_batch_manager.execute_agent(
                record.name,
                instructions,
                **execution_kwargs,
            )
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"Agent '{record.name}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{record.name}' failed: {str(exc)}")

    loop.create_task(_execute_async())

    return finish(ToolResult(
        success=True,
        payload={
            "status": "submitted",
            "agent_id": stable_id,
            "agent_name": record.name,
            "new_agent_created": is_new,
        },
    ),
        record=record,
        created=is_new,
        journal_sha256_before=journal_before,
        journal_sha256_after=journal_after,
    )


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str, *, conversation_log: Any = None) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    log = conversation_log or get_conversation_log()
    log.record_reply(message)

    return ToolResult(
        success=True,
        payload={"status": "delivered"},
        user_message=message,
        recorded_reply=True,
    )


# Format and record email draft for user review
def send_draft(
    to: str,
    subject: str,
    body: str,
    *,
    conversation_log: Any = None,
) -> ToolResult:
    """Record a draft update in the conversation log for the interaction agent."""
    if get_settings().lab_enabled:
        decision = LabToolPolicy().decide_model_tool("send_draft")
        emit_trace(
            TraceEventKind.GMAIL_EVIDENCE,
            {
                "boundary": "interaction_send_draft",
                "operation_name": "send_draft",
                "stage": "rejected",
                "allowed": decision.allowed,
                "policy_code": decision.code,
                "callable_executed": False,
            },
        )
        return ToolResult(
            success=False,
            payload=lab_tool_rejection("send_draft", decision),
        )

    log = conversation_log or get_conversation_log()

    message = f"To: {to}\nSubject: {subject}\n\n{body}"

    log.record_reply(message)
    logger.info(f"Draft recorded for: {to}")

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "to": to,
            "subject": subject,
        },
        recorded_reply=True,
    )


# Record silent wait state to avoid duplicate responses
def wait(reason: str, *, conversation_log: Any = None) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = conversation_log or get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return predefined tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
def handle_tool_call(
    name: str,
    arguments: Any,
    *,
    dispatch_context: DispatchContext | None = None,
    conversation_log: Any = None,
    directory: AgentDirectory | None = None,
) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        if name == "send_message_to_agent":
            return send_message_to_agent(**args, dispatch_context=dispatch_context, directory=directory)
        if name == "send_message_to_user":
            return send_message_to_user(**args, conversation_log=conversation_log)
        if name == "send_draft":
            return send_draft(**args, conversation_log=conversation_log)
        if name == "wait":
            return wait(**args, conversation_log=conversation_log)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
