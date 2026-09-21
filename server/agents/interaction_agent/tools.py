"""Tool definitions for interaction agent."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from ...config import get_settings
from ...logging_config import logger
from ...services.conversation import get_conversation_log
from ...services.evaluation_lab import LabToolPolicy, lab_tool_rejection
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
                "oneOf": [
                    {"required": ["agent_id"]},
                    {"required": ["agent_name", "agent_purpose"]}
                ],
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

    resolved_directory = directory or get_agent_directory()
    resolved_logs = log_store or get_execution_agent_logs()
    resolved_batch_manager = batch_manager or _EXECUTION_BATCH_MANAGER
    context = dispatch_context or DispatchContext()

    def routing_rejection(message: str) -> ToolResult:
        return ToolResult(
            success=False,
            payload={"code": "routing_not_authorized", "message": message},
        )

    if agent_id and (agent_name or agent_purpose or creation_intent_id):
        return ToolResult(
            success=False,
            payload={
                "code": "invalid_agent_reference",
                "message": "Provide agent_id for reuse or agent_name and agent_purpose for creation, not both.",
            },
        )

    is_new = False
    pending_creation = False
    if agent_id:
        if context.routing_action is not None and context.routing_action is not RoutingAction.REUSE:
            return routing_rejection("The routing policy did not authorize agent reuse for this turn.")
        try:
            parsed_agent_id = UUID(agent_id)
        except (TypeError, ValueError, AttributeError):
            parsed_agent_id = None
        if (
            context.routing_action is RoutingAction.REUSE
            and parsed_agent_id not in context.allowed_agent_ids
        ):
            return routing_rejection("Only the recommended execution agent may be reused for this turn.")
        try:
            record = resolved_directory.require(agent_id)
        except UnknownAgentError:
            return ToolResult(
                success=False,
                payload={
                    "code": "unknown_agent_id",
                    "message": "The selected execution agent no longer exists. Refresh candidates.",
                },
            )
    else:
        if context.routing_action is not None and context.routing_action is not RoutingAction.CREATE_NEW:
            return routing_rejection("The routing policy did not authorize agent creation for this turn.")
        if not agent_name or not agent_purpose:
            return ToolResult(
                success=False,
                payload={
                    "code": "missing_new_agent_metadata",
                    "message": "Creating an execution agent requires both agent_name and agent_purpose.",
                },
            )
        normalized_intent = normalize_agent_text(creation_intent_id or "")
        creation_key = (
            "intent" if normalized_intent else "name",
            normalized_intent or normalize_agent_text(agent_name),
        )
        existing_id = context.created_agent_ids.get(creation_key)
        if existing_id is not None:
            record = resolved_directory.require(existing_id)
        else:
            pending_creation = True

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    if pending_creation:
        record = resolved_directory.create(
            name=agent_name,
            purpose=agent_purpose,
            aliases=(agent_name,),
        )
        context.created_agent_ids[creation_key] = record.agent_id
        is_new = True

    record = resolved_directory.mark_used(record.agent_id)
    stable_id = str(record.agent_id)
    resolved_logs.record_request(stable_id, instructions)

    action = "Created" if is_new else "Reused"
    logger.info(f"{action} agent: {record.name} ({stable_id})")

    async def _execute_async() -> None:
        try:
            execution_kwargs: dict[str, Any] = {"agent_id": stable_id}
            if record.legacy_storage_key:
                execution_kwargs["legacy_storage_key"] = record.legacy_storage_key
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

    return ToolResult(
        success=True,
        payload={
            "status": "submitted",
            "agent_id": stable_id,
            "agent_name": record.name,
            "new_agent_created": is_new,
        },
    )


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    log = get_conversation_log()
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
) -> ToolResult:
    """Record a draft update in the conversation log for the interaction agent."""
    if get_settings().lab_enabled:
        decision = LabToolPolicy().decide_model_tool("send_draft")
        return ToolResult(
            success=False,
            payload=lab_tool_rejection("send_draft", decision),
        )

    log = get_conversation_log()

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
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
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
            return send_message_to_agent(**args, dispatch_context=dispatch_context)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
