import asyncio
from typing import Optional, Union

from fastapi import status
from fastapi.responses import JSONResponse, PlainTextResponse

from ...agents.interaction_agent.runtime import InteractionAgentRuntime
from ...logging_config import logger
from ...models import ChatMessage, ChatRequest
from ...utils import error_response
from ..evaluation_lab.usage import PhaseName, monotonic_phase


# Extract the most recent user message from the chat request payload
def _extract_latest_user_message(payload: ChatRequest) -> Optional[ChatMessage]:
    for message in reversed(payload.messages):
        if message.role.lower().strip() == "user" and message.content.strip():
            return message
    return None


async def handle_chat_request(
    payload: ChatRequest,
    system: Optional[str] = None,
) -> Union[PlainTextResponse, JSONResponse]:
    """Handle a chat request using the InteractionAgentRuntime."""

    # Extract user message
    user_message = _extract_latest_user_message(payload)
    if user_message is None:
        return error_response("Missing user message", status_code=status.HTTP_400_BAD_REQUEST)

    user_content = user_message.content.strip()  # Already checked in _extract_latest_user_message

    resolved_system = system or payload.system or "enhanced_deterministic"
    routing_mode = "jev" if resolved_system in ("enhanced_jev", "jev") else "deterministic"
    system_name = "enhanced_jev" if routing_mode == "jev" else "enhanced_deterministic"

    logger.info(
        "chat request",
        extra={
            "message_length": len(user_content),
            "routing_mode": routing_mode,
            "system_name": system_name,
        },
    )

    try:
        runtime = InteractionAgentRuntime(
            routing_mode=routing_mode,
            system_name=system_name,
        )
    except ValueError as ve:
        # Missing API key error
        logger.error("configuration error", extra={"error": str(ve)})
        return error_response(str(ve), status_code=status.HTTP_400_BAD_REQUEST)

    async def _run_interaction() -> None:
        try:
            with monotonic_phase(PhaseName.TOTAL_RUN):
                await runtime.execute(user_message=user_content)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("chat task failed", extra={"error": str(exc)})

    asyncio.create_task(_run_interaction())

    return PlainTextResponse("", status_code=status.HTTP_202_ACCEPTED)
