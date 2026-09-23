from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models import ChatHistoryClearResponse, ChatHistoryResponse, ChatRequest
from ..services import get_conversation_log, get_trigger_service, handle_chat_request

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/send", response_class=JSONResponse, summary="Submit a chat message and receive a completion")
# Handle incoming chat messages and route them to the interaction agent
async def chat_send(
    payload: ChatRequest,
    system: str | None = None,
) -> JSONResponse:
    resolved_system = system or payload.system
    return await handle_chat_request(payload, system=resolved_system)


@router.get("/history", response_model=ChatHistoryResponse)
# Retrieve the conversation history from the log
def chat_history(system: str | None = None) -> ChatHistoryResponse:
    log = get_conversation_log(system)
    return ChatHistoryResponse(messages=log.to_chat_messages())


@router.delete("/history", response_model=ChatHistoryClearResponse)
def clear_history(system: str | None = None) -> ChatHistoryClearResponse:
    from ..services import get_execution_agent_logs, get_agent_roster
    from ..services.execution.inspector import clear_inspector_state

    if system in ("enhanced_jev", "jev"):
        log = get_conversation_log("enhanced_jev")
        log.clear()
        get_agent_roster("enhanced_jev").clear()
        clear_inspector_state("enhanced_jev")
    elif system in ("enhanced_deterministic", "enhanced"):
        log = get_conversation_log("enhanced_deterministic")
        log.clear()
        get_agent_roster("enhanced_deterministic").clear()
        clear_inspector_state("enhanced_deterministic")
    else:
        # Clear all
        get_conversation_log().clear()
        get_conversation_log("enhanced_jev").clear()

        # Clear execution agent logs
        execution_logs = get_execution_agent_logs()
        execution_logs.clear_all()

        # Clear agent rosters
        get_agent_roster("enhanced_deterministic").clear()
        get_agent_roster("enhanced_jev").clear()

        # Clear stored triggers
        trigger_service = get_trigger_service()
        trigger_service.clear_all()

        # Clear inspector state
        clear_inspector_state()

    return ChatHistoryClearResponse()


__all__ = ["router"]
