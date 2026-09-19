"""Interaction agent module."""

from .agent import (
    CandidateContext,
    build_candidate_context,
    build_system_prompt,
    prepare_message_with_history,
)
from .runtime import InteractionAgentRuntime, InteractionResult
from .tools import DispatchContext, ToolResult, get_tool_schemas, handle_tool_call

__all__ = [
    "InteractionAgentRuntime",
    "InteractionResult",
    "CandidateContext",
    "build_candidate_context",
    "build_system_prompt",
    "prepare_message_with_history",
    "DispatchContext",
    "ToolResult",
    "get_tool_schemas",
    "handle_tool_call",
]
