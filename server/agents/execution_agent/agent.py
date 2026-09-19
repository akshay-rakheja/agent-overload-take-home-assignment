"""Execution Agent implementation."""

from pathlib import Path
from itertools import chain
from typing import List, Optional, Dict, Any

from ...config import get_settings
from ...services.execution import (
    AgentDirectory,
    ContextMetrics,
    ExecutionAgentLogStore,
    ExecutionContextPolicy,
    UnknownAgentError,
    get_agent_directory,
    get_execution_agent_logs,
)
from ...logging_config import logger


# Load system prompt template from file
_prompt_path = Path(__file__).parent / "system_prompt.md"
if _prompt_path.exists():
    SYSTEM_PROMPT_TEMPLATE = _prompt_path.read_text(encoding="utf-8").strip()
else:
    # Placeholder template - you'll replace this with actual instructions
    SYSTEM_PROMPT_TEMPLATE = """You are an execution agent responsible for completing specific tasks using available tools.

Agent Name: {agent_name}
Purpose: {agent_purpose}

Instructions:
[TO BE FILLED IN BY USER]

You have access to Gmail tools to help complete your tasks. When given instructions:
1. Analyze what needs to be done
2. Use the appropriate tools to complete the task
3. Provide clear status updates on your actions

Be thorough, accurate, and efficient in your execution."""


class ExecutionAgent:
    """Manages state and history for an execution agent."""

    # Initialize execution agent with name, conversation limits, and log store access
    def __init__(
        self,
        name: str,
        conversation_limit: Optional[int] = None,
        storage_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        legacy_storage_key: Optional[str] = None,
        log_store: Optional[ExecutionAgentLogStore] = None,
        directory: Optional[AgentDirectory] = None,
        context_policy: Optional[ExecutionContextPolicy] = None,
    ):
        """
        Initialize an execution agent.

        Args:
            name: Human-readable agent name (e.g., 'conversation with keith')
            conversation_limit: Optional limit on past conversations to include (None = all)
        """
        self.name = name
        self.storage_key = storage_key or name
        self.agent_id = agent_id
        self.legacy_storage_key = legacy_storage_key
        self.conversation_limit = conversation_limit
        self._log_store = log_store or get_execution_agent_logs()
        self._directory = directory or (get_agent_directory() if agent_id else None)
        settings = get_settings()
        recent_episode_limit = conversation_limit or settings.execution_context_max_recent_episodes
        self._context_policy = context_policy or ExecutionContextPolicy(
            max_recent_episodes=recent_episode_limit,
            max_characters=settings.execution_context_max_characters,
        )
        self.last_context_metrics = ContextMetrics(
            raw_entry_count=0,
            raw_history_characters=0,
            raw_history_bytes=0,
            rendered_characters=0,
            rendered_bytes=0,
            included_episode_count=0,
            omitted_entry_count=0,
            truncated_entry_count=0,
            summary_used=False,
        )

    # Generate system prompt template with agent name and purpose derived from name
    def build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        agent_purpose = f"Handle tasks related to: {self.name}"

        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.name,
            agent_purpose=agent_purpose
        )

    # Combine base system prompt with conversation history, applying conversation limits
    def build_system_prompt_with_history(self) -> str:
        """
        Build system prompt including agent history.

        Returns:
            System prompt with embedded history transcript
        """
        base_prompt = self.build_system_prompt()

        memory_summary = ""
        if self.agent_id and self._directory is not None:
            try:
                memory_summary = self._directory.require(self.agent_id).memory_summary
            except UnknownAgentError:
                memory_summary = ""

        entry_sources = []
        if self.legacy_storage_key and self.legacy_storage_key != self.storage_key:
            entry_sources.append(self._log_store.iter_entries(self.legacy_storage_key))
        entry_sources.append(self._log_store.iter_entries(self.storage_key))
        context = self._context_policy.render(
            chain.from_iterable(entry_sources),
            memory_summary=memory_summary,
        )
        self.last_context_metrics = context.metrics

        if context.text:
            return f"{base_prompt}\n\n# Execution History\n\n{context.text}"

        return base_prompt

    # Format current instruction as user message for LLM consumption
    def build_messages_for_llm(self, current_instruction: str) -> List[Dict[str, str]]:
        """
        Build message array for LLM call.

        Args:
            current_instruction: Current instruction from interaction agent

        Returns:
            List of messages in OpenRouter format
        """
        return [
            {"role": "user", "content": current_instruction}
        ]

    # Log the agent's final response to the execution log store
    def record_response(self, response: str) -> None:
        """Record agent's response to the log."""
        self._log_store.record_agent_response(self.storage_key, response)

    # Log tool invocation and results with truncated content for readability
    def record_tool_execution(self, tool_name: str, arguments: str, result: str) -> None:
        """Record tool execution details."""
        self._log_store.record_action(self.storage_key, f"Calling {tool_name} with: {arguments[:200]}")
        # Record the tool response
        self._log_store.record_tool_response(self.storage_key, tool_name, result[:500])
