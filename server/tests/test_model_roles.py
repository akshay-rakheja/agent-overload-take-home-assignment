from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from server.agents.execution_agent import runtime as execution_runtime
from server.agents.execution_agent.runtime import ExecutionAgentRuntime
from server.agents.execution_agent.tasks.search_email import tool as email_search
from server.agents.interaction_agent import runtime as interaction_runtime
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.config import ModelCallConfig, ModelRole
from server.services.conversation.summarization import summarizer
from server.services.conversation.summarization.prompt_builder import SummaryPrompt
from server.services.gmail import importance_classifier
from server.services.gmail.processing import ProcessedEmail


def _config(role: ModelRole) -> ModelCallConfig:
    return ModelCallConfig(model_id=f"fixture/{role.value}", temperature=0)


def test_interaction_and_execution_calls_bind_exact_roles(monkeypatch) -> None:
    captured = []

    async def fake_request(**kwargs):
        captured.append(kwargs)
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(interaction_runtime, "request_chat_completion", fake_request)
    monkeypatch.setattr(execution_runtime, "request_chat_completion", fake_request)

    interaction = InteractionAgentRuntime.__new__(InteractionAgentRuntime)
    interaction.model_config = _config(ModelRole.INTERACTION)
    interaction.model = interaction.model_config.model_id
    interaction.api_key = "fixture-key"
    interaction.tool_schemas = []
    execution = ExecutionAgentRuntime.__new__(ExecutionAgentRuntime)
    execution.model_config = _config(ModelRole.EXECUTION)
    execution.model = execution.model_config.model_id
    execution.api_key = "fixture-key"
    execution.tool_schemas = []
    execution.agent = SimpleNamespace(name="fixture")

    asyncio.run(interaction._make_llm_call("system", []))
    asyncio.run(execution._make_llm_call("system", [], with_tools=False))

    assert [(call["role"], call["config"]) for call in captured] == [
        (ModelRole.INTERACTION, interaction.model_config),
        (ModelRole.EXECUTION, execution.model_config),
    ]


def test_email_search_summarizer_and_classifier_bind_exact_roles(monkeypatch) -> None:
    captured = []

    async def fake_request(**kwargs):
        captured.append(kwargs)
        if kwargs["role"] is ModelRole.CLASSIFIER:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "mark_email_importance",
                                        "arguments": {"important": False},
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "summary"}}]}

    monkeypatch.setattr(email_search, "request_chat_completion", fake_request)
    monkeypatch.setattr(summarizer, "request_chat_completion", fake_request)
    monkeypatch.setattr(importance_classifier, "request_chat_completion", fake_request)

    search_config = _config(ModelRole.EMAIL_SEARCH)
    asyncio.run(
        email_search._run_email_search(
            search_query="fixture",
            composio_user_id="fixture-user",
            model=search_config.model_id,
            api_key="fixture-key",
            model_config=search_config,
        )
    )
    summary_config = _config(ModelRole.SUMMARIZER)
    asyncio.run(
        summarizer._call_openrouter(
            SummaryPrompt(system_prompt="system", messages=[]),
            summary_config.model_id,
            "fixture-key",
            model_config=summary_config,
        )
    )
    classifier_config = _config(ModelRole.CLASSIFIER)
    monkeypatch.setattr(
        importance_classifier,
        "get_settings",
        lambda: SimpleNamespace(
            openrouter_api_key="fixture-key",
            email_classifier_model=classifier_config.model_id,
            model_call_config=lambda role: classifier_config,
        ),
    )
    email = ProcessedEmail(
        id="fixture-id",
        thread_id=None,
        query="fixture",
        subject="fixture",
        sender="fixture",
        recipient="fixture",
        timestamp=datetime.now(UTC),
        label_ids=[],
        clean_text="fixture",
        has_attachments=False,
        attachment_count=0,
        attachment_filenames=[],
    )
    asyncio.run(importance_classifier.classify_email_importance(email))

    assert [call["role"] for call in captured] == [
        ModelRole.EMAIL_SEARCH,
        ModelRole.SUMMARIZER,
        ModelRole.CLASSIFIER,
    ]
    assert [call["config"] for call in captured] == [
        search_config,
        summary_config,
        classifier_config,
    ]
