from __future__ import annotations

import asyncio

import pytest

from server.config import ModelCallConfig
from server.openrouter_client import OpenRouterError
from server.services.conversation.summarization import summarizer
from server.services.conversation.summarization.prompt_builder import SummaryPrompt


@pytest.mark.parametrize(
    "outcome",
    [OpenRouterError("transport failed"), {"choices": []}],
)
def test_measured_zero_retry_summarizer_makes_exactly_one_request(
    monkeypatch: pytest.MonkeyPatch, outcome
) -> None:
    calls = []

    async def fake_request(**kwargs):
        calls.append(kwargs)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(summarizer, "request_chat_completion", fake_request)
    config = ModelCallConfig(model_id="openai/gpt-4.1-mini", max_retries=0)

    with pytest.raises(OpenRouterError):
        asyncio.run(
            summarizer._call_openrouter(
                SummaryPrompt(system_prompt="system", messages=[]),
                config.model_id,
                "fixture-key",
                model_config=config,
            )
        )

    assert len(calls) == 1


def test_ordinary_summarizer_preserves_historical_second_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OpenRouterError("first failed")
        return {"choices": [{"message": {"content": "summary"}}]}

    monkeypatch.setattr(summarizer, "request_chat_completion", fake_request)

    result = asyncio.run(
        summarizer._call_openrouter(
            SummaryPrompt(system_prompt="system", messages=[]),
            "anthropic/claude-sonnet-4",
            "fixture-key",
        )
    )

    assert result == "summary"
    assert len(calls) == 2
