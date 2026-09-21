from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING

from ....config import ModelCallConfig, ModelRole, get_settings
from ....logging_config import logger
from ....openrouter_client import OpenRouterError, request_chat_completion
from .prompt_builder import SummaryPrompt, build_summarization_prompt
from .state import LogEntry, SummaryState
from .working_memory_log import get_working_memory_log

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..log import ConversationLog


def _resolve_conversation_log() -> "ConversationLog":
    from ..log import get_conversation_log

    return get_conversation_log()


def _collect_entries(log) -> List[LogEntry]:
    entries: List[LogEntry] = []
    for index, (tag, timestamp, payload) in enumerate(log.iter_entries()):
        entries.append(LogEntry(tag=tag, payload=payload, index=index, timestamp=timestamp or None))
    return entries


async def _call_openrouter(
    prompt: SummaryPrompt,
    model: str,
    api_key: Optional[str],
    *,
    model_config: ModelCallConfig | None = None,
) -> str:
    last_error: Exception | None = None
    measured_zero_retry = (
        model_config is not None
        and "max_retries" in model_config.model_fields_set
        and model_config.max_retries == 0
    )
    attempts = 1 if measured_zero_retry else 2
    for attempt in range(attempts):
        try:
            response = await request_chat_completion(
                config=model_config or ModelCallConfig(model_id=model),
                role=ModelRole.SUMMARIZER,
                messages=prompt.messages,
                system=prompt.system_prompt,
                api_key=api_key,
            )
            choices = response.get("choices") or []
            if not choices:
                raise OpenRouterError("OpenRouter response missing choices")
            message = choices[0].get("message") or {}
            content = (message.get("content") or "").strip()
            if content:
                return content
            raise OpenRouterError("OpenRouter response missing content")
        except OpenRouterError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                logger.warning(
                    "conversation summarization attempt failed; retrying",
                    extra={"error": str(exc)},
                )
                continue
            logger.error(
                "conversation summarization failed",
                extra={"error": str(exc)},
            )
            break
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            logger.error(
                "conversation summarization unexpected failure",
                extra={"error": str(exc)},
            )
            break
    if last_error:
        raise last_error
    raise OpenRouterError("Conversation summarization failed")


async def summarize_conversation() -> bool:
    settings = get_settings()
    if not settings.summarization_enabled:
        return False

    conversation_log = _resolve_conversation_log()
    working_memory_log = get_working_memory_log()

    entries = _collect_entries(conversation_log)
    state = working_memory_log.load_summary_state()

    threshold = settings.conversation_summary_threshold
    tail_size = max(settings.conversation_summary_tail_size, 0)

    if threshold <= 0:
        return False

    unsummarized_entries = [entry for entry in entries if entry.index > state.last_index]
    if len(unsummarized_entries) < threshold + tail_size:
        return False

    batch = unsummarized_entries[:threshold]
    cutoff_index = batch[-1].index

    prompt = build_summarization_prompt(state.summary_text, batch)

    logger.info(
        "conversation summarization started",
        extra={
            "entries_total": len(entries),
            "unsummarized": len(unsummarized_entries),
            "batch_size": len(batch),
            "last_index_before": state.last_index,
            "cutoff_index": cutoff_index,
        },
    )

    summary_config = settings.model_call_config(ModelRole.SUMMARIZER)
    summary_text = await _call_openrouter(
        prompt,
        summary_config.model_id,
        settings.openrouter_api_key,
        model_config=summary_config,
    )
    summary_body = summary_text if summary_text else state.summary_text

    refreshed_entries = _collect_entries(conversation_log)
    remaining_entries = [entry for entry in refreshed_entries if entry.index > cutoff_index]

    new_state = SummaryState(
        summary_text=summary_body,
        last_index=cutoff_index,
        updated_at=datetime.now(timezone.utc),
        unsummarized_entries=remaining_entries,
    )

    working_memory_log.write_summary_state(new_state)

    logger.info(
        "conversation summarization completed",
        extra={
            "last_index_after": new_state.last_index,
            "remaining_unsummarized": len(new_state.unsummarized_entries),
        },
    )
    return True


__all__ = ["summarize_conversation"]
