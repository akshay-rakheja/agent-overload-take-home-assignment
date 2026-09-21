"""Aggregate execution agent tool schemas and registries."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ....config import Settings, get_settings
from ....services.evaluation_lab import LabToolPolicy
from . import gmail, triggers
from ..tasks import get_task_registry, get_task_schemas


# Return OpenAI/OpenRouter-compatible tool schemas
def get_tool_schemas(*, settings: Settings | None = None) -> List[Dict[str, Any]]:
    """Return OpenAI/OpenRouter-compatible tool schemas."""

    schemas = [
        *gmail.get_schemas(),
        *get_task_schemas(),
        *triggers.get_schemas(),
    ]
    resolved_settings = settings or get_settings()
    if not resolved_settings.lab_enabled:
        return schemas

    policy = LabToolPolicy()
    return [
        schema
        for schema in schemas
        if policy.decide_model_tool(schema["function"]["name"]).allowed
    ]


# Return Python callables for executing tools by name
def get_tool_registry(
    agent_name: str, *, settings: Settings | None = None
) -> Dict[str, Callable[..., Any]]:
    """Return Python callables for executing tools by name."""

    registry: Dict[str, Callable[..., Any]] = {}
    registry.update(gmail.build_registry(agent_name))
    registry.update(get_task_registry(agent_name))
    registry.update(triggers.build_registry(agent_name))
    resolved_settings = settings or get_settings()
    if not resolved_settings.lab_enabled:
        return registry

    policy = LabToolPolicy()
    return {
        name: tool
        for name, tool in registry.items()
        if policy.decide_model_tool(name).allowed
    }


__all__ = [
    "get_tool_registry",
    "get_tool_schemas",
]
