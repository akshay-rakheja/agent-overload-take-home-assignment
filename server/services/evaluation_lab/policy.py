"""Pure allowlist policy for Evaluation Lab Gmail and trigger operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


MUTATION_BLOCKED_REASON = (
    "Mutating Gmail and trigger tools are disabled in Evaluation Lab mode."
)
UNKNOWN_BLOCKED_REASON = (
    "Unclassified Gmail and trigger tools are disabled in Evaluation Lab mode."
)

_APPROVED_MODEL_TOOLS = frozenset(
    {
        "task_email_search",
        "gmail_get_contacts",
        "gmail_get_people",
        "gmail_list_drafts",
        "gmail_search_people",
        "listTriggers",
    }
)
_APPROVED_COMPOSIO_TOOLS = frozenset(
    {
        "GMAIL_GET_PROFILE",
        "GMAIL_FETCH_EMAILS",
        "GMAIL_GET_CONTACTS",
        "GMAIL_GET_PEOPLE",
        "GMAIL_LIST_DRAFTS",
        "GMAIL_SEARCH_PEOPLE",
    }
)
_MODEL_MUTATION_MARKERS = (
    "create",
    "send",
    "execute",
    "reply",
    "forward",
    "delete",
    "label",
    "filter",
)
_COMPOSIO_MUTATION_MARKERS = (
    "CREATE",
    "SEND",
    "REPLY",
    "FORWARD",
    "DELETE",
    "LABEL",
    "FILTER",
)


@dataclass(frozen=True)
class ToolPolicyDecision:
    allowed: bool
    code: Literal["allowed_read_only", "mutation_blocked", "unknown_blocked"]
    reason: str


class LabToolPolicy:
    """Classify lab tools using explicit read allowlists and fail-closed defaults."""

    def decide_model_tool(self, tool_name: str) -> ToolPolicyDecision:
        if tool_name in _APPROVED_MODEL_TOOLS:
            return ToolPolicyDecision(
                allowed=True,
                code="allowed_read_only",
                reason="Read-only Gmail and trigger tools are allowed in Evaluation Lab mode.",
            )

        normalized = (tool_name or "").lower()
        if (
            normalized == "send_draft"
            or normalized.startswith("gmail_")
            and any(marker in normalized for marker in _MODEL_MUTATION_MARKERS)
            or "trigger" in normalized
            and any(marker in normalized for marker in ("create", "update", "delete"))
        ):
            return ToolPolicyDecision(
                allowed=False,
                code="mutation_blocked",
                reason=MUTATION_BLOCKED_REASON,
            )

        return ToolPolicyDecision(
            allowed=False,
            code="unknown_blocked",
            reason=UNKNOWN_BLOCKED_REASON,
        )

    def decide_composio_tool(self, tool_name: str) -> ToolPolicyDecision:
        if tool_name in _APPROVED_COMPOSIO_TOOLS:
            return ToolPolicyDecision(
                allowed=True,
                code="allowed_read_only",
                reason="Read-only Gmail tools are allowed in Evaluation Lab mode.",
            )

        normalized = (tool_name or "").upper()
        if normalized.startswith("GMAIL_") and any(
            marker in normalized for marker in _COMPOSIO_MUTATION_MARKERS
        ):
            return ToolPolicyDecision(
                allowed=False,
                code="mutation_blocked",
                reason=MUTATION_BLOCKED_REASON,
            )

        return ToolPolicyDecision(
            allowed=False,
            code="unknown_blocked",
            reason=UNKNOWN_BLOCKED_REASON,
        )


def lab_tool_rejection(
    tool_name: str, decision: ToolPolicyDecision
) -> dict[str, dict[str, Any]]:
    """Render the structured rejection shared by every lab enforcement boundary."""

    return {
        "error": {
            "code": "lab_mutation_blocked",
            "tool": tool_name,
            "reason": decision.reason,
        }
    }
