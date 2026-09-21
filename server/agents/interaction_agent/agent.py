"""Interaction agent helpers for prompt construction."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List

from ...config import MAX_AGENT_CANDIDATES, get_settings
from ...services.evaluation_lab.models import TraceEventKind
from ...services.evaluation_lab.trace import emit_trace
from ...services.execution import (
    AgentCandidate,
    AgentDirectory,
    AgentRetriever,
    AgentRouter,
    RetrievalQuery,
    RoutingDecision,
    get_agent_directory,
)

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class CandidateContext:
    """Bounded candidate set and the deterministic policy recommendation."""

    candidates: tuple[AgentCandidate, ...]
    decision: RoutingDecision

    @property
    def prompt_candidates(self) -> tuple[AgentCandidate, ...]:
        """Return the final safety-bounded candidate set visible to the model."""

        return self.candidates[:MAX_AGENT_CANDIDATES]


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
    *,
    directory: AgentDirectory | None = None,
    candidate_context: CandidateContext | None = None,
) -> List[Dict[str, str]]:
    """Compose a message with history, a bounded candidate set, and the latest turn."""
    sections: List[str] = []
    candidate_context = candidate_context or build_candidate_context(
        latest_text, transcript, directory=directory
    )

    sections.append(_render_conversation_history(transcript))
    sections.append(render_agent_candidates(candidate_context))
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def build_candidate_context(
    latest_text: str,
    transcript: str,
    *,
    directory: AgentDirectory | None = None,
) -> CandidateContext:
    """Retrieve and route using only the current turn and bounded working context."""

    resolved_directory = directory or get_agent_directory()
    context_limit = get_settings().agent_routing_context_max_characters
    bounded_transcript = transcript[-context_limit:]
    query = RetrievalQuery(text=latest_text, conversation_context=bounded_transcript)
    candidates = AgentRetriever(resolved_directory.list_records).retrieve(query)
    decision = AgentRouter().route(query, candidates)
    context = CandidateContext(candidates=tuple(candidates), decision=decision)
    emit_trace(
        TraceEventKind.CANDIDATES,
        {
            "candidate_count": len(context.prompt_candidates),
            "candidates": [
                {
                    "agent_id": str(candidate.agent_id),
                    "name": candidate.name,
                    "purpose": candidate.purpose,
                    "status": candidate.status.value,
                    "score": candidate.score,
                    "score_components": candidate.score_components,
                    "reasons": candidate.reasons,
                }
                for candidate in context.prompt_candidates
            ],
        },
    )
    return context


def render_agent_candidates(context: CandidateContext) -> str:
    """Render stable IDs and concise evidence without numeric certainty claims."""

    action = escape(context.decision.action.value, quote=True)
    rendered = [f'<agent_candidates routing_action="{action}">']
    if not context.candidates:
        rendered.append("None")
    else:
        for candidate in context.prompt_candidates:
            identifier = escape(str(candidate.agent_id), quote=True)
            name = escape(candidate.name or "agent", quote=True)
            purpose = escape(candidate.purpose, quote=True)
            status = escape(candidate.status.value, quote=True)
            hints = escape("; ".join(candidate.reasons), quote=False)
            rendered.append(
                f'<agent_candidate id="{identifier}" name="{name}" '
                f'purpose="{purpose}" status="{status}">'
                f"{hints}</agent_candidate>"
            )
    rendered.append("</agent_candidates>")
    candidate_xml = "\n".join(rendered)
    emit_trace(
        TraceEventKind.PROMPT_EXPOSURE,
        {
            "surface": "interaction_agent_candidate_xml",
            "candidate_count": len(context.prompt_candidates),
            "candidate_ids": [
                str(candidate.agent_id) for candidate in context.prompt_candidates
            ],
            "routing_action": context.decision.action.value,
            "candidate_xml": candidate_xml,
        },
    )
    return candidate_xml


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
