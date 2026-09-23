from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

from server.app import app
from server.services.execution.inspector import (
    clear_inspector_state,
    finalize_turn_inspection,
    get_inspector_state,
    record_tool_dispatch,
    record_turn_candidate_context,
)
from server.services.execution.models import AgentStatus


def test_agent_inspector_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/agents/inspector")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["system"] in {"baseline", "enhanced"}
    assert isinstance(data["roster"], list)
    assert isinstance(data["candidate_ids"], list)
    assert isinstance(data["candidates"], list)


def test_inspector_lifecycle_tracking():
    clear_inspector_state()
    
    agent_id = str(uuid4())
    mock_candidate = MagicMock()
    mock_candidate.agent_id = agent_id
    mock_candidate.name = "Test Security Agent"
    mock_candidate.purpose = "Check emails for security alerts"
    mock_candidate.status = AgentStatus.HOT
    mock_candidate.score = 0.95
    mock_candidate.reasons = ["Keyword match"]

    mock_context = MagicMock()
    mock_context.prompt_candidates = [mock_candidate]
    mock_decision = MagicMock()
    mock_decision.action.value = "reuse"
    mock_decision.agent_id = agent_id
    mock_context.decision = mock_decision

    record_turn_candidate_context(mock_context)
    state = get_inspector_state()
    assert len(state["candidates"]) == 1
    assert state["candidates"][0]["name"] == "Test Security Agent"
    assert state["latest_turn"]["routing_action"] == "reuse"

    # Tool dispatch
    record_tool_dispatch("send_message_to_agent", {"agent_id": agent_id, "instructions": "Search inbox"})
    state = get_inspector_state()
    assert state["latest_turn"]["selected_agent_id"] == agent_id
    assert state["latest_turn"]["instructions"] == "Search inbox"

    finalize_turn_inspection("What was found?", "Found 1 alert.")
    state = get_inspector_state()
    assert state["latest_turn"]["user_message"] == "What was found?"
    assert state["latest_turn"]["response"] == "Found 1 alert."

    # Test clear
    clear_inspector_state()
    state = get_inspector_state()
    assert state["latest_turn"] is None
    if state["system"] == "baseline":
        assert len(state["candidates"]) == len(state["roster"])
    else:
        assert state["candidates"] == []


def test_inspector_abstain_lifecycle():
    clear_inspector_state()
    
    # When no tool dispatch occurs and turn finalizes, it should be marked as abstain
    finalize_turn_inspection("Can you write a draft?", "Here is your draft: ...")
    state = get_inspector_state()
    assert state["latest_turn"]["routing_action"] == "abstain"
    assert state["latest_turn"]["user_message"] == "Can you write a draft?"
    
    clear_inspector_state()


def test_inspector_jev_lifecycle_and_endpoint():
    clear_inspector_state()

    client = TestClient(app)
    res_det = client.get("/api/v1/agents/inspector?system=enhanced_deterministic")
    assert res_det.status_code == 200
    assert res_det.json()["system"] == "enhanced_deterministic"

    res_jev = client.get("/api/v1/agents/inspector?system=enhanced_jev")
    assert res_jev.status_code == 200
    assert res_jev.json()["system"] == "enhanced_jev"

    # Test Jev lifecycle with Jev context
    mock_candidate = MagicMock()
    mock_candidate.agent_id = str(uuid4())
    mock_candidate.name = "Instagram Scanner"
    mock_candidate.purpose = "Scan Instagram emails"
    mock_candidate.status = AgentStatus.HOT
    mock_candidate.score = 0.88
    mock_candidate.reasons = ["Semantic affinity"]

    mock_context = MagicMock()
    mock_context.prompt_candidates = [mock_candidate]
    mock_decision = MagicMock()
    mock_decision.action.value = "reuse"
    mock_decision.agent_id = mock_candidate.agent_id
    mock_context.decision = mock_decision

    mock_jev_context = MagicMock()
    mock_jev_dec = MagicMock()
    mock_jev_dec.action.value = "reuse"
    mock_jev_dec.confidence = 0.92
    mock_jev_dec.winner_margin = 0.45
    mock_jev_dec.rationale = "High confidence affinity"
    mock_jev_context.decision = mock_jev_dec
    mock_jev_context.map_latency_ms = 40.0
    mock_jev_context.reduce_latency_ms = 12.0
    mock_jev_context.total_latency_ms = 52.0
    mock_jev_context.api_calls_count = 2
    mock_jev_context.total_tokens = 350
    mock_jev_context.map_scores = ()
    mock_jev_context.shortlist = ()

    record_turn_candidate_context(
        mock_context,
        system="enhanced_jev",
        jev_context=mock_jev_context,
    )

    state_jev = get_inspector_state("enhanced_jev")
    assert state_jev["system"] == "enhanced_jev"
    assert len(state_jev["candidates"]) == 1
    assert state_jev["latest_turn"]["routing_action"] == "reuse"
    assert state_jev["latest_turn"]["confidence"] == 0.92
    assert "jev_details" in state_jev
    assert state_jev["jev_details"]["winner_margin"] == 0.45

    # Confirm isolation: deterministic inspector state is unchanged
    state_det = get_inspector_state("enhanced_deterministic")
    assert state_det["latest_turn"] is None

    clear_inspector_state()
