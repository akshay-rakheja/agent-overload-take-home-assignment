"""Execution agent inspection service for live multi-agent observability."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

from ...config import get_settings
from .models import AgentRecord
from .roster import get_agent_directory

_INSPECTOR_LOCK = threading.RLock()
_LATEST_TURN: dict[str, Any] | None = None
_LATEST_CANDIDATES: list[dict[str, Any]] = []

_TURNS: dict[str, dict[str, Any]] = {}
_CANDIDATES: dict[str, list[dict[str, Any]]] = {}
_JEV_CONTEXTS: dict[str, Any] = {}


def _normalize_system_key(system: str | None) -> str:
    if system in ("enhanced_jev", "jev"):
        return "enhanced_jev"
    if system in ("baseline",):
        return "baseline"
    return "enhanced_deterministic"


def get_system_name() -> str:
    env_system = os.environ.get("OPENPOKE_SYSTEM")
    if env_system in {"baseline", "enhanced", "enhanced_deterministic", "enhanced_jev"}:
        return env_system
    settings = get_settings()
    port = getattr(settings, "server_port", None)
    if port == 8001:
        return "baseline"
    if port == 8002:
        return "enhanced"
    return "enhanced" if getattr(settings, "lab_enabled", False) else "baseline"


def record_turn_candidate_context(
    context: Any,
    system: str = "enhanced_deterministic",
    jev_context: Any = None,
) -> None:
    """Record candidates exposed to the interaction agent on this turn."""
    global _LATEST_CANDIDATES, _LATEST_TURN, _CANDIDATES, _TURNS, _JEV_CONTEXTS
    key = _normalize_system_key(system)
    with _INSPECTOR_LOCK:
        candidates_data = []
        candidates_list = getattr(context, "prompt_candidates", None) or getattr(context, "candidates", [])
        for rank, c in enumerate(candidates_list, start=1):
            candidates_data.append({
                "rank": rank,
                "agent_id": str(getattr(c, "agent_id", "")),
                "name": getattr(c, "name", "agent"),
                "purpose": getattr(c, "purpose", ""),
                "status": getattr(c.status, "value", str(getattr(c, "status", "hot"))),
                "score": getattr(c, "score", None),
                "reasons": list(getattr(c, "reasons", [])),
            })
        _CANDIDATES[key] = candidates_data
        if key == "enhanced_deterministic":
            _LATEST_CANDIDATES = candidates_data
        
        decision = getattr(context, "decision", None)
        action_val = getattr(getattr(decision, "action", None), "value", None) if decision else None
        target_id = str(getattr(decision, "agent_id", "")) if getattr(decision, "agent_id", None) else None
        
        target_name = None
        if target_id:
            cand = next((c for c in candidates_data if c["agent_id"] == target_id), None)
            if cand:
                target_name = cand.get("name")
        
        turn_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "routing_action": action_val or "pending",
            "recommended_action": action_val,
            "recommended_agent_id": target_id,
            "selected_agent_id": target_id if action_val == "reuse" else None,
            "selected_agent_name": target_name if action_val == "reuse" else None,
            "instructions": None,
            "tool_calls": [],
        }

        if jev_context is not None:
            _JEV_CONTEXTS[key] = jev_context
            jev_dec = getattr(jev_context, "decision", None)
            if jev_dec:
                turn_data["confidence"] = getattr(jev_dec, "confidence", None)
                turn_data["winner_margin"] = getattr(jev_dec, "winner_margin", None)
                turn_data["rationale"] = getattr(jev_dec, "rationale", None)

        _TURNS[key] = turn_data
        if key == "enhanced_deterministic":
            _LATEST_TURN = turn_data


def record_tool_dispatch(
    tool_name: str,
    arguments: dict[str, Any],
    system: str = "enhanced_deterministic",
) -> None:
    """Record a tool dispatch during the interaction turn."""
    global _LATEST_TURN, _TURNS
    key = _normalize_system_key(system)
    with _INSPECTOR_LOCK:
        if key not in _TURNS or _TURNS[key] is None:
            _TURNS[key] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "routing_action": "pending",
                "recommended_action": None,
                "recommended_agent_id": None,
                "selected_agent_id": None,
                "selected_agent_name": None,
                "instructions": None,
                "tool_calls": [],
            }
        
        active_turn = _TURNS[key]
        active_turn["tool_calls"].append({
            "tool": tool_name,
            "arguments": arguments,
        })

        if tool_name == "send_message_to_agent":
            agent_id = arguments.get("agent_id")
            agent_name = arguments.get("agent_name")
            instructions = arguments.get("instructions")
            if agent_id:
                active_turn["routing_action"] = "reuse"
                active_turn["selected_agent_id"] = str(agent_id)
                cand_list = _CANDIDATES.get(key, [])
                cand = next((c for c in cand_list if str(c.get("agent_id")) == str(agent_id)), None)
                if cand and cand.get("name"):
                    active_turn["selected_agent_name"] = cand.get("name")
                else:
                    try:
                        rec = get_agent_directory(key).get_record(agent_id)
                        if rec:
                            active_turn["selected_agent_name"] = rec.name
                    except Exception:
                        pass
            elif agent_name:
                active_turn["routing_action"] = "create_new"
                active_turn["selected_agent_name"] = str(agent_name)
            else:
                active_turn["routing_action"] = "create_new"
            if instructions:
                active_turn["instructions"] = instructions

        if key == "enhanced_deterministic":
            _LATEST_TURN = active_turn


def record_dispatch_result(
    arguments: dict[str, Any],
    result: Any,
    system: str = "enhanced_deterministic",
) -> None:
    """Record the resolved agent identity after tool execution."""
    global _LATEST_TURN, _TURNS
    key = _normalize_system_key(system)
    with _INSPECTOR_LOCK:
        if key not in _TURNS or _TURNS[key] is None:
            return
        
        active_turn = _TURNS[key]
        payload = getattr(result, "payload", {}) if isinstance(getattr(result, "payload", None), dict) else {}
        agent_id = payload.get("agent_id") or arguments.get("agent_id")
        agent_name = payload.get("agent_name") or arguments.get("agent_name")
        
        if not agent_name and agent_id:
            try:
                rec = get_agent_directory(key).get_record(agent_id)
                if rec:
                    agent_name = rec.name
            except Exception:
                pass
        
        if not agent_name:
            try:
                records = get_agent_directory(key).list_records()
                if records:
                    latest_rec = max(records, key=lambda r: r.created_at or datetime.min)
                    agent_name = latest_rec.name
                    if not agent_id:
                        agent_id = str(latest_rec.agent_id)
            except Exception:
                pass

        if agent_name:
            active_turn["selected_agent_name"] = agent_name
        if agent_id:
            active_turn["selected_agent_id"] = str(agent_id)

        if key == "enhanced_deterministic":
            _LATEST_TURN = active_turn


def record_agent_callback_response(
    agent_message: str,
    response: str,
    system: str = "enhanced_deterministic",
) -> None:
    """Update response of the latest turn when an execution agent finishes its work."""
    global _LATEST_TURN, _TURNS
    key = _normalize_system_key(system)
    with _INSPECTOR_LOCK:
        if key in _TURNS and _TURNS[key] is not None:
            _TURNS[key]["response"] = response
        if key == "enhanced_deterministic" and _LATEST_TURN is not None:
            _LATEST_TURN["response"] = response


def finalize_turn_inspection(
    user_message: str,
    response: str,
    system: str = "enhanced_deterministic",
) -> None:
    """Finalize the turn. If no agent dispatch tool was called, mark as abstain."""
    global _LATEST_TURN, _TURNS
    key = _normalize_system_key(system)
    with _INSPECTOR_LOCK:
        if key not in _TURNS or _TURNS[key] is None:
            _TURNS[key] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "routing_action": "abstain",
                "recommended_action": None,
                "recommended_agent_id": None,
                "selected_agent_id": None,
                "selected_agent_name": None,
                "instructions": None,
                "tool_calls": [],
            }
        else:
            dispatched = any(
                t.get("tool") == "send_message_to_agent"
                for t in _TURNS[key].get("tool_calls", [])
            )
            if not dispatched:
                _TURNS[key]["routing_action"] = "abstain"
                _TURNS[key]["selected_agent_id"] = None
                _TURNS[key]["selected_agent_name"] = None
                _TURNS[key]["instructions"] = None
        
        _TURNS[key]["user_message"] = user_message
        _TURNS[key]["response"] = response

        if key == "enhanced_deterministic":
            _LATEST_TURN = _TURNS[key]


def clear_inspector_state(system: str | None = None) -> None:
    """Reset the turn inspection state."""
    global _LATEST_TURN, _LATEST_CANDIDATES, _TURNS, _CANDIDATES, _JEV_CONTEXTS
    with _INSPECTOR_LOCK:
        if system is not None:
            key = _normalize_system_key(system)
            _TURNS.pop(key, None)
            _CANDIDATES.pop(key, None)
            _JEV_CONTEXTS.pop(key, None)
            if key == "enhanced_deterministic":
                _LATEST_TURN = None
                _LATEST_CANDIDATES = []
        else:
            _LATEST_TURN = None
            _LATEST_CANDIDATES = []
            _TURNS.clear()
            _CANDIDATES.clear()
            _JEV_CONTEXTS.clear()


def get_inspector_state(system_override: str | None = None) -> dict[str, Any]:
    """Retrieve full inspector state including roster, candidates, and latest routing action."""
    with _INSPECTOR_LOCK:
        system = system_override or get_system_name()
        key = _normalize_system_key(system)
        directory = get_agent_directory(key)
        records: list[AgentRecord] = directory.list_records()
        
        roster_data = [
            {
                "agent_id": str(r.agent_id),
                "name": r.name,
                "purpose": r.purpose,
                "status": r.status.value,
                "use_count": r.use_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
            }
            for r in records
        ]

        
        candidates = list(_CANDIDATES.get(key, [])) if key in _CANDIDATES else list(_LATEST_CANDIDATES)
        # If in baseline mode and no candidates recorded yet, reflect full roster exposure for baseline
        if key == "baseline" and not candidates and roster_data:
            candidates = [
                {
                    "rank": idx,
                    "agent_id": r["agent_id"],
                    "name": r["name"],
                    "purpose": r["purpose"],
                    "status": r["status"],
                    "score": None,
                    "reasons": ["Historical full-roster exposure"],
                }
                for idx, r in enumerate(roster_data, start=1)
            ]

        latest_turn_raw = _TURNS.get(key) if key in _TURNS else _LATEST_TURN
        latest_turn = dict(latest_turn_raw) if latest_turn_raw else None
        if latest_turn:
            if not latest_turn.get("selected_agent_name") and latest_turn.get("selected_agent_id"):
                match = next((r for r in roster_data if r["agent_id"] == latest_turn["selected_agent_id"]), None)
                if match:
                    latest_turn["selected_agent_name"] = match["name"]
            elif not latest_turn.get("selected_agent_id") and latest_turn.get("selected_agent_name"):
                match = next((r for r in roster_data if r["name"] == latest_turn["selected_agent_name"]), None)
                if match:
                    latest_turn["selected_agent_id"] = match["agent_id"]

        candidate_ids = [c["agent_id"] for c in candidates if c.get("agent_id")]

        jev_details = None
        jev_ctx = _JEV_CONTEXTS.get(key)
        if jev_ctx is not None:
            dec = getattr(jev_ctx, "decision", None)
            jev_details = {
                "action": getattr(getattr(dec, "action", None), "value", None) if dec else None,
                "confidence": getattr(dec, "confidence", None) if dec else None,
                "winner_margin": getattr(dec, "winner_margin", None) if dec else None,
                "rationale": getattr(dec, "rationale", None) if dec else None,
                "map_latency_ms": getattr(jev_ctx, "map_latency_ms", None),
                "reduce_latency_ms": getattr(jev_ctx, "reduce_latency_ms", None),
                "total_latency_ms": getattr(jev_ctx, "total_latency_ms", None),
                "api_calls_count": getattr(jev_ctx, "api_calls_count", 0),
                "total_tokens": getattr(jev_ctx, "total_tokens", 0),
                "map_scores": [
                    {
                        "agent_id": str(s.agent_id),
                        "composite_score": s.composite_score,
                        "affinity_score": s.affinity_score,
                        "continuity_score": s.continuity_score,
                        "risk_score": s.risk_score,
                        "reasoning": s.reasoning,
                    }
                    for s in getattr(jev_ctx, "map_scores", ())
                ],
                "shortlist": [
                    str(s.agent_id) if hasattr(s, "agent_id") else str(s)
                    for s in getattr(jev_ctx, "shortlist", ())
                ],
            }

        result_payload: dict[str, Any] = {
            "ok": True,
            "system": system,
            "roster_count": len(roster_data),
            "roster": roster_data,
            "candidate_ids": candidate_ids,
            "candidates": candidates,
            "latest_turn": latest_turn,
        }
        if jev_details is not None:
            result_payload["jev_details"] = jev_details

        return result_payload
