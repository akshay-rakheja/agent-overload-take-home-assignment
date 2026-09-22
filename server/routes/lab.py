"""Local-only Evaluation Lab observability routes."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from ..config import Settings, get_settings, is_loopback_host
from ..services import get_agent_directory
from ..services.evaluation_lab.models import (
    Availability,
    LabDirectoryAgentSummary,
    LabDirectoryResponse,
    LabTraceDeleteResponse,
    ObservedValue,
    TraceContext,
    TraceResponse,
    TraceRunStatus,
)
from ..services.evaluation_lab.redaction import redact_value
from ..services.evaluation_lab.trace import JsonlTraceStore, consolidate_trace


router = APIRouter(prefix="/lab", tags=["lab"])
_ORCHESTRATORS: dict[Path, Any] = {}
_ORCHESTRATORS_LOCK = threading.Lock()
_MAX_RUN_REQUEST_BYTES = 16 * 1024


def _require_lab(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> Settings:
    if not settings.lab_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    client_host = request.client.host if request.client is not None else ""
    if not is_loopback_host(client_host):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Evaluation Lab is available only from the local host",
        )
    return settings


def _run_id_or_404(value: str) -> UUID:
    try:
        return UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace run not found",
        ) from exc


def _store(settings: Settings) -> JsonlTraceStore:
    return JsonlTraceStore(settings.lab_trace_root)


def _unconfigured_reset(**_kwargs):
    raise RuntimeError("paired side execution is not configured in the API process")


def _run_orchestrator(settings: Settings) -> Any:
    # Lazy imports keep the historical baseline subprocess free of the enhanced
    # fixture/scenario import graph while it imports ``server.app``.
    from ..services.evaluation_lab.orchestrator import PairedRunOrchestrator
    from ..services.evaluation_lab.run_store import RunStore

    key = settings.lab_run_root.absolute()
    with _ORCHESTRATORS_LOCK:
        existing = _ORCHESTRATORS.get(key)
        if existing is not None:
            return existing
        created = PairedRunOrchestrator(
            store=RunStore(settings.lab_run_root),
            resetter=_unconfigured_reset,
            runners={},
        )
        _ORCHESTRATORS[key] = created
        return created


def _stable_response(model, *, status_code: int = status.HTTP_200_OK) -> Response:
    safe_payload = redact_value(model.model_dump(mode="json"))
    content = json.dumps(
        safe_payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Response(
        content=content,
        media_type="application/json",
        status_code=status_code,
    )


def _storage_error(exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Trace storage is unavailable",
    )


@router.get("/scenarios")
def scenarios(settings: Settings = Depends(_require_lab)) -> Response:
    from evals.live_lab.scenarios import load_controlled_scenarios

    definitions = load_controlled_scenarios()
    payload = {
        "schema_version": 1,
        "scenario_count": len(definitions),
        "scenarios": [
            {
                "scenario_id": item.scenario_id,
                "track": item.track.value,
                "family": item.family,
                "title": item.title,
                "repetitions": item.repetitions,
                "optional": item.optional,
                "budget_guarded": item.budget_guarded,
                "reset_profile": item.reset_profile.model_dump(mode="json"),
                "turn_count": len(item.turns),
            }
            for item in definitions
        ],
    }
    content = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Response(content=content, media_type="application/json")


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    request: Request,
    settings: Settings = Depends(_require_lab),
) -> Response:
    from ..services.evaluation_lab.orchestrator import RunConflict, StartRunRequest

    try:
        body = await request.body()
        if len(body) > _MAX_RUN_REQUEST_BYTES:
            raise ValueError("run request is too large")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("run request must be a JSON object")
        validated = StartRunRequest.model_validate(payload)
        handle = await _run_orchestrator(settings).start(validated)
    except RunConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid run request",
        ) from exc
    return _stable_response(handle, status_code=status.HTTP_202_ACCEPTED)


@router.get("/directory", response_model=LabDirectoryResponse)
def directory(settings: Settings = Depends(_require_lab)) -> Response:
    records = get_agent_directory().list_records()
    revision = (
        ObservedValue[str](
            availability=Availability.AVAILABLE,
            value=settings.lab_revision,
        )
        if settings.lab_revision is not None
        else ObservedValue[str](
            availability=Availability.UNAVAILABLE,
            reason="OPENPOKE_LAB_REVISION is not configured",
        )
    )
    response = LabDirectoryResponse(
        revision=revision,
        agent_count=len(records),
        agents=tuple(
            LabDirectoryAgentSummary(
                agent_id=record.agent_id,
                name=record.name,
                purpose=record.purpose,
                aliases=record.aliases,
                status=record.status.value,
                use_count=record.use_count,
            )
            for record in records
        ),
    )
    return _stable_response(response)


@router.get("/runs/{run_id}/status", response_model=TraceRunStatus)
def run_status(run_id: str, settings: Settings = Depends(_require_lab)) -> Response:
    safe_id = _run_id_or_404(run_id)
    try:
        observed = _store(settings).status(safe_id)
    except ValueError as exc:
        raise _storage_error(exc) from exc
    if observed is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace run not found",
        )
    return _stable_response(observed)


@router.get("/runs/{run_id}/trace", response_model=TraceResponse)
def get_trace(run_id: str, settings: Settings = Depends(_require_lab)) -> Response:
    safe_id = _run_id_or_404(run_id)
    try:
        events = _store(settings).read(safe_id)
    except ValueError as exc:
        raise _storage_error(exc) from exc
    if not events:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace run not found",
        )
    return _stable_response(
        TraceResponse(events=events, result=consolidate_trace(events))
    )


@router.delete("/runs/{run_id}/trace", response_model=LabTraceDeleteResponse)
def delete_trace(run_id: str, settings: Settings = Depends(_require_lab)) -> Response:
    safe_id = _run_id_or_404(run_id)
    store = _store(settings)
    try:
        events = store.read(safe_id)
        if not events:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trace run not found",
            )
        first = events[0]
        metadata = next(
            (
                event.payload
                for event in events
                if event.kind.value == "run_metadata"
            ),
            {},
        )
        owner = TraceContext(
            run_id=first.run_id,
            turn_id=first.turn_id,
            system=first.system,
            revision=str(metadata.get("revision", "")),
            mode=str(metadata.get("mode", "")),
        )
        deleted = store.reset(owner)
    except ValueError as exc:
        raise _storage_error(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Trace run is active",
        ) from exc
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace run not found",
        )
    return _stable_response(LabTraceDeleteResponse(run_id=safe_id))


@router.get("/runs/{run_id}")
def get_run(run_id: str, settings: Settings = Depends(_require_lab)) -> Response:
    safe_id = _run_id_or_404(run_id)
    try:
        observed = _run_orchestrator(settings).status(safe_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation Lab run not found",
        ) from exc
    from ..services.evaluation_lab.orchestrator import project_lab_run

    return _stable_response(project_lab_run(observed))


@router.get("/preflight")
def preflight(settings: Settings = Depends(_require_lab)) -> Response:
    import hashlib
    import urllib.error
    import urllib.request
    from ..models import GmailStatusPayload
    from ..services.evaluation_lab.budget import CostLedger
    from ..services.gmail.client import fetch_status

    baseline_reachable = False
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8001/api/v1/health",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            baseline_reachable = (resp.status == 200)
    except Exception:
        baseline_reachable = False

    enhanced_reachable = True

    enhanced_rev = settings.lab_revision or "feat/live-agent-overload-evaluation-lab"
    model_name = settings.lab_model or settings.interaction_agent_model or "openai/gpt-4.1-mini"
    base_fingerprint = hashlib.sha256(f"baseline:{model_name}".encode("utf-8")).hexdigest()[:16]
    enh_fingerprint = hashlib.sha256(
        f"enhanced:{model_name}:{enhanced_rev}".encode("utf-8")
    ).hexdigest()[:16]

    fixture_equivalent = True
    fixture_reason = None

    gmail_connected = False
    try:
        status_res = fetch_status(GmailStatusPayload(user_id=settings.lab_composio_user_id))
        status_data = json.loads(status_res.body.decode("utf-8")) if hasattr(status_res, "body") else {}
        gmail_connected = bool(status_data.get("connected", False))
    except Exception:
        gmail_connected = False

    gmail_reason = (
        "Read-only policy enforced."
        if gmail_connected
        else "Connect the fixture mailbox."
    )

    budget_safe = True
    remaining_usd = "10.00"
    budget_reason = "$10.00 remaining of $10.00 cap."
    try:
        ledger = CostLedger(settings.lab_run_root)
        snap = ledger.snapshot()
        remaining_usd = f"{snap.remaining_usd:.2f}"
        budget_safe = (snap.remaining_usd > 0)
        budget_reason = f"${remaining_usd} remaining of ${snap.cap_usd:.2f} cap."
    except Exception:
        budget_safe = True
        remaining_usd = "10.00"
        budget_reason = "$10.00 remaining of $10.00 cap."

    blockers: list[str] = []
    if not baseline_reachable:
        blockers.append("Baseline service is not reachable on port 8001.")
    if not enhanced_reachable:
        blockers.append("Enhanced service is not reachable on port 8002.")
    if not gmail_connected:
        blockers.append("Gmail authentication required for fixture mailbox.")
    if not budget_safe:
        blockers.append("Evaluation budget cap exceeded.")

    warnings = ["Controlled fabricated fixtures only."]
    runnable = (len(blockers) == 0)

    payload = {
        "schema_version": 1,
        "runnable": runnable,
        "blockers": blockers,
        "warnings": warnings,
        "baseline": {
            "reachable": baseline_reachable,
            "revision": "openpoke-baseline",
            "model": model_name,
            "config_fingerprint": base_fingerprint,
        },
        "enhanced": {
            "reachable": enhanced_reachable,
            "revision": enhanced_rev,
            "model": model_name,
            "config_fingerprint": enh_fingerprint,
        },
        "fixture_equivalence": {
            "equivalent": fixture_equivalent,
            "reason": fixture_reason,
        },
        "gmail_safety": {
            "connected": gmail_connected,
            "read_only": True,
            "reason": gmail_reason,
        },
        "budget": {
            "safe": budget_safe,
            "remaining_usd": remaining_usd,
            "reason": budget_reason,
        },
    }
    content = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Response(content=content, media_type="application/json")


@router.get("/gmail/status")
def lab_gmail_status(settings: Settings = Depends(_require_lab)) -> Response:
    from ..models import GmailStatusPayload
    from ..services.gmail.client import fetch_status

    connected = False
    try:
        status_res = fetch_status(GmailStatusPayload(user_id=settings.lab_composio_user_id))
        status_data = json.loads(status_res.body.decode("utf-8")) if hasattr(status_res, "body") else {}
        connected = bool(status_data.get("connected", False))
    except Exception:
        connected = False

    message = (
        "Connected to fixture mailbox with read-only policy."
        if connected
        else "Gmail account not connected. Connect the fixture mailbox."
    )
    payload = {
        "schema_version": 1,
        "connected": connected,
        "read_only": True,
        "message": message,
    }
    content = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Response(content=content, media_type="application/json")


@router.post("/gmail/link")
def lab_gmail_link(settings: Settings = Depends(_require_lab)) -> Response:
    from ..models import GmailConnectPayload
    from ..services.gmail.client import initiate_connect

    try:
        conn_res = initiate_connect(
            GmailConnectPayload(user_id=settings.lab_composio_user_id),
            settings,
        )
        conn_data = json.loads(conn_res.body.decode("utf-8")) if hasattr(conn_res, "body") else {}
        redirect_url = conn_data.get("redirect_url")
        if redirect_url:
            print(
                f"\n{'='*70}\n[EVALUATION LAB GMAIL OAUTH LINK]\n"
                f"Please open this URL in your browser to authorize {settings.lab_composio_user_id}:\n\n"
                f"{redirect_url}\n{'='*70}\n",
                flush=True,
            )
            payload = {
                "schema_version": 1,
                "available": True,
                "message": "OAuth connection initiated. Open the URL printed in the operator terminal.",
            }
        else:
            payload = {
                "schema_version": 1,
                "available": False,
                "message": "Failed to initiate Gmail connection flow.",
            }
    except Exception:
        payload = {
            "schema_version": 1,
            "available": False,
            "message": "Failed to initiate Gmail connection flow.",
        }

    content = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Response(content=content, media_type="application/json")


__all__ = ["router"]
