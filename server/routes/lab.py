"""Local-only Evaluation Lab observability routes."""

from __future__ import annotations

import json
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


def _stable_response(model) -> Response:
    safe_payload = redact_value(model.model_dump(mode="json"))
    content = json.dumps(
        safe_payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return Response(content=content, media_type="application/json")


def _storage_error(exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Trace storage is unavailable",
    )


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


__all__ = ["router"]
