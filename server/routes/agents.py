from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..services.execution.inspector import get_inspector_state

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get(
    "/inspector",
    response_class=JSONResponse,
    summary="Inspect execution agent roster, candidates, and routing state",
)
def inspect_agents(request: Request) -> dict[str, Any]:
    """Return execution agent directory, candidates evaluated, and latest turn action."""
    system_param = request.query_params.get("system")
    port = request.url.port
    system_override = system_param or ("baseline" if port == 8001 else "enhanced" if port == 8002 else None)
    return get_inspector_state(system_override=system_override)


__all__ = ["router"]
