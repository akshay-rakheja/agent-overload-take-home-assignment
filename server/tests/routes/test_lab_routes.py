"""Guarded, direct-fact Evaluation Lab route contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import Settings, get_settings
from server.routes import api_router
from server.services.evaluation_lab.models import TraceEvent, TraceEventKind, TraceResponse
from server.services.evaluation_lab.trace import JsonlTraceStore, consolidate_trace
from server.services.execution.directory import AgentDirectory


RUN_A = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
RUN_B = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
TURN_A = UUID("11111111-1111-4111-8111-111111111111")
TURN_B = UUID("22222222-2222-4222-8222-222222222222")
OCCURRED_AT = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def _event(
    run_id: UUID,
    turn_id: UUID,
    sequence: int,
    kind: TraceEventKind,
    payload: dict[str, object],
) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        occurred_at=OCCURRED_AT,
        system="enhanced",
        kind=kind,
        payload=payload,
    )


def _settings(
    tmp_path: Path,
    *,
    enabled: bool,
    trace_root: Path | None = None,
) -> Settings:
    return Settings(
        lab_enabled=enabled,
        lab_composio_user_id="opaque-fixture-user" if enabled else None,
        lab_trace_root=trace_root or tmp_path / ".lab" / "traces",
        lab_revision="revision-fixture-07",
    )


def _client(settings: Settings) -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def _write_trace(
    root: Path,
    *,
    run_id: UUID = RUN_A,
    turn_id: UUID = TURN_A,
    candidate_name: str = "Stored Candidate A",
) -> tuple[TraceEvent, ...]:
    events = (
        _event(
            run_id,
            turn_id,
            1,
            TraceEventKind.RUN_METADATA,
            {"revision": "stored-revision", "mode": "fixture", "roster_count": 17},
        ),
        _event(
            run_id,
            turn_id,
            2,
            TraceEventKind.CANDIDATES,
            {
                "candidate_count": 1,
                "candidates": [
                    {
                        "agent_id": "00000000-0000-4000-8000-000000000001",
                        "name": candidate_name,
                    }
                ],
            },
        ),
        _event(
            run_id,
            turn_id,
            3,
            TraceEventKind.FINAL_RESPONSE,
            {"response": "Stored response", "api_key": "sk-private-route-secret"},
        ),
    )
    store = JsonlTraceStore(root)
    for event in events:
        store.emit(event)
    return store.read(run_id)


def test_lab_off_all_routes_are_404_and_do_not_create_trace_directory(tmp_path) -> None:
    root = tmp_path / ".lab" / "traces"
    client = _client(_settings(tmp_path, enabled=False, trace_root=root))

    responses = (
        client.get("/api/v1/lab/directory"),
        client.get(f"/api/v1/lab/runs/{RUN_A}/status"),
        client.get(f"/api/v1/lab/runs/{RUN_A}/trace"),
        client.delete(f"/api/v1/lab/runs/{RUN_A}/trace"),
    )

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
    assert not root.exists()


def test_directory_returns_stable_revision_and_direct_persisted_summaries(
    tmp_path,
    monkeypatch,
) -> None:
    roster_path = tmp_path / "roster.json"
    directory = AgentDirectory(
        roster_path,
        clock=lambda: datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
        id_factory=lambda: UUID("00000000-0000-4000-8000-000000000007"),
    )
    directory.create(
        name="Direct Directory Name",
        purpose="Bearer private-directory-token",
        aliases=("mailbox@example.invalid",),
        memory_summary="sk-private-directory-secret",
    )
    monkeypatch.setattr("server.routes.lab.get_agent_directory", lambda: directory)
    client = _client(_settings(tmp_path, enabled=True))

    first = client.get("/api/v1/lab/directory")
    second = client.get("/api/v1/lab/directory")

    assert first.status_code == 200
    assert first.content == second.content
    assert first.json() == {
        "schema_version": 1,
        "revision": {
            "availability": "available",
            "value": "revision-fixture-07",
            "reason": None,
        },
        "agent_count": 1,
        "agents": [
            {
                "agent_id": "00000000-0000-4000-8000-000000000007",
                "name": "Direct Directory Name",
                "purpose": "[REDACTED]",
                "aliases": ["[REDACTED_EMAIL]"],
                "status": "hot",
                "use_count": 0,
            }
        ],
    }


def test_status_and_trace_return_only_direct_stored_facts_with_stable_bytes(tmp_path) -> None:
    root = tmp_path / ".lab" / "traces"
    stored = _write_trace(root)
    client = _client(_settings(tmp_path, enabled=True, trace_root=root))

    status = client.get(f"/api/v1/lab/runs/{RUN_A}/status")
    first = client.get(f"/api/v1/lab/runs/{RUN_A}/trace")
    second = client.get(f"/api/v1/lab/runs/{RUN_A}/trace")

    assert status.status_code == 200
    assert status.json() == {
        "run_id": str(RUN_A),
        "event_count": 3,
        "last_sequence": 3,
        "complete": True,
    }
    expected = TraceResponse(events=stored, result=consolidate_trace(stored))
    expected_bytes = json.dumps(
        expected.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert first.status_code == 200
    assert first.content == expected_bytes
    assert second.content == expected_bytes
    assert first.json()["events"][1]["payload"]["candidate_count"] == 1
    assert first.json()["events"][1]["payload"]["candidates"][0]["name"] == "Stored Candidate A"
    assert b"sk-private-route-secret" not in first.content
    assert b"mailbox" not in first.content.lower()


def test_delete_resets_exactly_one_stored_run_and_missing_runs_are_404(tmp_path) -> None:
    root = tmp_path / ".lab" / "traces"
    _write_trace(root, run_id=RUN_A, turn_id=TURN_A)
    _write_trace(root, run_id=RUN_B, turn_id=TURN_B, candidate_name="Stored Candidate B")
    client = _client(_settings(tmp_path, enabled=True, trace_root=root))

    deleted = client.delete(f"/api/v1/lab/runs/{RUN_A}/trace")

    assert deleted.status_code == 200
    assert deleted.json() == {"run_id": str(RUN_A), "deleted": True}
    assert client.get(f"/api/v1/lab/runs/{RUN_A}/trace").status_code == 404
    assert client.get(f"/api/v1/lab/runs/{RUN_B}/trace").status_code == 200
    assert client.delete(f"/api/v1/lab/runs/{RUN_A}/trace").status_code == 404


def test_missing_traversal_and_symlink_requests_fail_closed_without_creation(
    tmp_path,
) -> None:
    missing_root = tmp_path / ".lab" / "missing-traces"
    missing_client = _client(
        _settings(tmp_path, enabled=True, trace_root=missing_root)
    )

    assert missing_client.get(f"/api/v1/lab/runs/{RUN_A}/trace").status_code == 404
    assert missing_client.get("/api/v1/lab/runs/not-a-uuid/trace").status_code == 404
    assert missing_client.get("/api/v1/lab/runs/%2e%2e/trace").status_code == 404
    assert not missing_root.exists()

    lab_root = tmp_path / ".lab"
    lab_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = lab_root / "linked-traces"
    linked_root.symlink_to(outside, target_is_directory=True)
    linked_client = _client(_settings(tmp_path, enabled=True, trace_root=linked_root))

    assert linked_client.get(f"/api/v1/lab/runs/{RUN_A}/trace").status_code == 400
    assert not list(outside.iterdir())

    run_root = lab_root / "run-file-traces"
    run_root.mkdir()
    outside_file = outside / "outside.jsonl"
    outside_file.write_bytes(b"outside-sentinel")
    run_link = JsonlTraceStore(run_root).path_for(RUN_A)
    run_link.symlink_to(outside_file)
    run_link_client = _client(_settings(tmp_path, enabled=True, trace_root=run_root))

    assert run_link_client.get(f"/api/v1/lab/runs/{RUN_A}/trace").status_code == 400
    assert run_link_client.delete(f"/api/v1/lab/runs/{RUN_A}/trace").status_code == 400
    assert outside_file.read_bytes() == b"outside-sentinel"


def test_concurrent_trace_requests_never_leak_events_between_runs(tmp_path) -> None:
    root = tmp_path / ".lab" / "traces"
    _write_trace(root, run_id=RUN_A, turn_id=TURN_A, candidate_name="ONLY-RUN-A")
    _write_trace(root, run_id=RUN_B, turn_id=TURN_B, candidate_name="ONLY-RUN-B")
    client = _client(_settings(tmp_path, enabled=True, trace_root=root))

    with ThreadPoolExecutor(max_workers=2) as pool:
        response_a, response_b = pool.map(
            client.get,
            (
                f"/api/v1/lab/runs/{RUN_A}/trace",
                f"/api/v1/lab/runs/{RUN_B}/trace",
            ),
        )

    assert b"ONLY-RUN-A" in response_a.content
    assert b"ONLY-RUN-B" not in response_a.content
    assert b"ONLY-RUN-B" in response_b.content
    assert b"ONLY-RUN-A" not in response_b.content
