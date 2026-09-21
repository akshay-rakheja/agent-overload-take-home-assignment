"""Guarded serialized Evaluation Lab paired-run routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import Settings, get_settings
from server.routes import api_router
from server.services.evaluation_lab.orchestrator import StartRunRequest


REQUEST_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
REQUEST_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _settings(tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        lab_enabled=enabled,
        lab_composio_user_id="opaque-fixture-user" if enabled else None,
        server_host="127.0.0.1",
        lab_trace_root=tmp_path / ".lab" / "traces",
        lab_run_root=tmp_path / ".lab" / "runs",
    )


def _client(settings: Settings, *, host: str = "127.0.0.1") -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, client=(host, 40_001))


def test_scenario_list_and_run_round_trip_are_stable_and_idempotent(tmp_path) -> None:
    client = _client(_settings(tmp_path))
    scenarios = client.get("/api/v1/lab/scenarios")

    assert scenarios.status_code == 200
    listed = scenarios.json()
    assert listed["schema_version"] == 1
    assert listed["scenario_count"] == 14
    assert all(item["repetitions"] == 3 for item in listed["scenarios"])
    assert {item["track"] for item in listed["scenarios"]} == {"controlled"}
    scenario_id = listed["scenarios"][0]["scenario_id"]
    body = {"request_id": REQUEST_A, "scenario_ids": [scenario_id]}

    first = client.post("/api/v1/lab/runs", json=body)
    duplicate = client.post("/api/v1/lab/runs", json=body)

    assert first.status_code == duplicate.status_code == 202
    assert duplicate.content == first.content
    run_id = first.json()["run_id"]
    read = client.get(f"/api/v1/lab/runs/{run_id}")
    assert read.status_code == 200
    assert read.json()["status"] == "queued"
    assert read.json()["request"]["request_id"] == REQUEST_A
    assert read.json()["schedule"][0]["scenario_id"] == scenario_id


def test_concurrent_different_run_is_refused_while_reads_remain_available(tmp_path) -> None:
    client = _client(_settings(tmp_path))
    scenario_id = client.get("/api/v1/lab/scenarios").json()["scenarios"][0]["scenario_id"]
    first = client.post(
        "/api/v1/lab/runs",
        json={"request_id": REQUEST_A, "scenario_ids": [scenario_id]},
    )
    run_id = first.json()["run_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        create_future = pool.submit(
            client.post,
            "/api/v1/lab/runs",
            json={"request_id": REQUEST_B, "scenario_ids": [scenario_id]},
        )
        read_future = pool.submit(client.get, f"/api/v1/lab/runs/{run_id}")
        refused = create_future.result()
        observed = read_future.result()

    assert refused.status_code == 409
    assert observed.status_code == 200
    assert observed.json()["run_id"] == run_id


def test_new_routes_keep_lab_and_loopback_guards_and_fail_closed(tmp_path) -> None:
    off = _client(_settings(tmp_path, enabled=False))
    assert off.get("/api/v1/lab/scenarios").status_code == 404
    assert off.post("/api/v1/lab/runs", json={}).status_code == 404
    assert off.get(f"/api/v1/lab/runs/{UUID(int=0)}").status_code == 404
    assert not (tmp_path / ".lab").exists()

    remote = _client(_settings(tmp_path), host="203.0.113.7")
    assert remote.get("/api/v1/lab/scenarios").status_code == 403
    assert remote.post("/api/v1/lab/runs", json={}).status_code == 403
    assert remote.get(f"/api/v1/lab/runs/{UUID(int=0)}").status_code == 403

    local = _client(_settings(tmp_path))
    assert local.get("/api/v1/lab/runs/not-a-uuid").status_code == 404
    assert local.get("/api/v1/lab/runs/%2e%2e").status_code == 404


def test_request_validation_rejects_unknown_scenario_without_persisting(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = _client(settings)

    response = client.post(
        "/api/v1/lab/runs",
        json={"request_id": REQUEST_A, "scenario_ids": ["not-predeclared"]},
    )

    assert response.status_code == 422
    assert not settings.lab_run_root.exists()


def test_run_creation_validation_errors_are_bounded_and_redacted(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = _client(settings)
    synthetic_token = "sk-fakeONLY_REVIEW_12345678"
    synthetic_email = "review-secret@example.invalid"

    response = client.post(
        "/api/v1/lab/runs",
        json={
            "scenario_ids": [synthetic_token, synthetic_email],
            "unexpected": synthetic_token,
        },
    )

    assert response.status_code == 422
    assert synthetic_token.encode() not in response.content
    assert synthetic_email.encode() not in response.content
    assert len(response.content) < 256


def test_run_creation_redacts_framework_body_shape_and_json_errors(tmp_path) -> None:
    client = _client(_settings(tmp_path))
    synthetic_token = "sk-fakeONLY_REVIEW_12345678"
    synthetic_email = "review-secret@example.invalid"
    responses = (
        client.post("/api/v1/lab/runs", json=[synthetic_token]),
        client.post("/api/v1/lab/runs", json=synthetic_email),
        client.post("/api/v1/lab/runs", json=42),
        client.post("/api/v1/lab/runs", json=None),
        client.post(
            "/api/v1/lab/runs",
            content=f'{{"secret":"{synthetic_token}"'.encode(),
            headers={"content-type": "application/json"},
        ),
    )

    for response in responses:
        assert response.status_code == 422
        assert synthetic_token.encode() not in response.content
        assert synthetic_email.encode() not in response.content
        assert len(response.content) < 256


def test_run_creation_rejects_oversized_body_without_echoing_it(tmp_path) -> None:
    client = _client(_settings(tmp_path))
    synthetic_token = "sk-fakeONLY_REVIEW_12345678"
    response = client.post(
        "/api/v1/lab/runs",
        content=(synthetic_token * 2_000).encode(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert synthetic_token.encode() not in response.content
    assert len(response.content) < 256


def test_start_request_rejects_duplicate_scenario_ids() -> None:
    scenario_id = "exact-instagram-security"
    try:
        StartRunRequest(
            request_id=UUID(REQUEST_A),
            scenario_ids=(scenario_id, scenario_id),
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate scenario ids must be rejected")
