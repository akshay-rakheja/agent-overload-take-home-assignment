from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest

from server.config import ModelCallConfig, ModelRole
from server.openrouter_client.client import OpenRouterError, request_chat_completion
from server.services.evaluation_lab.budget import (
    BudgetExceeded,
    CostController,
    CostLedger,
    budget_scope,
)
from server.services.evaluation_lab.pricing import (
    CostSource,
    PriceSchedule,
    PricingSnapshot,
    PricingSnapshotStore,
    calculate_cost,
    conservative_reservation,
)
from server.services.evaluation_lab.usage import normalize_usage


NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def _pricing(*, alternatives: tuple[PriceSchedule, ...] = ()) -> PricingSnapshot:
    return PricingSnapshot(
        model_id="openai/gpt-4.1-mini",
        prompt_price_per_token=Decimal("0.0000004"),
        completion_price_per_token=Decimal("0.0000016"),
        cached_prompt_price_per_token=Decimal("0.0000001"),
        context_limit=1_047_576,
        source_url="https://openrouter.ai/api/v1/models",
        retrieved_at=NOW,
        response_sha256="a" * 64,
        alternative_prices=alternatives,
    )


def test_pricing_snapshot_sanitizes_metadata_and_persists_only_bounded_fields(
    tmp_path: Path,
) -> None:
    raw = {
        "data": [
            {
                "id": "openai/gpt-4.1-mini",
                "context_length": 1_047_576,
                "pricing": {
                    "prompt": "0.0000004",
                    "completion": "0.0000016",
                    "input_cache_read": "0.0000001",
                },
                "description": "Bearer metadata-secret-that-must-not-persist",
            }
        ],
        "api_key": "metadata-key-that-must-not-persist",
    }
    snapshot = PricingSnapshot.from_model_metadata(
        raw,
        model_id="openai/gpt-4.1-mini",
        source_url=(
            "https://metadata-user:metadata-password@openrouter.ai/api/v1/models"
            "?api_key=metadata-query-secret#fragment"
        ),
        retrieved_at=NOW,
    )

    assert snapshot.prompt_price_per_token == Decimal("0.0000004")
    assert snapshot.completion_price_per_token == Decimal("0.0000016")
    assert snapshot.cached_prompt_price_per_token == Decimal("0.0000001")
    assert snapshot.context_limit == 1_047_576
    assert snapshot.source_url == "https://openrouter.ai/api/v1/models"
    assert len(snapshot.response_sha256) == 64

    store = PricingSnapshotStore(tmp_path / ".lab" / "pricing")
    stored_path = store.write(snapshot)
    restored = store.read(snapshot.model_id)
    raw_disk = stored_path.read_text(encoding="utf-8")

    assert restored == snapshot
    assert "metadata-secret" not in raw_disk
    assert "metadata-key" not in raw_disk
    assert "metadata-password" not in raw_disk
    assert "metadata-query-secret" not in raw_disk


def test_calculate_cost_prefers_exact_provider_reported_charge() -> None:
    usage = normalize_usage(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "cached_tokens": 3,
                "cost": 0.123456789,
            }
        }
    )

    record = calculate_cost(usage, _pricing())

    assert record.source is CostSource.PROVIDER_REPORTED
    assert record.amount_usd == Decimal("0.123456789")


def test_calculate_cost_uses_decimal_prompt_completion_and_cached_prices() -> None:
    usage = normalize_usage(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "cached_tokens": 3,
            }
        }
    )

    record = calculate_cost(usage, _pricing())

    assert record.source is CostSource.CONSERVATIVE_ESTIMATE
    assert record.amount_usd == Decimal("0.0000095")


@pytest.mark.parametrize(
    "missing",
    ["prompt_price_per_token", "completion_price_per_token"],
)
def test_calculate_cost_marks_missing_required_price_unavailable(missing: str) -> None:
    values = _pricing().model_dump()
    values[missing] = None
    snapshot = PricingSnapshot(**values)
    usage = normalize_usage(
        {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    )

    record = calculate_cost(usage, snapshot)

    assert record.source is CostSource.UNAVAILABLE
    assert record.amount_usd is None


def test_conservative_reservation_uses_characters_max_tokens_and_highest_prices() -> None:
    expensive = PriceSchedule(
        provider="ambiguous-provider",
        prompt_price_per_token=Decimal("0.000002"),
        completion_price_per_token=Decimal("0.000004"),
        cached_prompt_price_per_token=Decimal("0.000003"),
    )

    estimate = conservative_reservation(
        serialized_prompt="five!",
        max_tokens=7,
        pricing=_pricing(alternatives=(expensive,)),
    )

    assert estimate.prompt_token_upper_bound == 5
    assert estimate.completion_token_upper_bound == 7
    assert estimate.amount_usd == Decimal("0.000043")
    assert estimate.source is CostSource.CONSERVATIVE_ESTIMATE


def test_ledger_refuses_only_when_next_reservation_would_cross_ten_dollars(
    tmp_path: Path,
) -> None:
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    allowed = ledger.reserve(Decimal("10.00"), call_id=uuid4())

    assert ledger.snapshot().reserved_usd == Decimal("10.00")
    with pytest.raises(BudgetExceeded):
        ledger.reserve(Decimal("0.000000001"), call_id=uuid4())

    ledger.abandon(allowed, reason="not submitted")
    first = ledger.reserve(Decimal("9.99"), call_id=uuid4())
    ledger.reconcile(
        first,
        Decimal("9.99"),
        source=CostSource.PROVIDER_REPORTED,
        outcome="success",
    )
    with pytest.raises(BudgetExceeded):
        ledger.reserve(Decimal("0.02"), call_id=uuid4())


def test_reconcile_is_idempotent_and_never_double_charges(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    reservation = ledger.reserve(Decimal("0.10"), call_id=uuid4())

    first = ledger.reconcile(
        reservation,
        Decimal("0.031"),
        source=CostSource.PROVIDER_REPORTED,
        outcome="success",
    )
    second = ledger.reconcile(
        reservation,
        Decimal("0.031"),
        source=CostSource.PROVIDER_REPORTED,
        outcome="success",
    )

    assert first.spent_usd == Decimal("0.031")
    assert second.spent_usd == Decimal("0.031")
    assert second.reserved_usd == Decimal("0")
    with pytest.raises(ValueError, match="different"):
        ledger.reconcile(
            reservation,
            Decimal("0.032"),
            source=CostSource.PROVIDER_REPORTED,
            outcome="success",
        )


def test_failed_call_reported_charge_is_retained(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    reservation = ledger.reserve(Decimal("0.10"), call_id=uuid4())

    snapshot = ledger.reconcile(
        reservation,
        Decimal("0.007"),
        source=CostSource.PROVIDER_REPORTED,
        outcome="failed",
    )

    assert snapshot.spent_usd == Decimal("0.007")
    entry = next(item for item in snapshot.entries if item.call_id == reservation.call_id)
    assert entry.outcome == "failed"
    assert entry.source is CostSource.PROVIDER_REPORTED


def test_abandon_releases_unsubmitted_reservation_without_spending(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    reservation = ledger.reserve(Decimal("0.10"), call_id=uuid4())

    snapshot = ledger.abandon(reservation, reason="never submitted")

    assert snapshot.spent_usd == Decimal("0")
    assert snapshot.reserved_usd == Decimal("0")
    assert snapshot.entries[0].reason == "never submitted"


def test_crash_recovery_conservatively_moves_active_reservations_to_spent(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".lab" / "runs" / "cost-ledger"
    first_process = CostLedger(root)
    reservation = first_process.reserve(Decimal("0.25"), call_id=uuid4())

    restarted = CostLedger(root)
    before = restarted.snapshot()
    recovered = restarted.recover_crashed_reservations(reason="controller restart")

    assert before.reserved_usd == Decimal("0.25")
    assert recovered.reserved_usd == Decimal("0")
    assert recovered.spent_usd == Decimal("0.25")
    entry = next(item for item in recovered.entries if item.call_id == reservation.call_id)
    assert entry.source is CostSource.CRASH_RECOVERY_ESTIMATE
    assert entry.outcome == "crash_recovered"


def test_concurrent_reservations_are_serialized_across_ledger_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".lab" / "runs" / "cost-ledger"
    call_ids = [uuid4() for _ in range(40)]

    def reserve(call_id: UUID) -> None:
        CostLedger(root).reserve(Decimal("0.25"), call_id=call_id)

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(reserve, call_id) for call_id in call_ids]

    succeeded = 0
    rejected = 0
    for future in futures:
        try:
            future.result()
            succeeded += 1
        except BudgetExceeded:
            rejected += 1

    snapshot = CostLedger(root).snapshot()
    assert succeeded == 40
    assert rejected == 0
    assert snapshot.reserved_usd == Decimal("10.00")
    assert len(snapshot.entries) == 40


def test_concurrent_reservations_cannot_race_past_cap(tmp_path: Path) -> None:
    root = tmp_path / ".lab" / "runs" / "cost-ledger"

    def reserve(call_id: UUID) -> bool:
        try:
            CostLedger(root).reserve(Decimal("0.30"), call_id=call_id)
        except BudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        accepted = list(pool.map(reserve, [uuid4() for _ in range(40)]))

    snapshot = CostLedger(root).snapshot()
    assert accepted.count(True) == 33
    assert accepted.count(False) == 7
    assert snapshot.reserved_usd == Decimal("9.90")
    assert snapshot.spent_usd + snapshot.reserved_usd <= Decimal("10.00")


def test_failed_atomic_replace_leaves_last_committed_ledger_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.services.evaluation_lab import budget as budget_module

    root = tmp_path / ".lab" / "runs" / "cost-ledger"
    ledger = CostLedger(root)
    first = ledger.reserve(Decimal("0.20"), call_id=uuid4())
    real_replace = budget_module.os.replace

    def fail_replace(*_args, **_kwargs):
        raise OSError("fixture atomic replace failure")

    monkeypatch.setattr(budget_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="fixture atomic replace failure"):
        ledger.reserve(Decimal("0.30"), call_id=uuid4())
    monkeypatch.setattr(budget_module.os, "replace", real_replace)

    snapshot = CostLedger(root).snapshot()
    assert snapshot.reserved_usd == Decimal("0.20")
    assert [entry.call_id for entry in snapshot.entries] == [first.call_id]
    assert not list(root.glob("*.tmp"))


def test_ledger_rejects_symlinked_root_and_state_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_root = tmp_path / ".lab" / "linked"
    symlink_root.parent.mkdir()
    symlink_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        CostLedger(symlink_root).snapshot()

    root = tmp_path / ".lab" / "safe-ledger"
    ledger = CostLedger(root)
    ledger.snapshot()
    state_path = root / "cost-ledger.json"
    state_path.symlink_to(outside / "stolen.json")
    with pytest.raises(ValueError, match="regular file"):
        ledger.reserve(Decimal("0.01"), call_id=uuid4())


class _FakeAsyncClient:
    response: httpx.Response
    calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url, *, headers, json, timeout):
        self.__class__.calls += 1
        return self.__class__.response


def _response(payload: dict[str, object], status: int = 200) -> httpx.Response:
    request = httpx.Request("POST", "https://router.test/chat/completions")
    return httpx.Response(status, request=request, content=json.dumps(payload).encode())


def test_client_reserves_before_http_and_reconciles_provider_charge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.openrouter_client import client as client_module

    _FakeAsyncClient.calls = 0
    _FakeAsyncClient.response = _response(
        {
            "model": "openai/gpt-4.1-mini",
            "provider": "OpenAI",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "total_tokens": 22,
                "cost": 0.0000112,
            },
        }
    )
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    controller = CostController(ledger=ledger, pricing=_pricing())

    with budget_scope(controller):
        asyncio.run(
            request_chat_completion(
                config=ModelCallConfig(
                    model_id="openai/gpt-4.1-mini",
                    temperature=0,
                    top_p=1,
                    max_tokens=1000,
                    max_retries=0,
                ),
                role=ModelRole.INTERACTION,
                messages=[{"role": "user", "content": "fixture prompt"}],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    snapshot = ledger.snapshot()
    assert _FakeAsyncClient.calls == 1
    assert snapshot.reserved_usd == Decimal("0")
    assert snapshot.spent_usd == Decimal("0.0000112")
    assert snapshot.entries[0].source is CostSource.PROVIDER_REPORTED


def test_client_preserves_reported_charge_from_failed_http_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.openrouter_client import client as client_module

    _FakeAsyncClient.calls = 0
    _FakeAsyncClient.response = _response(
        {"error": {"code": "fixture"}, "usage": {"cost": 0.004}},
        status=429,
    )
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    controller = CostController(ledger=ledger, pricing=_pricing())

    with budget_scope(controller), pytest.raises(OpenRouterError):
        asyncio.run(
            request_chat_completion(
                config=ModelCallConfig(
                    model_id="openai/gpt-4.1-mini",
                    max_tokens=1000,
                    max_retries=0,
                ),
                role=ModelRole.INTERACTION,
                messages=[{"role": "user", "content": "fixture prompt"}],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    snapshot = ledger.snapshot()
    assert snapshot.spent_usd == Decimal("0.004")
    assert snapshot.entries[0].outcome == "failed"


def test_budget_stop_occurs_before_http_and_offline_snapshot_stays_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.openrouter_client import client as client_module

    _FakeAsyncClient.calls = 0
    _FakeAsyncClient.response = _response({"choices": []})
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    full = ledger.reserve(Decimal("10.00"), call_id=uuid4())
    ledger.reconcile(
        full,
        Decimal("10.00"),
        source=CostSource.PROVIDER_REPORTED,
        outcome="success",
    )
    controller = CostController(ledger=ledger, pricing=_pricing())

    with budget_scope(controller), pytest.raises(BudgetExceeded):
        asyncio.run(
            request_chat_completion(
                config=ModelCallConfig(
                    model_id="openai/gpt-4.1-mini", max_tokens=1000
                ),
                role=ModelRole.INTERACTION,
                messages=[{"role": "user", "content": "fixture prompt"}],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    assert _FakeAsyncClient.calls == 0
    assert ledger.snapshot().spent_usd == Decimal("10.00")


def test_client_rejects_model_without_matching_pricing_snapshot_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.openrouter_client import client as client_module

    _FakeAsyncClient.calls = 0
    _FakeAsyncClient.response = _response({"choices": []})
    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    ledger = CostLedger(tmp_path / ".lab" / "runs" / "cost-ledger")
    controller = CostController(ledger=ledger, pricing=_pricing())

    with budget_scope(controller), pytest.raises(ValueError, match="pricing snapshot"):
        asyncio.run(
            request_chat_completion(
                config=ModelCallConfig(
                    model_id="google/gemini-2.5-flash", max_tokens=1000
                ),
                role=ModelRole.INTERACTION,
                messages=[{"role": "user", "content": "fixture prompt"}],
                api_key="fixture-key",
                base_url="https://router.test",
            )
        )

    assert _FakeAsyncClient.calls == 0
    assert ledger.snapshot().entries == ()
