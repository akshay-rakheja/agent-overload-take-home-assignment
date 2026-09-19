"""Tests for deterministic bounded execution context rendering."""

from __future__ import annotations

from evals.fixtures import make_history
from server.config import Settings
from server.services.execution.context_policy import ExecutionContextPolicy


def test_default_context_configuration_has_two_independent_hard_limits() -> None:
    settings = Settings()

    assert settings.execution_context_max_recent_episodes == 8
    assert settings.execution_context_max_characters == 12_000


def test_empty_history_and_missing_summary_render_empty_context() -> None:
    result = ExecutionContextPolicy(max_recent_episodes=4, max_characters=500).render([])

    assert result.text == ""
    assert result.metrics.rendered_characters == 0
    assert result.metrics.included_episode_count == 0
    assert result.metrics.omitted_entry_count == 0
    assert result.metrics.summary_used is False


def test_small_history_is_preserved_when_it_fits_both_budgets() -> None:
    entries = make_history(8)
    result = ExecutionContextPolicy(max_recent_episodes=10, max_characters=10_000).render(entries)

    assert result.metrics.omitted_entry_count == 0
    assert result.metrics.included_episode_count == 2
    assert "<agent_request" in result.text
    assert "Synthetic history entry 00007" in result.text
    assert "older history omitted" not in result.text


def test_context_remains_bounded_as_history_grows() -> None:
    policy = ExecutionContextPolicy(max_recent_episodes=3, max_characters=1_200)

    for size in (10, 100, 1_000, 10_000):
        result = policy.render(make_history(size), memory_summary="Durable relationship summary.")
        assert result.metrics.rendered_characters <= 1_200
        assert len(result.text) == result.metrics.rendered_characters
        assert result.metrics.included_episode_count <= 3
        assert result.metrics.summary_used is True
        if size >= 100:
            assert result.metrics.omitted_entry_count > 0
            assert "older history omitted" in result.text


def test_single_oversized_entry_gets_explicit_bounded_representation() -> None:
    entries = [("agent_request", "2026-09-19 12:00:00", "é" * 5_000)]

    result = ExecutionContextPolicy(max_recent_episodes=4, max_characters=500).render(entries)

    assert len(result.text) <= 500
    assert "truncated" in result.text
    assert result.metrics.truncated_entry_count == 1


def test_unicode_multiline_payload_is_escaped_and_preserved() -> None:
    entries = [
        ("agent_request", "2026-09-19 12:00:00", "Résumé for Zoë\n<draft>"),
        ("agent_response", "2026-09-19 12:00:01", "Ready — awaiting approval"),
    ]

    result = ExecutionContextPolicy(max_recent_episodes=2, max_characters=2_000).render(entries)

    assert "Résumé for Zoë" in result.text
    assert "&lt;draft&gt;" in result.text
    assert "awaiting approval" in result.text


def test_missing_or_malformed_summary_falls_back_to_recent_episodes() -> None:
    entries = make_history(6)
    policy = ExecutionContextPolicy(max_recent_episodes=2, max_characters=1_000)

    missing = policy.render(entries, memory_summary=None)
    malformed = policy.render(entries, memory_summary="bad\x00summary")

    assert missing.metrics.summary_used is False
    assert malformed.metrics.summary_used is False
    assert "Synthetic history entry 00005" in missing.text
    assert "Synthetic history entry 00005" in malformed.text


def test_recent_approval_sensitive_episode_is_retained() -> None:
    entries = make_history(40) + [
        ("agent_request", "2026-09-19 12:00:00", "Draft the payment approval email"),
        ("agent_action", "2026-09-19 12:00:01", "Prepared draft only"),
        ("agent_response", "2026-09-19 12:00:02", "Waiting for user approval before sending"),
    ]

    result = ExecutionContextPolicy(max_recent_episodes=2, max_characters=1_500).render(entries)

    assert "Draft the payment approval email" in result.text
    assert "Waiting for user approval before sending" in result.text
    assert result.metrics.omitted_entry_count > 0
