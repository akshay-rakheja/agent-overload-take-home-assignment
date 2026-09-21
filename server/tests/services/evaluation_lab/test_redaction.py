from __future__ import annotations

from server.services.evaluation_lab.redaction import REDACTED, redact_value


def test_recursive_redaction_removes_nested_credentials_mail_and_provider_data() -> None:
    source = {
        "authorization": "Bearer fake-secret-token",
        "api_key": "sk-test-super-secret",
        "auth_config_id": "ac_test_123456",
        "oauth_url": "https://provider.test/oauth?code=fake-code&state=private",
        "oauth": {"code": "nested-oauth-code", "granted": True},
        "headers": {
            "X-Api-Key": "nested-secret",
            "Content-Type": "application/json",
        },
        "contacts": [
            {
                "email": "alice@example.test",
                "message_id": "msg-private-123",
                "thread_id": "thread-private-456",
                "snippet": "Mailbox preview for Alice",
                "body": "Full private mailbox body",
            }
        ],
        "query_string": "from=alice@example.test&token=secret",
        "provider_error": RuntimeError(
            "request for alice@example.test failed with Bearer provider-secret"
        ),
    }

    redacted = redact_value(source)

    assert redacted == {
        "api_key": REDACTED,
        "auth_config_id": REDACTED,
        "authorization": REDACTED,
        "contacts": [
            {
                "body": REDACTED,
                "email": REDACTED,
                "message_id": REDACTED,
                "snippet": REDACTED,
                "thread_id": REDACTED,
            }
        ],
        "headers": {"Content-Type": REDACTED, "X-Api-Key": REDACTED},
        "oauth_url": "https://provider.test/oauth",
        "oauth": {"code": REDACTED, "granted": True},
        "provider_error": REDACTED,
        "query_string": REDACTED,
    }
    exported = repr(redacted)
    for marker in (
        "fake-secret-token",
        "sk-test-super-secret",
        "ac_test_123456",
        "fake-code",
        "nested-oauth-code",
        "alice@example.test",
        "msg-private-123",
        "Mailbox preview",
        "Full private",
        "provider-secret",
    ):
        assert marker not in exported


def test_redaction_preserves_only_explicit_safe_observation_facts() -> None:
    source = {
        "candidate_count": 5,
        "accepted": True,
        "operation_name": "GMAIL_FETCH_EMAILS",
        "fixture_fact_ids": ["fixture_fact_001", "fixture_fact_002"],
        "request_sha256": "a" * 64,
        "nested": {"journal_hash": "b" * 64},
    }

    assert redact_value(source) == source


def test_redaction_is_deterministic_and_does_not_mutate_input() -> None:
    source = {"z": [{"email_address": "person@example.test"}], "a": 2}

    first = redact_value(source)
    second = redact_value(source)

    assert first == second == {"a": 2, "z": [{"email_address": REDACTED}]}
    assert list(first) == ["a", "z"]
    assert source["z"][0]["email_address"] == "person@example.test"


def test_redaction_sanitizes_secret_patterns_even_under_unknown_keys() -> None:
    value = {
        "notes": [
            "Authorization: Bearer opaque-secret",
            "contact alice@example.test for details",
            "https://provider.test/callback?code=oauth-secret&state=private",
        ]
    }

    redacted = redact_value(value)

    assert redacted == {
        "notes": [REDACTED, "contact [REDACTED_EMAIL] for details", "https://provider.test/callback"]
    }
