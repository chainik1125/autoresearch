"""Tests for the notification dispatch layer.

Doesn't hit the network — mocks httpx.post to capture invocations and
verify each provider's wire shape.
"""

from __future__ import annotations

from unittest.mock import patch

from autoresearch.core import notify


def test_infer_provider_from_url() -> None:
    assert notify._infer_provider("https://ntfy.sh/topic") == "ntfy"
    assert notify._infer_provider("https://hooks.slack.com/services/T/X/Y") == "slack"
    assert notify._infer_provider("https://discord.com/api/webhooks/123/abc") == "discord"
    assert notify._infer_provider("https://discordapp.com/api/webhooks/...") == "discord"
    assert notify._infer_provider("https://my-bespoke.example.com/hook") == "generic_post"


def test_ntfy_wire_shape() -> None:
    with patch("autoresearch.core.notify.httpx.post") as p:
        ok = notify.send_notification(
            "https://ntfy.sh/dmitry-autoresearch",
            "test-title", "body text",
        )
    assert ok
    args, kwargs = p.call_args
    assert args[0] == "https://ntfy.sh/dmitry-autoresearch"
    assert kwargs.get("content") == b"body text"
    assert kwargs.get("headers") == {"Title": "test-title"}


def test_slack_wire_shape() -> None:
    with patch("autoresearch.core.notify.httpx.post") as p:
        ok = notify.send_notification(
            "https://hooks.slack.com/services/T/X/Y",
            "test-title", "body text",
        )
    assert ok
    _args, kwargs = p.call_args
    payload = kwargs["json"]
    assert "test-title" in payload["text"]
    assert "body text" in payload["text"]


def test_discord_wire_shape() -> None:
    with patch("autoresearch.core.notify.httpx.post") as p:
        notify.send_notification(
            "https://discord.com/api/webhooks/123/abc",
            "test-title", "body text",
        )
    _args, kwargs = p.call_args
    assert "test-title" in kwargs["json"]["content"]


def test_explicit_provider_overrides_inference() -> None:
    with patch("autoresearch.core.notify.httpx.post") as p:
        notify.send_notification(
            "https://example.com/wat",     # would infer generic_post
            None, "msg",
            provider="ntfy",
        )
    _args, kwargs = p.call_args
    # ntfy adapter sends bytes via content=
    assert kwargs.get("content") == b"msg"


def test_unknown_provider_returns_false_no_raise() -> None:
    with patch("autoresearch.core.notify.httpx.post") as p:
        ok = notify.send_notification(
            "https://example.com",
            None, "msg",
            provider="nonexistent",
        )
    assert ok is False
    p.assert_not_called()


def test_http_failure_returns_false_no_raise() -> None:
    import httpx
    with patch(
        "autoresearch.core.notify.httpx.post",
        side_effect=httpx.ConnectError("nope"),
    ):
        ok = notify.send_notification("https://ntfy.sh/t", None, "msg")
    # Failure is swallowed; caller sees False.
    assert ok is False
