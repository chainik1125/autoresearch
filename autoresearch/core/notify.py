"""Notification dispatch — fire-and-forget pings when a Run finishes.

One generic interface (`send_notification`) that adapts to several
webhook-shaped notification providers. Adding a new provider is one
function plus a registry entry.

### Usage

```python
from autoresearch.core import notify

notify.send_notification(
    url="https://ntfy.sh/dmitry-autoresearch",
    title="autoresearch /transfer 4f1e77ae57ef",
    message="COMPLETED • $0.31 spent • results at github.com/.../tree/autoresearch/results-4f1e77ae57ef",
    # provider auto-detected from URL; override with provider="..." if needed
)
```

### Providers

  - **ntfy** (default): plain POST with the message as body, optional
    `Title` header. https://ntfy.sh
  - **slack**: `{"text": "..."}` JSON to an incoming-webhook URL.
  - **discord**: `{"content": "..."}` JSON to a webhook URL.
  - **generic_post**: same wire format as ntfy but no header (fallback).

### How the postflight pod uses this

After the postflight agent finishes writing + pushing the summary, the
workflow reads `Settings.notification_url` and calls `send_notification`
once with the run's headline. Best-effort: failure is logged as a
finding but never re-raises (we don't want the chain to fail because
the user's webhook is down).
"""

from __future__ import annotations

import logging
from typing import Callable

import httpx

_log = logging.getLogger("autoresearch.notify")


def _send_ntfy(url: str, title: str | None, message: str) -> None:
    headers = {"Title": title} if title else {}
    httpx.post(url, content=message.encode("utf-8"), headers=headers, timeout=10.0)


def _send_slack(url: str, title: str | None, message: str) -> None:
    body = f"*{title}*\n{message}" if title else message
    httpx.post(url, json={"text": body}, timeout=10.0)


def _send_discord(url: str, title: str | None, message: str) -> None:
    body = f"**{title}**\n{message}" if title else message
    httpx.post(url, json={"content": body}, timeout=10.0)


def _send_generic_post(url: str, title: str | None, message: str) -> None:
    body = f"{title}\n{message}" if title else message
    httpx.post(url, content=body.encode("utf-8"), timeout=10.0)


# Provider → adapter function. Each adapter signature is (url, title, message)
# and raises on HTTP failure (we catch at the call site).
_PROVIDERS: dict[str, Callable[[str, str | None, str], None]] = {
    "ntfy": _send_ntfy,
    "slack": _send_slack,
    "discord": _send_discord,
    "generic_post": _send_generic_post,
}


def _infer_provider(url: str) -> str:
    """Best-effort host-pattern detection. Override with `provider=...`."""
    if "ntfy.sh" in url or "/ntfy/" in url:
        return "ntfy"
    if "hooks.slack.com" in url:
        return "slack"
    if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url:
        return "discord"
    return "generic_post"


def send_notification(
    url: str,
    title: str | None,
    message: str,
    *,
    provider: str | None = None,
) -> bool:
    """Fire one notification. Returns True on success, False on any failure.

    Failure is never raised — notification is opportunistic. Callers can
    inspect the return value or rely on a finding written next to the call
    site to record what happened.
    """
    chosen = provider or _infer_provider(url)
    fn = _PROVIDERS.get(chosen)
    if fn is None:
        _log.warning("unknown notification provider %r; skipping", chosen)
        return False
    try:
        fn(url, title, message)
        return True
    except Exception as exc:  # noqa: BLE001 -- notifications are best-effort
        _log.warning("notification via %s failed: %s", chosen, exc)
        return False
