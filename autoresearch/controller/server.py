"""FastAPI controller — the persistent process on Railway.

Two surfaces:
  - `/healthz`, `/version`  — Railway probes
  - `/mcp`                   — MCP Streamable HTTP for local Claude Code

Background:
  - Supervisor task that polls heartbeats and restarts stale pods. Started in
    the FastAPI lifespan; cleanly stopped on shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from urllib.parse import urlparse

from fastapi import FastAPI

from autoresearch.config import Settings, build_compute, build_model_client, build_storage
from autoresearch.controller import healthz
from autoresearch.controller.mcp_surface import build_mcp
from autoresearch.controller.supervisor import Supervisor


_log = logging.getLogger("autoresearch.controller")


def create_app() -> FastAPI:
    settings = Settings.load()
    storage = build_storage(settings)
    compute = build_compute(settings)
    model_client = build_model_client(settings)

    supervisor = (
        Supervisor(settings=settings, storage=storage, compute=compute)
        if compute is not None
        else None
    )
    # streamable_http_path="" so the mounted route ends up at /mcp/ (no /mcp/mcp/
    # redirect quirk for POST clients that don't follow 307 on non-GET methods).
    mcp = build_mcp(settings=settings, storage=storage, compute=compute, model_client=model_client)
    mcp.settings.streamable_http_path = "/"

    # MCP defaults to DNS-rebinding protection that whitelists only localhost.
    # For our public-internet deployment, add the controller's hostname so
    # requests from the wild aren't rejected with "Invalid Host header".
    public_url = settings.controller_public_url or settings.controller_url
    if public_url:
        host = urlparse(public_url).hostname
        if host:
            mcp.settings.transport_security.allowed_hosts.append(host)
            mcp.settings.transport_security.allowed_hosts.append(f"{host}:*")
            scheme = urlparse(public_url).scheme or "https"
            mcp.settings.transport_security.allowed_origins.append(f"{scheme}://{host}")
            mcp.settings.transport_security.allowed_origins.append(f"{scheme}://{host}:*")

    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _log.info(
            "controller starting: storage=%s compute=%s validators=%s",
            settings.storage, settings.compute, model_client is not None,
        )
        # The MCP streamable app has its own lifespan that initializes its session
        # task group. Starlette's Mount does NOT auto-propagate lifespans, so we
        # have to enter it here ourselves alongside our own startup.
        async with mcp_app.router.lifespan_context(mcp_app):
            if supervisor is not None:
                supervisor.start()
                _log.info("supervisor started (poll=%ss)", settings.supervisor_poll_seconds)
            else:
                _log.info("supervisor disabled (no compute backend)")
            try:
                yield
            finally:
                if supervisor is not None:
                    await supervisor.stop()
                    _log.info("supervisor stopped")

    app = FastAPI(title="autoresearch controller", lifespan=lifespan)
    app.include_router(healthz.router)
    app.mount("/mcp", mcp_app)
    return app


def run() -> None:
    """Entrypoint used by `autoresearch serve`."""
    import os

    import uvicorn

    settings = Settings.load()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Railway/Fly/Heroku all set $PORT dynamically; honor that over our setting.
    port = int(os.environ.get("PORT") or settings.controller_port)
    uvicorn.run(
        "autoresearch.controller.server:create_app",
        factory=True,
        host=settings.controller_host,
        port=port,
        log_level="info",
        # Behind a TLS-terminating proxy (Railway, Fly, etc.) the Host header
        # is rewritten and X-Forwarded-* carries the real client info. Without
        # these, uvicorn rejects requests with "421 Invalid Host header".
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
