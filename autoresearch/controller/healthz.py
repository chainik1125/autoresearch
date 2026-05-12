"""Tiny health + version endpoints for Railway probes."""

from __future__ import annotations

from fastapi import APIRouter

from autoresearch import __version__

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__}
