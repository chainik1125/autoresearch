"""RunPodCompute — the v1 ComputeBackend.

Uses RunPod's REST API (https://rest.runpod.io/v1/). The pod is created attached
to a network volume; pod placement defaults to the volume's data center.

GPU type IDs follow RunPod's naming. A small alias table resolves common short
names ("H100 80GB" → "NVIDIA H100 80GB HBM3"); unrecognized strings are passed
through unchanged so users can name exact IDs if they prefer.
"""

from __future__ import annotations

from typing import Any

import httpx

from autoresearch.backends.compute.base import SessionHandle, SessionSpec
from autoresearch.core.hardware import GpuOffer


_API_BASE = "https://rest.runpod.io/v1"
_GRAPHQL_URL = "https://api.runpod.io/graphql"

# Friendly-name → exact RunPod gpuTypeId. Unknown short names pass through.
_GPU_ALIASES: dict[str, str] = {
    "H100": "NVIDIA H100 80GB HBM3",
    "H100 80GB": "NVIDIA H100 80GB HBM3",
    "H100 PCIe": "NVIDIA H100 PCIe",
    "A100": "NVIDIA A100-SXM4-80GB",
    "A100 80GB": "NVIDIA A100 80GB PCIe",
    "A100 SXM": "NVIDIA A100-SXM4-80GB",
    "A40": "NVIDIA A40",
    "A6000": "NVIDIA RTX A6000",
    "L40": "NVIDIA L40",
    "L40S": "NVIDIA L40S",
}


def _resolve_gpu(name: str) -> str:
    return _GPU_ALIASES.get(name, name)


def _interpret_status(state: str | None) -> str:
    """Map RunPod's desiredStatus / state strings onto our shorter set."""
    if not state:
        return "unknown"
    s = state.upper()
    if s in ("RUNNING", "READY"):
        return "running"
    if s in ("CREATED", "PROVISIONING", "STARTING", "RESTARTING"):
        return "queued"
    if s in ("EXITED", "STOPPED"):
        return "exited"
    if s in ("TERMINATED", "DELETED"):
        return "terminated"
    if s in ("FAILED", "ERROR"):
        return "failed"
    return "unknown"


class RunPodCompute:
    name = "runpod"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            base_url=_API_BASE,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def create_session(self, spec: SessionSpec) -> SessionHandle:
        ports: list[str] = list(spec.extra_ports)
        if spec.expose_ssh and not any(p.startswith("22/") for p in ports):
            ports.append("22/tcp")

        # RunPod's `gpuTypeIds` accepts a list and picks any one with stock.
        # `spec.gpu` may be a single name or a preference-ordered list (see
        # core/hardware.py for the producer).
        gpu_ids: list[str]
        if isinstance(spec.gpu, str):
            gpu_ids = [_resolve_gpu(spec.gpu)]
        else:
            gpu_ids = [_resolve_gpu(g) for g in spec.gpu]

        body: dict[str, Any] = {
            "name": spec.name or "autoresearch-pod",
            "imageName": spec.image,
            "gpuTypeIds": gpu_ids,
            "gpuCount": 1,
            "networkVolumeId": spec.network_volume_id,
            "containerDiskInGb": spec.container_disk_gb,
            "env": dict(spec.env),
            "ports": ports or None,                  # RunPod expects array, not comma-joined string
            "interruptible": spec.spot,
            "containerRegistryAuthId": spec.container_registry_auth_id,
        }
        body = {k: v for k, v in body.items() if v is not None}

        r = self._client.post("/pods", json=body)
        r.raise_for_status()
        return _handle_from_raw(r.json())

    def get_session(self, session_id: str) -> SessionHandle:
        r = self._client.get(f"/pods/{session_id}")
        r.raise_for_status()
        return _handle_from_raw(r.json())

    def terminate_session(self, session_id: str) -> None:
        r = self._client.delete(f"/pods/{session_id}")
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()

    def list_gpu_offers(self, data_center_id: str | None = None) -> list[GpuOffer]:
        """Query RunPod's GPU catalog via GraphQL. One network round-trip.

        Returns a `GpuOffer` per gpuType. `price_per_hour` comes from
        `lowestPrice.uninterruptablePrice` (None when no inventory). When
        `data_center_id` is set, the priceQuery is scoped to that DC; offers
        with `price_per_hour is None` are out of stock there.
        """
        if data_center_id is None:
            price_query = "lowestPrice(input:{gpuCount:1})"
        else:
            price_query = f'lowestPrice(input:{{gpuCount:1,dataCenterId:"{data_center_id}"}})'
        query = (
            "query{gpuTypes{id memoryInGb "
            + price_query
            + "{uninterruptablePrice}}}"
        )
        # Use a fresh httpx call against GraphQL — the REST client's base_url
        # is for /v1/.
        with httpx.Client(
            timeout=self._client.timeout,
            headers=dict(self._client.headers),
        ) as gq:
            r = gq.post(_GRAPHQL_URL, json={"query": query})
            r.raise_for_status()
            payload = r.json()
        gpus = payload.get("data", {}).get("gpuTypes") or []
        offers: list[GpuOffer] = []
        for g in gpus:
            price = (g.get("lowestPrice") or {}).get("uninterruptablePrice")
            offers.append(
                GpuOffer(
                    id=g["id"],
                    memory_gb=int(g.get("memoryInGb") or 0),
                    price_per_hour=float(price) if price is not None else None,
                    available_in_dc=price is not None,
                )
            )
        return offers

    def close(self) -> None:
        self._client.close()


def _handle_from_raw(raw: dict[str, Any]) -> SessionHandle:
    """Best-effort mapping of RunPod's payload into our SessionHandle."""
    state = raw.get("desiredStatus") or raw.get("status") or raw.get("currentStatus")
    public_ip: str | None = None
    ssh_port: int | None = None
    for port in raw.get("portMappings") or raw.get("ports") or []:
        if isinstance(port, dict) and (port.get("privatePort") == 22 or port.get("port") == 22):
            public_ip = public_ip or port.get("ip") or port.get("publicIp")
            ssh_port = ssh_port or port.get("publicPort") or port.get("hostPort")
    if not public_ip and raw.get("publicIp"):
        public_ip = raw["publicIp"]
    return SessionHandle(
        id=raw.get("id", ""),
        status=_interpret_status(state),
        public_ip=public_ip,
        ssh_port=ssh_port,
        raw=raw,
    )
