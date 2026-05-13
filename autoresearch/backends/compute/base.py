"""ComputeBackend protocol.

A ComputeBackend dispatches a long-lived pod (RunPod) — eventually also ephemeral
serverless functions (Modal, v2). For v1 only the session shape is real.

`SessionSpec` is what the dispatcher hands the backend. `SessionHandle` is what
the backend returns — the durable identifier plus enough info to either monitor,
SSH into, or terminate the session.

Pods are *attached* to a network volume that outlives the pod. The supervisor's
restart path reuses the same volume on the new pod so HF caches and pipeline
on-disk state survive.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SessionSpec(BaseModel):
    # Either a single GPU type ("H100 80GB") OR a list of acceptable types in
    # preference order. The backend picks the first one with current inventory.
    # See `core/hardware.py` for the selection logic that produces the list.
    gpu: str | list[str]
    image: str                                 # Docker image
    network_volume_id: str                     # required: where the HF cache & user pipelines live
    env: dict[str, str] = Field(default_factory=dict)  # injected into the pod
    name: str | None = None                    # human-readable label
    container_disk_gb: int = 50                # ephemeral disk on top of the network volume
    expose_ssh: bool = True                    # open port 22 for takeover handoff
    extra_ports: list[str] = Field(default_factory=list)  # e.g. ["8000/http"]
    spot: bool = False                         # community/spot pods are cheaper but preemptible
    container_registry_auth_id: str | None = None  # backend-side saved registry credential


class SessionHandle(BaseModel):
    id: str
    status: Literal["queued", "running", "exited", "terminated", "failed", "unknown"]
    public_ip: str | None = None
    ssh_port: int | None = None
    raw: dict | None = None                    # backend-native payload for debugging

    def ssh_command(self, user: str = "root") -> str | None:
        if not (self.public_ip and self.ssh_port):
            return None
        return f"ssh {user}@{self.public_ip} -p {self.ssh_port}"


@runtime_checkable
class ComputeBackend(Protocol):
    name: str                                  # "runpod" / "modal" / ...

    def create_session(self, spec: SessionSpec) -> SessionHandle: ...
    def get_session(self, session_id: str) -> SessionHandle: ...
    def terminate_session(self, session_id: str) -> None: ...

    def list_gpu_offers(self, data_center_id: str | None = None) -> "list":
        """Return a list of `GpuOffer` for the backend's current catalog.

        Returns `core.hardware.GpuOffer` objects. If `data_center_id` is set,
        each offer's `available_in_dc` reflects stock specifically in that
        DC; otherwise `available_in_dc` mirrors global availability.

        Backends that don't have a pricing/inventory API can return an empty
        list — the dispatcher will fall back to the user's explicit `gpu`
        argument or `settings.default_gpu`.
        """
        ...
