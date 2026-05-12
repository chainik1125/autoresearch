"""REPLICATE — v2 stub.

Use case: "Claude, replicate the FRA paper on Qwen-32B." The agent fetches the
paper + repo, fights deps, runs the experiment, compares to claimed results.

This is the workflow we *originally* planned for v1 but deferred because it
needs a Claude Agent SDK loop (free-form shell-driving agent) which is the
single load-bearing unknown of the architecture. v1 ships TRANSFER first
(deterministic, no SDK loop) so the rest of the system is proven; REPLICATE
gets built once we know the FSM + heartbeat + checkpoint primitives hold up
under pressure.

Implementation outline (when this lands):
  - Pipeline.run() body is replaced by a Claude Agent SDK loop running on the pod.
  - The loop's tool set: shell, file ops, git clone, web fetch, the standard
    `share_finding` / `evaluate` MCP tools from the W2S pattern.
  - Checkpoint extends: agent message history serialized as part of partial_result.
  - Spike to validate SDK checkpoint-resume before committing — see the v1 plan's
    "Critical risks #1".
"""

from __future__ import annotations


def replicate(*_args, **_kwargs):
    raise NotImplementedError(
        "REPLICATE is a v2 workflow. It needs a Claude Agent SDK loop on the pod "
        "for free-form shell-driving; that primitive isn't shipped in v1. "
        "See workflows/replicate.py for the implementation outline."
    )
