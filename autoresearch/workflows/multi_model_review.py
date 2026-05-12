"""MULTI_MODEL_REVIEW — v2 stub.

Use case: "Here's my abstract / experiment design. Spin up a debate protocol
between three models (Claude, GPT, Gemini) to refine it."

This workflow is the right fit for Modal's ephemeral compute: each model call
is bounded, parallel, and short-lived. Fan out N model calls, collect into a
consolidated review, iterate.

Why deferred from v1: TRANSFER is sequential and uses RunPod sessions exclusively;
MULTI_MODEL_REVIEW would force Modal integration (the ephemeral compute side of
the ComputeBackend protocol), which we're keeping as a v1 stub.

Implementation outline (when this lands):
  - Tools: `model_client.complete()` against multiple providers in parallel
    (Anthropic, OpenAI, Google via gemini SDK).
  - Cost: each model's complete() reports cost; budget aggregates across them.
  - Persistence: each model's response = one finding (type=hypothesis).
    Consolidated review = one finding (type=insight).
  - No GPU needed — runs on Modal CPU function or local; doesn't need RunPod.
"""

from __future__ import annotations


def multi_model_review(*_args, **_kwargs):
    raise NotImplementedError(
        "MULTI_MODEL_REVIEW is a v2 workflow. Needs the OpenAI / Gemini ModelClient "
        "impls and (optionally) the Modal ephemeral compute path."
    )
