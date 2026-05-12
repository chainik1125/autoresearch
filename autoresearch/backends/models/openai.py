"""OpenAIClient — v2 stub.

The ModelClient protocol exists for non-agent LLM calls (preflight, postflight,
error summary, summarize_run). v1 ships only the Anthropic implementation
because the workflows we have don't yet benefit from cross-provider validation.

This stub is the forcing function for the protocol: if the protocol can't be
implemented honestly for OpenAI's SDK (different streaming, tool-use shape,
token accounting), the protocol needs rework before v2.

Implementation outline (when needed):
  - Map `complete(system=, user=)` → `responses.create(input=...)`
  - Cost: read from `usage.total_tokens` + per-model rate table (mirror anthropic.py).
  - Return ModelResponse(text=..., input_tokens=..., output_tokens=..., cost_usd=...).
"""

from __future__ import annotations

from autoresearch.backends.models.base import ModelResponse


_NOT_IMPLEMENTED = (
    "OpenAIClient is a v1 stub. The ModelClient protocol exists for cross-provider "
    "validation calls; v1 only ships AnthropicClient because workflows don't yet "
    "compare providers. Implement when needed."
)


class OpenAIClient:
    model: str

    def __init__(self, *_args, model: str = "gpt-4o-mini", **_kwargs) -> None:
        self.model = model
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED)
