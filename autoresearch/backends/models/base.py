"""ModelClient — provider-agnostic interface for single-turn LLM completions.

In v1 this is used only by `core/validation.py` for the small bounded LLM calls
(preflight, postflight, error summarization). It is NOT used for an agent loop —
v1 doesn't have one.

The protocol is deliberately small: single-turn, no streaming, no tool use. When
the package grows to need those, this interface will be extended (or the agent
loop will use the provider SDK directly).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class ModelResponse(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ModelClient(Protocol):
    model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> ModelResponse: ...
