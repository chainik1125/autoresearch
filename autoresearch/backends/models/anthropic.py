"""AnthropicClient — Anthropic API impl of ModelClient.

Default model is Haiku 4.5 (fast/cheap, appropriate for the bounded validation
calls that v1 uses this for). Cost is best-effort: rates are hardcoded for
known models; unknown models return 0.0 cost.
"""

from __future__ import annotations

import anthropic

from autoresearch.backends.models.base import ModelResponse

# Per-million-tokens rates (USD), as of 2026-05. Best-effort; users should treat
# spend tracking as advisory. Unknown models -> 0.0 cost.
_RATES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _RATES.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


class AnthropicClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return ModelResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=_cost(self.model, resp.usage.input_tokens, resp.usage.output_tokens),
        )
