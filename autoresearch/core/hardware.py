"""Hardware selection — pick GPU type(s) for a dispatch.

The dispatcher hands the compute backend a *list* of acceptable GPU type IDs
(RunPod's `gpuTypeIds` accepts a list and picks any available one). This
module decides what goes in that list. It has two layers, both small enough
to iterate on independently:

  1. **Deterministic layer.** `select_gpu_offers(offers, ...)` filters by VRAM
     and DC, then sorts by a named heuristic ("cheapest", etc.). Pure
     Python, no I/O, fully unit-testable. Used by the supervisor (no LLM in
     the loop on restart) and as a fallback when the LLM advisor is
     unavailable or off.

  2. **LLM advisor layer.** `advise_gpu_offers(offers, ..., intent, client)`
     wraps the deterministic layer with an Anthropic call that gets richer
     context — the user's intent text, the pipeline's estimated duration,
     observed cost/perf tradeoffs — and returns its own ranked list. The
     prompt is a module-level constant (`_ADVISOR_PROMPT`) so it's a single
     grep-target to iterate on.

The two layers share `GpuOffer` and the same return type (`list[str]` of
gpuTypeIds in preference order), so callers can swap one for the other.

### Override hatch

If the dispatcher receives an explicit `gpu` argument (string or list), this
whole module is bypassed. That preserves the "I know exactly what I want"
path for advanced users while making auto-selection the default.

### What to iterate on here

This is one of the modules the user has flagged for active development.
Likely next changes:

- **More heuristics**: "best_perf_per_dollar" once we have observed
  tokens/sec per (gpu, pipeline) tuple stored somewhere.
- **Spot-pod fallthrough**: pass `interruptible=true` to the backend when
  the user signals tolerance for preemption (e.g. "this is exploratory").
- **Cross-DC search**: today we filter to the volume's DC. For pipelines
  that don't need volume reuse, allowing any DC opens way more inventory.
- **Prompt richness**: feed historical run data ("this pipeline averaged
  N hours on H100") to the advisor so it gives calibrated answers.
- **Streaming inventory probes**: today `list_gpu_offers` is a single
  snapshot. In a tight loop the advisor could re-query mid-decision to
  avoid picking something that goes dry in the next 30s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from autoresearch.backends.models.base import ModelClient


@dataclass(frozen=True)
class GpuOffer:
    """A single rentable GPU type, as the compute backend reports it.

    All four fields are needed for selection:
      - `id`: the backend-native identifier passed to the create-session API.
      - `memory_gb`: VRAM. Used to filter against pipeline requirements.
      - `price_per_hour`: USD/hour for on-demand (non-spot) pods. None means
        "no inventory at this price" or "price unknown" — gets filtered out
        by `select_gpu_offers` when `exclude_unavailable=True`.
      - `available_in_dc`: True if the offer has stock in the queried DC.
    """

    id: str
    memory_gb: int
    price_per_hour: float | None
    available_in_dc: bool


Heuristic = Literal[
    "fastest_least_complicated",  # default: smallest VRAM bucket that comfortably fits, newest gen within bucket
    "cheapest",
    "biggest_memory_first",
    "fastest_first",
]


@dataclass(frozen=True)
class HardwareRecommendation:
    """The output of `recommend()` — what the /transfer skill consumes.

    - `picks`: gpuTypeIds to hand the backend, preference order. Empty list
      means "nothing safe to recommend; the skill must escalate."
    - `confidence`: "high" → the skill can auto-dispatch and just narrate
      what it picked; "needs_review" → the skill should ask the user
      before dispatching.
    - `rationale`: one or two sentences the skill surfaces to the user
      either as an FYI (high confidence) or as the framing for the
      conversation question (needs_review).
    - `alternatives`: secondary picks worth mentioning when prompting the
      user. Empty in the high-confidence case.
    """

    picks: list[str]
    confidence: Literal["high", "needs_review"]
    rationale: str
    alternatives: list[str]


# ---------------------------------------------------------------------------
# Layer 1: deterministic selection. Pure-Python, unit-testable.
# ---------------------------------------------------------------------------


def select_gpu_offers(
    offers: list[GpuOffer],
    *,
    required_vram_gb: int,
    data_center_id: str | None = None,
    prefer: Heuristic = "fastest_least_complicated",
    max_candidates: int = 5,
    exclude_unavailable: bool = True,
) -> list[str]:
    """Filter + rank `offers`; return up to `max_candidates` gpuTypeIds.

    Empty list means nothing fits — the caller decides whether to fall back
    to a manual override, wait, or fail.

    Pure function: same inputs → same outputs, no I/O. The compute backend's
    `list_gpu_offers()` is where the network call lives.
    """
    candidates = [o for o in offers if o.memory_gb >= required_vram_gb]

    if data_center_id is not None:
        candidates = [o for o in candidates if o.available_in_dc]

    if exclude_unavailable:
        candidates = [o for o in candidates if o.price_per_hour is not None]

    if prefer == "fastest_least_complicated":
        # "Smallest VRAM bucket that comfortably fits, newest gen within bucket."
        # Price-per-hour is the proxy for "newer generation" — Hopper > Lovelace
        # > Ampere by both price and speed. The 1.2x headroom filter avoids
        # picking a card whose VRAM is exactly equal to the floor (no margin for
        # optimizer state quirks, fragmentation, the SAE Adam blowup we saw on
        # the Qwen-32B canary).
        headroom_floor = int(required_vram_gb * 1.2)
        with_headroom = [o for o in candidates if o.memory_gb >= headroom_floor]
        if with_headroom:
            candidates = with_headroom
        # else: nothing has headroom — accept tight-fit candidates from above.
        candidates.sort(key=lambda o: (o.memory_gb, -(o.price_per_hour or 0.0)))
    elif prefer == "cheapest":
        candidates.sort(
            key=lambda o: o.price_per_hour if o.price_per_hour is not None else float("inf"),
        )
    elif prefer == "biggest_memory_first":
        candidates.sort(key=lambda o: -o.memory_gb)
    elif prefer == "fastest_first":
        # No "speed" field on offers yet — approximate "fastest" with
        # "biggest_memory_first AND highest price" (a rough proxy for newer-
        # generation cards). When we add `tflops` or `tokens_per_sec_<model>`
        # this branch should use those directly.
        candidates.sort(key=lambda o: (-o.memory_gb, -(o.price_per_hour or 0.0)))
    else:  # pragma: no cover -- defensive
        raise ValueError(f"unknown heuristic {prefer!r}")

    return [o.id for o in candidates[:max_candidates]]


# ---------------------------------------------------------------------------
# Layer 2: LLM advisor. Iterates fast; this is the "smart" default for
# `/transfer`-style ergonomic dispatches.
# ---------------------------------------------------------------------------


# Iterate on me. This is the single grep-target for "how does the model
# decide what GPU to recommend?" Keep the prompt small enough that a Haiku-
# class model handles it cheaply.
#
# The output schema is intentionally narrow:
#   - `picks` MUST be a subset of the offered gpuTypeIds (validated server-
#     side; hallucinated names are dropped).
#   - `confidence` is two-valued — "high" means "just dispatch, narrate the
#     choice", "needs_review" means "ask the user before dispatching."
#   - `rationale` is a short sentence the /transfer skill surfaces to the
#     user verbatim, either as an FYI or as framing for the question.
#   - `alternatives` is for the needs_review case so the user has options
#     to pick between in conversation.
#
# Guidance for "needs_review" should fire when ANY of these is true:
#   - The cheapest sufficient option costs > $3/hr (real money, worth a
#     confirm step).
#   - The required VRAM is borderline against the smallest fitting card
#     (no headroom for surprise optimizer state — we saw this fail on
#     Qwen-32B/H100).
#   - Inventory is so thin the picks list has < 2 in-stock options.
#   - The user's intent string is empty / vague enough that the model can't
#     trade off speed vs cost.
_ADVISOR_PROMPT = """You are picking GPU hardware for a research run on RunPod.

Hard constraints:
- The run requires AT LEAST {required_vram_gb} GB of VRAM. Anything smaller will OOM.
- The volume is bound to data-center {data_center_id!r}. Offers outside that DC are not usable.

Default heuristic: "fastest, least complicated."
  - Prefer the smallest VRAM bucket that has headroom (>= 1.2x the requirement). Don't
    pay for a 180GB GPU when a 48GB one works.
  - Within that bucket, prefer the newer / faster card (higher price typically maps to
    newer generation: Blackwell > Hopper > Lovelace > Ampere).
  - For well-tested pipelines, prefer GPUs the project has used before (typically H100
    or B200 — those have been validated end-to-end).

Run context:
- Pipeline name: {pipeline_name}
- Estimated duration on a typical GPU: {estimated_minutes} minutes
- Operator intent: {intent}

Available offers (price USD/hour on-demand; "stock" = available NOW in the target DC):
{offers_table}

Return ONLY this JSON shape (no prose, no code fences):
{{
  "picks": ["<gpuTypeId from the offers above>", ...],
  "confidence": "high" | "needs_review",
  "rationale": "<one short sentence the user will read>",
  "alternatives": ["<gpuTypeId>", ...]
}}

Confidence rules:
- "high" — your top pick is a clear best-fit AND its hourly rate is reasonable
  (< $3/hr) AND there are at least 2 in-stock options as fallbacks.
- "needs_review" — borderline VRAM, expensive top pick (>= $3/hr), thin
  inventory, or the operator intent is too vague to trade off speed vs cost.
  Put the trade-off in `rationale` so the user can answer in conversation.

`picks` should have 3-5 entries when possible (RunPod uses them as a fallback ladder).
`alternatives` are only relevant for needs_review; include 1-2 plausible different
choices so the user has something to pick between.
"""


def _format_offers_for_prompt(offers: list[GpuOffer]) -> str:
    """Render the offers table for the advisor prompt. Sorted by price."""
    sorted_offers = sorted(
        offers,
        key=lambda o: o.price_per_hour if o.price_per_hour is not None else float("inf"),
    )
    lines = []
    for o in sorted_offers:
        price = f"${o.price_per_hour:.2f}/hr" if o.price_per_hour is not None else "no price"
        stock = "in stock" if o.available_in_dc else "OUT OF STOCK in this DC"
        lines.append(f"  - {o.id!r}: {o.memory_gb}GB VRAM, {price}, {stock}")
    return "\n".join(lines)


def _parse_json_lenient(text: str) -> object:
    """Parse JSON tolerantly. Strip ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def _deterministic_recommendation(
    offers: list[GpuOffer],
    *,
    required_vram_gb: int,
    data_center_id: str | None,
    max_candidates: int,
) -> HardwareRecommendation:
    """Produce a HardwareRecommendation without any LLM. The fallback path.

    The confidence rule here is conservative: high only if the top pick is
    < $3/hr AND has 1.2x VRAM headroom AND at least 2 picks total. Anything
    looser is "needs_review" so a human gets a chance to confirm.
    """
    picks = select_gpu_offers(
        offers,
        required_vram_gb=required_vram_gb,
        data_center_id=data_center_id,
        prefer="fastest_least_complicated",
        max_candidates=max_candidates,
    )
    if not picks:
        return HardwareRecommendation(
            picks=[],
            confidence="needs_review",
            rationale=(
                f"No GPUs in {data_center_id or 'any DC'} have >={required_vram_gb}GB VRAM "
                f"in stock right now. Switch DC, wait, or relax the VRAM floor."
            ),
            alternatives=[],
        )
    id_to_offer = {o.id: o for o in offers}
    top = id_to_offer[picks[0]]
    has_headroom = top.memory_gb >= int(required_vram_gb * 1.2)
    affordable = (top.price_per_hour or 0.0) < 3.0
    enough_fallbacks = len(picks) >= 2
    confident = has_headroom and affordable and enough_fallbacks
    rationale = (
        f"Top pick: {top.id!r} ({top.memory_gb}GB VRAM, "
        f"${top.price_per_hour:.2f}/hr). Picked via fastest-least-complicated "
        f"heuristic — smallest VRAM bucket with headroom over the {required_vram_gb}GB floor."
    )
    if not has_headroom:
        rationale += " WARNING: VRAM is at the floor — no headroom for optimizer-state surprises."
    if not affordable:
        rationale += f" Hourly rate is steep (${top.price_per_hour:.2f}/hr) — confirm before committing."
    if not enough_fallbacks:
        rationale += " Inventory is thin — only one in-stock fallback available."
    return HardwareRecommendation(
        picks=picks,
        confidence="high" if confident else "needs_review",
        rationale=rationale,
        alternatives=picks[1:3] if not confident else [],
    )


def recommend(
    offers: list[GpuOffer],
    *,
    required_vram_gb: int,
    data_center_id: str | None = None,
    intent: str | None = None,
    pipeline_name: str = "<unknown>",
    estimated_minutes: int = 0,
    client: "ModelClient | None" = None,
    max_candidates: int = 5,
) -> HardwareRecommendation:
    """Top-level: produce a HardwareRecommendation.

    With an `intent` AND a `client`, calls the LLM advisor and returns its
    structured response (validated against `offers`). Without either, uses
    the deterministic fastest-least-complicated heuristic with conservative
    confidence rules.

    The advisor's output is always validated:
      - Picks are filtered to gpuTypeIds that actually appear in `offers`.
      - Confidence is normalized to "high" | "needs_review".
      - If validation fails or the LLM errors, falls back to deterministic
        — dispatch never blocks on the advisor.
    """
    # Always available as a safe fallback for either layer.
    deterministic = _deterministic_recommendation(
        offers,
        required_vram_gb=required_vram_gb,
        data_center_id=data_center_id,
        max_candidates=max_candidates,
    )

    if client is None or intent is None or not deterministic.picks:
        return deterministic

    try:
        prompt = _ADVISOR_PROMPT.format(
            required_vram_gb=required_vram_gb,
            data_center_id=data_center_id or "any",
            max_candidates=max_candidates,
            pipeline_name=pipeline_name,
            estimated_minutes=estimated_minutes,
            intent=intent,
            offers_table=_format_offers_for_prompt(offers),
        )
        resp = client.complete(
            system="You select hardware for cloud GPU dispatches. Output strict JSON only, no prose.",
            user=prompt,
            max_tokens=400,
        )
        parsed = _parse_json_lenient(resp.text)
        if not isinstance(parsed, dict):
            raise ValueError("advisor response is not a JSON object")
        valid_ids = {o.id for o in offers if o.memory_gb >= required_vram_gb}
        picks = [p for p in parsed.get("picks", []) if isinstance(p, str) and p in valid_ids][:max_candidates]
        if not picks:
            raise ValueError("advisor returned no valid picks")
        confidence_raw = str(parsed.get("confidence", "")).lower()
        confidence: Literal["high", "needs_review"] = (
            "high" if confidence_raw == "high" else "needs_review"
        )
        rationale = str(parsed.get("rationale", "")).strip() or deterministic.rationale
        alternatives = [
            p for p in parsed.get("alternatives", [])
            if isinstance(p, str) and p in valid_ids and p not in picks
        ][:3]
        return HardwareRecommendation(
            picks=picks,
            confidence=confidence,
            rationale=rationale,
            alternatives=alternatives,
        )
    except Exception:  # noqa: BLE001 -- never block on advisor
        return deterministic
