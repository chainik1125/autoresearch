"""Hardware selection tests.

Covers the deterministic layer of `core/hardware.py` plus the LLM-advisor
integration with a fake ModelClient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from autoresearch.backends.models.base import ModelResponse
from autoresearch.core.hardware import GpuOffer, recommend, select_gpu_offers


def _offer(id: str, mem: int, price: float | None, in_dc: bool = True) -> GpuOffer:
    return GpuOffer(id=id, memory_gb=mem, price_per_hour=price, available_in_dc=in_dc)


def test_filters_by_vram_floor() -> None:
    offers = [
        _offer("A40", 48, 0.35),
        _offer("H100", 80, 2.69),
        _offer("RTX 4090", 24, 0.40),
    ]
    # Explicit `cheapest` so we test the floor-filter without the
    # fastest_least_complicated headroom rule mucking with results.
    picks = select_gpu_offers(offers, required_vram_gb=48, prefer="cheapest")
    assert "RTX 4090" not in picks
    assert "A40" in picks
    assert "H100" in picks


def test_cheapest_ranking_when_explicit() -> None:
    offers = [
        _offer("H100", 80, 2.69),
        _offer("A40", 48, 0.35),
        _offer("L40S", 48, 0.79),
        _offer("A6000", 48, 0.33),
    ]
    picks = select_gpu_offers(offers, required_vram_gb=48, prefer="cheapest")
    # A6000 ($0.33) < A40 ($0.35) < L40S ($0.79) < H100 ($2.69)
    assert picks == ["A6000", "A40", "L40S", "H100"]


def test_data_center_filter_drops_out_of_stock() -> None:
    offers = [
        _offer("A40", 48, 0.35, in_dc=False),
        _offer("L40S", 48, 0.79, in_dc=True),
        _offer("A6000", 48, None, in_dc=False),
    ]
    picks = select_gpu_offers(offers, required_vram_gb=48, data_center_id="US-CA-2")
    assert picks == ["L40S"]


def test_max_candidates_cap() -> None:
    offers = [_offer(f"GPU-{i}", 80, 1.0 + i * 0.1) for i in range(10)]
    picks = select_gpu_offers(offers, required_vram_gb=80, max_candidates=3)
    assert len(picks) == 3


def test_empty_when_nothing_fits() -> None:
    offers = [
        _offer("A40", 48, 0.35),
        _offer("RTX 4090", 24, 0.40),
    ]
    picks = select_gpu_offers(offers, required_vram_gb=80)
    assert picks == []


def test_biggest_memory_first_heuristic() -> None:
    offers = [
        _offer("A40", 48, 0.35),
        _offer("H100", 80, 2.69),
        _offer("B200", 180, 5.98),
    ]
    picks = select_gpu_offers(offers, required_vram_gb=48, prefer="biggest_memory_first")
    assert picks == ["B200", "H100", "A40"]


def test_exclude_unavailable_drops_offers_without_price() -> None:
    offers = [
        _offer("A40", 48, None),                     # advertised but no stock anywhere
        _offer("A6000", 48, 0.33),
    ]
    picks = select_gpu_offers(
        offers, required_vram_gb=48, exclude_unavailable=True,
    )
    assert picks == ["A6000"]


def test_include_unavailable_when_caller_explicitly_opts_in() -> None:
    offers = [
        _offer("A40", 48, None),
        _offer("A6000", 48, 0.33),
    ]
    picks = select_gpu_offers(
        offers, required_vram_gb=48, exclude_unavailable=False,
    )
    # A40 (no price) ranks after A6000 (priced); both returned.
    assert "A6000" in picks and "A40" in picks


def test_fastest_least_complicated_picks_smallest_bucket_newest_within() -> None:
    """For Qwen-7B (required=30): should pick L40S (newest 48GB) over B200 (overkill)
    and over A40 (older 48GB)."""
    offers = [
        _offer("A40", 48, 0.35),
        _offer("A6000", 48, 0.33),
        _offer("L40", 48, 0.69),
        _offer("L40S", 48, 0.79),
        _offer("H100", 80, 2.69),
        _offer("B200", 180, 5.98),
    ]
    picks = select_gpu_offers(
        offers, required_vram_gb=30, prefer="fastest_least_complicated",
    )
    # Smallest fitting bucket (>= 30 * 1.2 = 36) = 48GB cards. Within them,
    # newest (highest price) first: L40S > L40 > A40 > A6000.
    assert picks[0] == "L40S"
    # H100 / B200 should come AFTER the 48GB cluster
    h100_pos = picks.index("H100") if "H100" in picks else 99
    l40s_pos = picks.index("L40S")
    assert l40s_pos < h100_pos


# ----------------------------------------------------------------------
# Deterministic recommend() — confidence rules
# ----------------------------------------------------------------------


def test_recommend_deterministic_high_confidence_cheap_picks() -> None:
    """Cheap + headroom + multiple options → high confidence."""
    offers = [
        _offer("L40S", 48, 0.79),
        _offer("L40", 48, 0.69),
        _offer("A6000", 48, 0.33),
        _offer("H100", 80, 2.69),
    ]
    rec = recommend(
        offers,
        required_vram_gb=30,
        data_center_id="US-CA-2",
    )
    assert rec.confidence == "high"
    assert rec.picks[0] == "L40S"
    assert rec.alternatives == []  # high confidence omits alternatives


def test_recommend_deterministic_needs_review_when_expensive() -> None:
    """Top pick > $3/hr → needs_review."""
    offers = [_offer("B200", 180, 5.98)]
    rec = recommend(
        offers,
        required_vram_gb=80,
        data_center_id="US-CA-2",
    )
    assert rec.confidence == "needs_review"
    assert "steep" in rec.rationale.lower() or "$5.98" in rec.rationale


def test_recommend_deterministic_needs_review_when_no_fits() -> None:
    """No GPU in stock with enough VRAM → needs_review, empty picks."""
    offers = [_offer("RTX 4090", 24, 0.40)]
    rec = recommend(offers, required_vram_gb=80, data_center_id="US-CA-2")
    assert rec.picks == []
    assert rec.confidence == "needs_review"


# ----------------------------------------------------------------------
# LLM-advised recommend() with a fake client
# ----------------------------------------------------------------------


@dataclass
class _FakeClient:
    """Returns the configured `text` from `complete()`."""
    model: str = "fake-model"
    response_text: str = ""

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> ModelResponse:
        return ModelResponse(text=self.response_text)


def test_recommend_uses_advisor_when_client_and_intent_present() -> None:
    """Advisor response (well-formed JSON) wins over deterministic picks."""
    offers = [
        _offer("L40S", 48, 0.79),
        _offer("H100", 80, 2.69),
    ]
    advisor_response = json.dumps({
        "picks": ["H100"],          # Different from deterministic default (L40S)
        "confidence": "high",
        "rationale": "H100 is the more reliable choice for sae_lens training.",
        "alternatives": [],
    })
    client = _FakeClient(response_text=advisor_response)
    rec = recommend(
        offers,
        required_vram_gb=30,
        data_center_id="US-CA-2",
        intent="reliable production run",
        pipeline_name="sae_training",
        client=client,
    )
    assert rec.picks == ["H100"]
    assert rec.confidence == "high"
    assert "reliable" in rec.rationale.lower()


def test_recommend_falls_back_to_deterministic_on_bad_advisor_json() -> None:
    """Malformed advisor output → fall through to deterministic, never block."""
    offers = [_offer("L40S", 48, 0.79), _offer("A6000", 48, 0.33)]
    client = _FakeClient(response_text="not json at all")
    rec = recommend(
        offers,
        required_vram_gb=30,
        data_center_id="US-CA-2",
        intent="anything",
        client=client,
    )
    # Should still get a valid picks list
    assert rec.picks
    # And it's the deterministic top pick (L40S — newest 48GB)
    assert rec.picks[0] == "L40S"


def test_recommend_invokes_advisor_for_cheapest_prefer_too() -> None:
    """Regression: even when caller asks for `prefer='cheapest'`, the advisor
    is still invoked (not short-circuited to deterministic-only). The whole
    point of Claude-in-the-loop is to apply context the ranker can't see —
    skipping the advisor for agent workflows defeats that.

    Setup: deterministic ranker would pick A4000 (cheapest sufficient).
    Advisor overrides with A5000 citing a hypothetical past failure on A4000.
    We assert the advisor's override was honored.
    """
    offers = [
        _offer("A4000", 16, 0.17),
        _offer("A5000", 24, 0.26),
        _offer("L4",    24, 0.43),
    ]
    advisor_response = json.dumps({
        "picks": ["A5000", "L4"],
        "confidence": "high",
        "rationale": "A4000 OOM'd on last prep run with sae_lens deps; A5000 is the next cheapest with margin.",
        "alternatives": [],
    })
    client = _FakeClient(response_text=advisor_response)
    rec = recommend(
        offers,
        required_vram_gb=8,
        data_center_id="US-CA-2",
        intent="prep agent: cheap + reliable",
        pipeline_name="prep",
        workflow="prepare",
        client=client,
        prefer="cheapest",  # the key bit — advisor still consulted
    )
    assert rec.picks == ["A5000", "L4"]    # advisor's override won
    assert "A4000 OOM" in rec.rationale     # context made it through


def test_recommend_drops_advisor_picks_out_of_dc_stock() -> None:
    """Regression: an advisor pick for a SKU marked OUT OF STOCK in the
    target DC must be dropped, even though VRAM is sufficient.

    Backstory: a real /transfer dispatch landed on B200 even though B200
    wasn't in US-CA-2 stock. The advisor saw B200 in the offers table
    (with the OOS marker for context), picked it anyway, and validation
    only checked VRAM — not DC stock. Result: dispatch 422'd, user paid
    for the failed attempt.
    """
    offers = [
        _offer("A40",  48, None),      # OUT OF STOCK in US-CA-2 (price=None)
        _offer("L40S", 48, 0.79),      # in stock
        _offer("B200", 180, None),     # OUT OF STOCK (price=None)
    ]
    advisor_response = json.dumps({
        "picks": ["B200", "L40S"],     # advisor picks B200 first, despite OOS marker
        "confidence": "high",
        "rationale": "B200 is newest gen.",
        "alternatives": [],
    })
    client = _FakeClient(response_text=advisor_response)
    rec = recommend(
        offers,
        required_vram_gb=30,
        data_center_id="US-CA-2",
        intent="x",
        client=client,
    )
    # B200 filtered out (OOS), only L40S survives.
    assert "B200" not in rec.picks
    assert rec.picks == ["L40S"]


def test_recommend_drops_hallucinated_advisor_picks() -> None:
    """Advisor names a GPU that isn't in offers → it's dropped, not blindly trusted."""
    offers = [_offer("L40S", 48, 0.79)]
    advisor_response = json.dumps({
        "picks": ["NVIDIA H1000 4000GB", "L40S"],  # First is hallucinated
        "confidence": "high",
        "rationale": "x",
        "alternatives": [],
    })
    client = _FakeClient(response_text=advisor_response)
    rec = recommend(
        offers,
        required_vram_gb=30,
        data_center_id="US-CA-2",
        intent="x",
        client=client,
    )
    assert rec.picks == ["L40S"]  # hallucination filtered, L40S kept
