"""Budget — advisory spend tracking and cap enforcement.

Each Run has `budget_cap_usd` (set at dispatch, default
`Settings.default_budget_usd = $30`) and `budget_spent_usd` (accumulated as
the run progresses).

### What's tracked today

  - **Validator LLM calls**: `core.validation.run_validation` charges
    `ModelResponse.cost_usd` via `add_spend`. Covers the preflight,
    postflight, and error-summarization calls.
  - **`summarize_run` MCP tool**: `controller.mcp_surface.summarize_run`
    charges the same way.

### What's NOT yet tracked (v2 TODOs — see notes/ideas.md)

  - **Hardware advisor** (`core.hardware.recommend` with `client=...`):
    the call costs ~$0.001-$0.01 each but isn't aggregated into
    `budget_spent_usd`. Tiny, but should still be tracked for honesty.
  - **Prep / mechanic / postflight agents** (future): when these land
    as pod-side Claude Agent SDK loops, every tool-call cycle needs to
    flow back into `budget_spent_usd` of the parent Run.
  - **Compute pod-hours**: the pipeline result reports an extrapolated
    cost based on `cost_per_hour × elapsed`, but the in-flight pod-hour
    spend is never added to `budget_spent_usd`. So `check()` cannot
    hard-stop a Run that's blowing its budget on compute, only on LLM.
    Wiring this up requires the supervisor to poll RunPod for billable
    seconds and call `add_spend` periodically.

Enforcement is *advisory*, not hard: in-flight tokens or in-flight
pod-seconds can push past the cap. The contract is "hard-stop at the
next checkpoint after spend crosses cap" — and only for spend that's
actually tracked (so currently, only LLM spend).
"""

from __future__ import annotations

from autoresearch.backends.storage import StorageBackend
from autoresearch.core.run import Run


class BudgetExceeded(Exception):
    """Raised when the run has crossed its budget cap."""


def remaining(run: Run) -> float:
    return max(0.0, run.budget_cap_usd - run.budget_spent_usd)


def check(run: Run) -> None:
    """Raise BudgetExceeded if the run has crossed its cap."""
    if run.budget_cap_usd > 0 and run.budget_spent_usd >= run.budget_cap_usd:
        raise BudgetExceeded(
            f"Run {run.id} spent ${run.budget_spent_usd:.4f} of ${run.budget_cap_usd:.4f}"
        )


def add_spend(storage: StorageBackend, run: Run, usd: float) -> Run:
    """Add `usd` to the run's spent total and persist. Returns the updated Run.

    Re-reads the Run from storage to minimize stale-write races (still single-writer
    per run, but the runner may have updated other fields between the caller's last
    save and this add)."""
    fresh = Run.load(storage, run.id)
    fresh.budget_spent_usd += usd
    fresh.save(storage)
    return fresh
