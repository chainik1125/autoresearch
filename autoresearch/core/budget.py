"""Budget — advisory spend tracking and cap enforcement.

Budget is checked before each LLM call and before each pod-hour rollover. Enforcement
is *advisory*, not hard: in-flight tokens or in-flight pod-seconds can push past
the cap. The contract is "hard-stop at the next checkpoint after spend crosses cap."
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
