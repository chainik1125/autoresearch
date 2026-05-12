"""Validation — bounded single-call LLM helpers.

These are not an agent loop. Each function makes exactly one ModelClient call,
returns text, and updates the run's spend. The caller decides whether to write
the returned text as a finding.

Workflows own the prompts (system + user templates); this module owns the
machinery (call client, accumulate cost, return text or skip if disabled).
"""

from __future__ import annotations

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.core import budget
from autoresearch.core.run import Run


def run_validator(
    storage: StorageBackend,
    run: Run,
    client: ModelClient,
    *,
    system: str,
    user: str,
    max_tokens: int = 1024,
) -> str:
    """Run a single validation call. Charges its cost to the run's budget.

    Returns the model's text response. Caller is responsible for writing it as
    a finding if desired.
    """
    resp = client.complete(system=system, user=user, max_tokens=max_tokens)
    if resp.cost_usd > 0:
        budget.add_spend(storage, run, resp.cost_usd)
    return resp.text
