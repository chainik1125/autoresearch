"""Agent runner — wraps a single LLM call for the prep / mechanic / postflight workflows.

v0: this is a single-shot `client.complete()` call wrapped in our standard
finding-emission + budget-charging machinery. NOT a Claude Agent SDK tool-using
loop yet — that's v1. The point of v0 is to land the workflow scaffolding so
each agent type has a real home, and the system prompts have a single grep-able
location to iterate on.

### Why single-shot is OK for v0

  - Prep: most projects have at most 2-3 machine-specific assumptions to flag
    (hardcoded paths, env vars). A single carefully-prompted call can list them.
    Applying patches is v1 (needs the tool-using loop + plan-mode UX).
  - Mechanic: opt-in, fires once at compute start. "Does this run look likely
    to succeed?" One call.
  - Postflight: one call to produce the experiment_summary.md. Writing files
    + git operations are subprocess steps the workflow does after the LLM
    call (not via the agent's tool use).

### v1 trajectory (notes/ideas.md)

Replace `client.complete()` here with a Claude Agent SDK `query()` that has
the Read/Edit/Bash toolset enabled. Each workflow's system prompt + max_turns
+ allowed tools become configuration on this runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.core import budget, findings
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run


@dataclass
class AgentResult:
    """What `run_agent` returns to the calling workflow."""

    text: str
    cost_usd: float
    input_tokens: int
    output_tokens: int


def run_agent(
    *,
    run: Run,
    storage: StorageBackend,
    client: ModelClient,
    system: str,
    user: str,
    max_tokens: int = 2000,
    label: str = "agent",
) -> AgentResult:
    """Run one LLM call, charge the Run's budget, emit a finding with the output.

    `label` shows up in the finding body so a later `list_findings` reader can
    tell prep vs mechanic vs postflight apart without parsing the rest of the
    text.
    """
    resp = client.complete(system=system, user=user, max_tokens=max_tokens)
    if resp.cost_usd > 0:
        budget.add_spend(storage, run, resp.cost_usd)
    findings.append(
        storage, run, FindingType.OBSERVATION,
        f"=== {label} ===\n{resp.text}",
    )
    return AgentResult(
        text=resp.text,
        cost_usd=resp.cost_usd,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
