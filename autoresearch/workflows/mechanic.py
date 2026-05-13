"""MECHANIC workflow — pre-failure health check on a running compute job.

v0: opt-in; fires once at compute start (or at user request) and looks at
the target Run's findings / log tail / heartbeat to decide whether the run
is plausibly going to succeed or already in trouble.

It does NOT loop / poll / babysit continuously in v0 — that's the v1 upgrade,
along with the ability to actually intervene (cancel, redispatch with
recovery params).

### v0 outputs

  - One OBSERVATION finding on the *mechanic's own Run* describing what it
    saw on the *target Run*. The finding's text includes one of:
      - "HEALTHY: looks like training is progressing"
      - "STALLED: no heartbeat update in N min; consider cancel"
      - "OOM_LIKELY: log shows growing memory; consider redispatch on bigger GPU"
      - "UNKNOWN: insufficient signal"
  - A `recommended_action` field in the result dict (`continue` / `cancel`
    / `redispatch`) — the local Claude / `/transfer` skill can act on it
    or ignore it.

### v1 trajectory

  - Continuous polling on a configurable cadence
  - Real intervention: call `cancel(run_id)` or `redispatch` directly
  - Integration with the `takeover` MCP flow so a human can override
  - Detect common failure modes (OOM, CUDA error, dataset 404, dead heartbeat)
    via regex over the training-log snapshot rather than only LLM judgement
"""

from __future__ import annotations

import json
import re
from typing import Any

from autoresearch.backends.models.base import ModelClient
from autoresearch.backends.storage import StorageBackend
from autoresearch.core import findings as findings_mod
from autoresearch.core import logs as logs_mod
from autoresearch.core.agent_runner import run_agent
from autoresearch.core.run import Run, RunStatus


MECHANIC_SYSTEM = """You are a mechanic agent doing a one-shot health check on
an in-flight GPU training run. The researcher is off their laptop; you're the
only one watching.

Inputs you'll receive:
  - The target Run's params + status
  - The 200-line tail of its log
  - Its findings so far (chronological)

Decide whether the run is:
  - HEALTHY: training is progressing; no action needed
  - STALLED: heartbeat/log shows no progress; recommend cancel
  - OOM_LIKELY: log shows OOM patterns or memory growth; recommend redispatch on bigger GPU
  - UNKNOWN: not enough signal to judge

End your output with EXACTLY one of these tokens on its own line:
RECOMMENDATION: continue
RECOMMENDATION: cancel
RECOMMENDATION: redispatch
RECOMMENDATION: investigate

Be terse — 3-5 short bullets max before the recommendation line."""


_RECOMMENDATION_RE = re.compile(r"^RECOMMENDATION:\s*(\w+)\s*$", re.M)


def mechanic(
    run: Run,
    *,
    storage: StorageBackend,
    workspace: Any,  # not used in v0; kept for signature parity with prepare/postflight
    model_client: ModelClient,
    target_run_id: str | None = None,
) -> dict[str, Any]:
    """Inspect `target_run_id`'s state and return a recommendation.

    The mechanic's own Run records the inspection-cost in budget; the
    OBSERVATION finding lands under the mechanic's run id, NOT the target
    run id, so a reader looking up a transfer run's findings doesn't see
    surprise mechanic chatter mixed in. The recommendation surfaces via
    the result dict; the caller decides what to do.
    """
    target_run_id = target_run_id or (run.params or {}).get("target_run_id")
    if not target_run_id:
        raise ValueError(
            "mechanic: missing target_run_id (pass via run.params or call arg)"
        )

    target = Run.load(storage, target_run_id)
    target_findings = findings_mod.list_findings(storage, target)
    log_tail = logs_mod.tail(storage, target, lines=200)

    user = (
        f"Target Run: {target.id}\n"
        f"  workflow: {target.workflow}\n"
        f"  pipeline: {target.pipeline_name}\n"
        f"  status:   {target.status.value}\n"
        f"  params:   {json.dumps(target.params, default=str)}\n"
        f"  last_heartbeat_at: {target.last_heartbeat_at}\n"
        f"  last_error:        {target.last_error}\n\n"
        f"Findings so far ({len(target_findings)}):\n"
        + "\n".join(f"  [{f.type.value}] {f.body[:300]}" for f in target_findings[:20])
        + f"\n\nLog tail (last 200 lines):\n```\n{log_tail or '(empty)'}\n```\n"
    )

    # Mechanic is single-shot (no tool loop). Cost is naturally bounded by
    # max_tokens=1000 — worst case ~$0.10. We don't wire SDK's max_budget_usd
    # because there's no tool loop to cap, but we still honor a $10 envelope
    # at the Run level: if the Run has already spent more than that on prior
    # calls, refuse to proceed.
    from autoresearch.core import budget
    _MECHANIC_BUDGET_CAP = 10.0
    if Run.load(storage, run.id).budget_spent_usd >= _MECHANIC_BUDGET_CAP:
        raise RuntimeError(
            f"mechanic refused: Run {run.id} has already spent more than "
            f"the per-agent cap of ${_MECHANIC_BUDGET_CAP:.2f}"
        )
    agent = run_agent(
        run=run, storage=storage, client=model_client,
        system=MECHANIC_SYSTEM, user=user, max_tokens=1000, label="mechanic",
    )

    m = _RECOMMENDATION_RE.search(agent.text)
    recommendation = m.group(1).lower() if m else "investigate"

    fresh = Run.load(storage, run.id)
    fresh.status = RunStatus.COMPLETED
    fresh.save(storage)

    return {
        "workflow": "mechanic",
        "target_run_id": target_run_id,
        "recommendation": recommendation,
        "review_text": agent.text,
        "cost_usd": agent.cost_usd,
    }
