"""SWEEP — v2 stub.

Use case: "Run the FRA pipeline across N parameter values." E.g. across all
checkpoints in a training run, or across a grid of hyperparameters.

This workflow is the second pressure test for the Pipeline protocol. v1 has
N=1 (TRANSFER); SWEEP is N=many. When this lands, the Pipeline protocol may
need a small refactor — that's expected and called out in the plan's
"Critical risks #1".

Implementation outline (when this lands):
  - Inputs: pipeline + list[params].
  - Strategy A — sequential on one pod (cheapest if pipeline.run() is short).
  - Strategy B — fan out N parallel pods, each running one param config
    (Modal ephemeral if pure-API, RunPod sessions if GPU-heavy).
  - Findings: one per param config, plus a consolidated comparison finding
    at the end.
  - Budget: per-config or aggregate cap.
"""

from __future__ import annotations


def sweep(*_args, **_kwargs):
    raise NotImplementedError(
        "SWEEP is a v2 workflow. It will exercise the Pipeline protocol at N>1, "
        "which may require small protocol refactors."
    )
