"""NIGHT_RUN — v2 stub (and possibly not a separate workflow at all).

Use case: "Claude, keep this running overnight, fix bugs as they come up, let
me know in the morning."

This may NOT be a distinct workflow — more likely it's a *mode* on TRANSFER or
SWEEP that:
  - Sets a high budget cap and long heartbeat threshold.
  - Has the validation hooks summarize errors and retry the pipeline up to N
    times if the failure looks transient.
  - Pings the user (Slack? email?) when done or when stuck.

Worth deferring the decision on whether NIGHT_RUN is its own workflow vs a flag
on TRANSFER until we have lived experience with TRANSFER overnight a few times.
"""

from __future__ import annotations


def night_run(*_args, **_kwargs):
    raise NotImplementedError(
        "NIGHT_RUN may not need to be its own workflow — see the docstring for "
        "design notes. Deferred from v1."
    )
