"""WandBTracker — v2 stub.

The RunTracker protocol exists so users who already use WandB can plug in for
free; v1 doesn't ship the adapter because JSONStorageTracker covers the basic
needs and WandB adds an external dependency + account.

Implementation outline:
  - In __init__, lazy-import wandb, `wandb.init(project=..., id=run_id, resume="allow")`.
  - log_metric → `wandb.log({key: value}, step=step)`.
  - log_metrics → `wandb.log(metrics, step=step)`.
"""

from __future__ import annotations


_NOT_IMPLEMENTED = (
    "WandBTracker is a v1 stub. JSONStorageTracker covers v1 needs; implement WandB "
    "when a user actually wants to plug into their existing WandB workspace."
)


class WandBTracker:
    name = "wandb"

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def log_metric(self, run_id: str, key: str, value: float, *, step: int | None = None) -> None:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def log_metrics(self, run_id: str, metrics: dict[str, float], *, step: int | None = None) -> None:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED)
