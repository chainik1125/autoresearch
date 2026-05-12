"""Verify v1 stubs are importable and raise clear NotImplementedError on use."""

from __future__ import annotations

import pytest

from autoresearch.backends.compute.modal import ModalCompute
from autoresearch.backends.models.openai import OpenAIClient
from autoresearch.backends.tracking.wandb import WandBTracker
from autoresearch.workflows import multi_model_review, night_run, replicate, sweep


def test_modal_compute_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="v1 stub"):
        ModalCompute()


def test_openai_client_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="v1 stub"):
        OpenAIClient()


def test_wandb_tracker_stub_raises() -> None:
    with pytest.raises(NotImplementedError, match="v1 stub"):
        WandBTracker()


@pytest.mark.parametrize("fn", [
    replicate.replicate,
    multi_model_review.multi_model_review,
    sweep.sweep,
    night_run.night_run,
])
def test_workflow_stubs_raise(fn) -> None:
    with pytest.raises(NotImplementedError):
        fn()
