"""Hooks for the command-line entries - production defaults, tests override.

Production sets these to the real implementations at import. Tests replace
them with fakes before exercising the code under test, so there is no
conditional in the entry itself -- it calls the hook.

Only the two seams that need real weights and a real GPU are here. Everything
else in the scorer is pure and is exercised directly, because a fake in front
of pure code tests the fake.
"""

from __future__ import annotations

from typing import Protocol

from platform_core.determinism_record import DeterminismRecord

from model_trainer.core._hook_protocols_ml import PinTorchThreadsProto
from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItem
from model_trainer.core.contracts.model import PreparedLMModel


class LoadHubModelProto(Protocol):
    """Protocol for loading an untrained model straight from the hub."""

    def __call__(self, hub_model_id: str, /) -> PreparedLMModel:
        """Load the named model with nothing applied to it."""
        ...


class ScoreClozeProto(Protocol):
    """Protocol for the cloze scorer.

    Keyword-only, matching the real signature: the scorer takes five
    arguments whose order carries no meaning and would be easy to transpose.
    """

    def __call__(
        self,
        *,
        items: list[ClozeItem],
        model: PreparedLMModel,
        device: str,
        max_seq_len: int,
    ) -> ClozeEvalResult:
        """Score every item and report accuracy against the guessing baseline."""
        ...


class ApplyDeterminismProto(Protocol):
    """Protocol for the determinism pin.

    Behind a hook because it writes process-global torch state and the
    environment, which a test must be able to observe without a real CUDA
    stack.
    """

    def __call__(self) -> DeterminismRecord:
        """Pin kernel determinism and report what was actually applied."""
        ...


def _default_load_hub_model(hub_model_id: str, /) -> PreparedLMModel:
    """Production hub loader - used as default hook.

    Imported inside the function so that importing this module does not pull
    torch into a process that only wanted to parse a command line and print
    a usage error.

    Args:
        hub_model_id: HuggingFace model id, for example ``gpt2-medium``.

    Returns:
        The prepared model, with nothing applied to it.
    """
    from model_trainer.core.services.model.backends.hf_lm.io import (
        load_prepared_hf_lm_from_hub,
    )

    return load_prepared_hf_lm_from_hub(hub_model_id)


def _default_score_cloze(
    *,
    items: list[ClozeItem],
    model: PreparedLMModel,
    device: str,
    max_seq_len: int,
) -> ClozeEvalResult:
    """Production scorer - used as default hook.

    Unpacks the prepared model into the model and its encoder, which is the
    only reason this is not the scorer itself: the entry holds a
    PreparedLMModel and the scorer takes the two halves.

    Args:
        items: The cloze items to score.
        model: The prepared model and its encoder.
        device: Device to score on.
        max_seq_len: Token budget per item.

    Returns:
        The scored result.
    """
    from model_trainer.core.services.model.cloze import score_cloze_items

    return score_cloze_items(
        items=items,
        model=model.model,
        encoder=model.tok_for_dataset,
        device=device,
        max_seq_len=max_seq_len,
    )


def _default_apply_determinism() -> DeterminismRecord:
    """Production determinism pin - used as default hook.

    Delegates to the same hook the workers use, so a run scored from the
    command line and one scored through the queue pin identically. A second
    spelling here would be a second posture nobody noticed diverging.

    Returns:
        What was actually applied.
    """
    from model_trainer.core import _test_hooks as core_hooks

    return core_hooks.apply_determinism_hook()


def _default_pin_torch_threads(threads: int) -> int:
    """Production torch thread pin - used as default hook.

    Delegates to the worker's hook for the same reason
    :func:`_default_apply_determinism` does: a probe run from the command
    line and a job run through the queue must pin by the same call, or the
    two postures diverge without anyone noticing.

    Args:
        threads: Count to request.

    Returns:
        The count torch resolved to, which may differ from the request.
    """
    from model_trainer.core import _test_hooks as core_hooks

    return core_hooks.pin_torch_threads(threads)


load_hub_model: LoadHubModelProto = _default_load_hub_model

score_cloze: ScoreClozeProto = _default_score_cloze

apply_determinism_hook: ApplyDeterminismProto = _default_apply_determinism

pin_torch_threads: PinTorchThreadsProto = _default_pin_torch_threads


__all__ = [
    "ApplyDeterminismProto",
    "LoadHubModelProto",
    "ScoreClozeProto",
    "apply_determinism_hook",
    "load_hub_model",
    "score_cloze",
]
