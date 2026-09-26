"""The one place a fine-tuning strategy's name is written down.

WHY THIS MODULE EXISTS. The set of strategy names was written out nine times
across seven files -- the ``StrategyName`` literal, the ``finetuning_strategy``
field on :class:`~model_trainer.core.contracts.model.ModelTrainConfig`, the
request literal on both the API schema and its validator, a
``_FINETUNING_STRATEGIES`` frozenset, and a ``_VALID_STRATEGY_NAMES`` frozenset
beside a hand-written narrowing chain in the ``hf_lm`` backend. Adding a
strategy meant finding all nine. Missing one did not fail to compile: a request
naming the new strategy would pass the API validator and then be refused deeper
in, or be accepted everywhere and silently dropped from a checkpoint's metadata,
depending on which copy was stale.

That is the drift this module removes. :class:`StrategyName` is declared once
here and imported everywhere else, so a new strategy is one new member. The
enum is both the type and the iterable set, which is why no parallel tuple of
names and no narrowing chain sit beside it: :func:`require_strategy_name`
narrows through :func:`platform_core.members.find_member`, and a message or a
test iterates the enum itself.
"""

from __future__ import annotations

from enum import StrEnum

from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
from platform_core.members import find_member


class StrategyName(StrEnum):
    """How a model is adapted before training.

    ``full`` trains every parameter. ``lora`` and ``qlora`` train low-rank
    adapters over a frozen base, the second over a quantized one.
    ``cartridge`` trains a key-value prefix over a frozen base, touching no
    weight at all. Members are declared in registration order.
    """

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    CARTRIDGE = "cartridge"


def require_strategy_name(value: str) -> StrategyName:
    """Narrow an untrusted string to a declared strategy name.

    The single entry point from string data -- a request body, a queue payload,
    a checkpoint's metadata -- into the typed name. Callers that already hold a
    :class:`StrategyName` do not need it.

    Args:
        value: String to narrow, from outside this process.

    Returns:
        The member whose value is ``value``.

    Raises:
        AppError: With ``STRATEGY_NAME_UNKNOWN`` when the string names no
            declared strategy. A 400, because the caller chose the value.
    """
    name = find_member(value, StrategyName)
    if name is None:
        declared = ", ".join(sorted(StrategyName))
        raise AppError(
            ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN,
            (
                f"no fine-tuning strategy is named {value!r}; "
                f"the declared strategies are [{declared}]"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN),
        )
    return name


__all__ = [
    "StrategyName",
    "require_strategy_name",
]
