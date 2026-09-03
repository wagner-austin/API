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

That is the drift this module removes. The literal is declared once here and
imported everywhere else, so a new strategy is one edit to :data:`StrategyName`
and one to :data:`STRATEGY_NAMES`. The ``strategy-names-single-source`` guard
rule refuses any other module that writes the member strings as a group.

WHY THE TUPLE AND THE NARROWING CHAIN ARE BOTH HERE, which looks like the
duplication this module exists to delete and is not. ``typing.get_args`` is
unavailable under this package's ``disallow_any_expr``, and mypy will not narrow
a ``str`` to a member of a variadic ``tuple[StrategyName, ...]`` through ``in``.
So the chain in :func:`require_strategy_name` is the only construct that turns
an untrusted string into the literal type, and the tuple is the only construct
that can be iterated for a message or a test. They are kept honest by
``test_every_declared_name_survives_a_round_trip``, which walks the tuple and
requires the chain to accept each member -- so a name added to one and not the
other fails a test rather than reaching a caller.
"""

from __future__ import annotations

from typing import Literal

from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)

StrategyName = Literal["full", "lora", "qlora"]
"""How a model is adapted before training.

``full`` trains every parameter. ``lora`` and ``qlora`` train low-rank adapters
over a frozen base, the second over a quantized one.
"""

STRATEGY_NAMES: tuple[StrategyName, ...] = ("full", "lora", "qlora")
"""Every declared strategy name, in registration order.

Iterable form of :data:`StrategyName`, for error messages and for the test that
holds the two in agreement.
"""


def require_strategy_name(value: str) -> StrategyName:
    """Narrow an untrusted string to a declared strategy name.

    The single entry point from string data -- a request body, a queue payload,
    a checkpoint's metadata -- into the typed name. Callers that already hold a
    :data:`StrategyName` do not need it.

    Args:
        value: String to narrow, from outside this process.

    Returns:
        The same name, typed as :data:`StrategyName`.

    Raises:
        AppError: With ``STRATEGY_NAME_UNKNOWN`` when the string names no
            declared strategy. A 400, because the caller chose the value.
    """
    if value == "full":
        return "full"
    if value == "lora":
        return "lora"
    if value == "qlora":
        return "qlora"
    declared = ", ".join(sorted(STRATEGY_NAMES))
    raise AppError(
        ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN,
        (f"no fine-tuning strategy is named {value!r}; the declared strategies are [{declared}]"),
        model_trainer_status_for(ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN),
    )


__all__ = [
    "STRATEGY_NAMES",
    "StrategyName",
    "require_strategy_name",
]
