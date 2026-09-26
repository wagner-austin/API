"""The optimizer vocabulary shared by every torch training path.

Model-Trainer, covenant_nn, covenant_ml and covenant-radar-api each accepted
the same three optimizer words and each kept its own map from a word to the
torch.optim class it names. Both live here once so the words and the class
names cannot drift apart between services.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class OptimizerName(StrEnum):
    """An optimizer a training config may name, by its wire word."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"

    @property
    def torch_class_name(self) -> str:
        """The name of the torch.optim class this optimizer constructs.

        Returns:
            The attribute name on ``torch.optim``, e.g. ``"AdamW"``.
        """
        return _TORCH_CLASS_NAMES[self]


_TORCH_CLASS_NAMES: Final[dict[OptimizerName, str]] = {
    OptimizerName.ADAMW: "AdamW",
    OptimizerName.ADAM: "Adam",
    OptimizerName.SGD: "SGD",
}


__all__ = ["OptimizerName"]
