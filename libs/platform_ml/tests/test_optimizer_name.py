"""Tests for the shared optimizer vocabulary."""

from __future__ import annotations

import torch.optim

import platform_ml
from platform_ml.optimizer_name import OptimizerName


def test_words_are_the_wire_vocabulary() -> None:
    """The members carry exactly the three optimizer words, in declaration order."""
    assert [member.value for member in OptimizerName] == ["adamw", "adam", "sgd"]


def test_every_member_names_a_real_torch_optimizer_class() -> None:
    """Each member's torch class name is the name of the torch.optim class it means."""
    expected: dict[OptimizerName, type[torch.optim.Optimizer]] = {
        OptimizerName.ADAMW: torch.optim.AdamW,
        OptimizerName.ADAM: torch.optim.Adam,
        OptimizerName.SGD: torch.optim.SGD,
    }
    for member in OptimizerName:
        assert member.torch_class_name == expected[member].__name__


def test_package_exports_the_same_class() -> None:
    """The package root re-exports the module's class, not a copy."""
    assert platform_ml.OptimizerName is OptimizerName
