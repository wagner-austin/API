"""The declared strategy names, and the one function that admits a string."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.strategy_names import (
    StrategyName,
    require_strategy_name,
)


class TestTheDeclaredNames:
    """The enum is both the type and the iterable set of names."""

    def test_every_declared_name_survives_a_round_trip(self) -> None:
        """Every member's wire word narrows back to that same member."""
        assert [require_strategy_name(name.value) for name in StrategyName] == list(StrategyName)

    def test_the_declared_names_are_exactly_these(self) -> None:
        """Pins the words in registration order, so adding one is a deliberate edit.

        Asserted as a list rather than a membership check, so a name that is
        REMOVED fails too.
        """
        assert [str(name) for name in StrategyName] == ["full", "lora", "qlora", "cartridge"]


class TestRequireStrategyName:
    """The single door from an untrusted string to the typed name."""

    def test_an_undeclared_name_is_refused_with_its_own_code(self) -> None:
        """Not a generic decode error: the caller picked a value that does not exist."""
        with pytest.raises(AppError) as excinfo:
            require_strategy_name("prefix-tuning")
        assert excinfo.value.code is ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN

    def test_the_refusal_names_the_value_and_the_alternatives(self) -> None:
        """A caller who typo'd needs to see both halves to fix it themselves."""
        with pytest.raises(AppError) as excinfo:
            require_strategy_name("lorra")
        message = str(excinfo.value)
        assert "'lorra'" in message
        assert "cartridge, full, lora, qlora" in message

    def test_the_empty_string_is_refused_like_any_other_unknown(self) -> None:
        """An absent value must not arrive here as "" and be treated as a default."""
        with pytest.raises(AppError) as excinfo:
            require_strategy_name("")
        assert excinfo.value.code is ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN

    def test_a_name_differing_only_in_case_is_refused(self) -> None:
        """The names are identifiers on the wire, not human-facing labels."""
        with pytest.raises(AppError) as excinfo:
            require_strategy_name("LoRA")
        assert excinfo.value.code is ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN
