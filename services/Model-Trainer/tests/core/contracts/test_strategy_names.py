"""The declared strategy names, and the one function that admits a string.

These hold the two halves of ``strategy_names`` in agreement. The module has
to state its names twice -- once as a ``Literal`` for the type checker and
once as a tuple for anything that iterates -- because ``typing.get_args`` is
unavailable under this package's ``disallow_any_expr`` and mypy will not
narrow through membership in a variadic tuple. Nothing in the language holds
those two in step, so these tests do.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.strategy_names import (
    STRATEGY_NAMES,
    require_strategy_name,
)


class TestTheDeclaredNames:
    """The tuple is the iterable face of the literal, and must match it."""

    def test_every_declared_name_survives_a_round_trip(self) -> None:
        """Walks the tuple and requires the narrowing chain to accept each.

        This is the test that catches the half-edit: a strategy added to
        ``STRATEGY_NAMES`` but not to ``require_strategy_name`` fails here,
        rather than at the first request that names it.
        """
        assert [require_strategy_name(name) for name in STRATEGY_NAMES] == list(STRATEGY_NAMES)

    def test_the_declared_names_are_exactly_these(self) -> None:
        """Pins the set, so adding one is a deliberate edit to this line.

        Asserted as a sorted list rather than a membership check, so a name
        that is REMOVED fails too.
        """
        assert sorted(STRATEGY_NAMES) == ["full", "lora", "qlora"]

    def test_no_name_is_declared_twice(self) -> None:
        """A duplicate would make the registry's last registration win silently."""
        assert len(set(STRATEGY_NAMES)) == len(STRATEGY_NAMES)


class TestRequireStrategyName:
    """The single door from an untrusted string to the typed name."""

    def test_an_undeclared_name_is_refused_with_its_own_code(self) -> None:
        """Not a generic decode error: the caller picked a value that does not exist."""
        with pytest.raises(AppError) as excinfo:
            require_strategy_name("cartridge")
        assert excinfo.value.code is ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN

    def test_the_refusal_names_the_value_and_the_alternatives(self) -> None:
        """A caller who typo'd needs to see both halves to fix it themselves."""
        with pytest.raises(AppError) as excinfo:
            require_strategy_name("lorra")
        message = str(excinfo.value)
        assert "'lorra'" in message
        assert "full, lora, qlora" in message

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
