"""Tests for the require helper and @enforce_contract decorator."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import ContractError, NoUnsourcedFactError
from tankpit_bot.contracts.enforcement import enforce_contract, require


def test_require_passes_silently_when_condition_holds() -> None:
    """require returns None and raises nothing on a true condition."""
    require(True, NoUnsourcedFactError, field="source")


def test_require_raises_specific_error_with_details() -> None:
    """require raises the given error class with the given details."""
    with pytest.raises(NoUnsourcedFactError) as exc:
        require(False, NoUnsourcedFactError, field="source", got="None")
    assert exc.value.details == {"field": "source", "got": "None"}
    assert exc.value.contract_name == "no_unsourced_fact"


def test_require_records_caller_location() -> None:
    """require records this test file as the violation site."""
    with pytest.raises(NoUnsourcedFactError) as exc:
        require(False, NoUnsourcedFactError)
    assert exc.value.violated_at.startswith("test_enforcement.py:")


class _RecordingContract:
    """Contract that records the arguments it was checked with."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    @property
    def name(self) -> str:
        return "recording"

    def check(self, *, left: int, right: int) -> None:
        self.seen.append((left, right))


class _RejectingContract:
    """Contract that always raises."""

    @property
    def name(self) -> str:
        return "rejecting"

    def check(self, *, left: int, right: int) -> None:
        require(False, ContractError, reason="always rejects")


def test_enforce_contract_runs_check_before_the_function() -> None:
    """Decorated function sees the contract check run on its arguments."""
    contract = _RecordingContract()

    @enforce_contract(contract)
    def apply_example(*, left: int, right: int) -> int:
        """Add two numbers."""
        return left + right

    result = apply_example(left=2, right=3)
    assert result == 5
    assert contract.seen == [(2, 3)]


def test_enforce_contract_blocks_the_call_on_violation() -> None:
    """A raising contract prevents the function body from running."""
    calls: list[str] = []

    @enforce_contract(_RejectingContract())
    def apply_example(*, left: int, right: int) -> int:
        calls.append("ran")
        return left + right

    with pytest.raises(ContractError) as exc:
        apply_example(left=1, right=2)
    assert calls == []
    assert exc.value.details == {"reason": "always rejects"}


def test_enforce_contract_preserves_name_and_doc() -> None:
    """The enforced wrapper keeps the function's name and docstring."""
    contract = _RecordingContract()

    @enforce_contract(contract)
    def apply_example(*, left: int, right: int) -> int:
        """Docstring survives."""
        return left + right

    assert apply_example.__name__ == "apply_example"
    assert apply_example.__doc__ == "Docstring survives."


def test_contract_names_are_exposed() -> None:
    """Concrete contracts report their names via the protocol property."""
    assert _RecordingContract().name == "recording"
    assert _RejectingContract().name == "rejecting"
