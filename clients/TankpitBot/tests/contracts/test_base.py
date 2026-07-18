"""Tests for the contract error hierarchy."""

from __future__ import annotations

import pytest

from tankpit_bot.contracts.base import (
    ConfidenceOutOfBoundsError,
    ContractError,
    NoUnsourcedFactError,
    ProvenanceRootednessError,
)


def test_contract_error_message_includes_name_site_and_details() -> None:
    """Error message renders contract name, site, and sorted details."""
    error = ContractError(violated_at="module.py:12", details={"b": "2", "a": "1"})
    assert str(error) == "contract violated at module.py:12: a=1, b=2"


def test_contract_error_violation_record() -> None:
    """violation() returns the structured record."""
    error = ContractError(violated_at="module.py:12", details={"key": "value"})
    assert error.violation() == {
        "contract_name": "contract",
        "violated_at": "module.py:12",
        "details": {"key": "value"},
    }


@pytest.mark.parametrize(
    ("error_class", "expected_name"),
    [
        (NoUnsourcedFactError, "no_unsourced_fact"),
        (ConfidenceOutOfBoundsError, "confidence_in_bounds"),
        (ProvenanceRootednessError, "provenance_rootedness"),
    ],
)
def test_specific_contracts_name_themselves(
    error_class: type[ContractError], expected_name: str
) -> None:
    """Each specific contract error carries its own contract name."""
    error = error_class(violated_at="facts.py:1", details={})
    assert error.contract_name == expected_name
    assert error.violation()["contract_name"] == expected_name
