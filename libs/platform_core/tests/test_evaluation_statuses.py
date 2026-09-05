"""The one narrowing from a string to a covenant evaluation status.

Three paths reach it: the covenant metrics event decoder here, and the
streaming evaluation decoder and Google AI response reader in
covenant-radar-api. Each had its own copy before, and one of the three raised
ValueError where the other two raised JSONTypeError -- so a service disagreed
with itself about the type of its own refusal depending on which decoder read
the payload. Tested directly rather than only through whichever decoder
happens to call it.
"""

from __future__ import annotations

import pytest

from platform_core.evaluation_statuses import (
    EVALUATION_STATUSES,
    as_evaluation_status,
    require_evaluation_status,
)
from platform_core.json_utils import JSONObject, JSONTypeError


class TestTheDeclaredSet:
    def test_it_names_the_three_statuses(self) -> None:
        assert EVALUATION_STATUSES == ("OK", "BREACH", "WARNING")

    def test_every_declared_status_survives_the_narrowing(self) -> None:
        """The tuple and the chain are two constructs holding one set. This is
        what keeps them honest: a status added to the tuple and not to the
        chain fails here rather than at a caller."""
        narrowed = [as_evaluation_status(s, "status") for s in EVALUATION_STATUSES]

        assert narrowed == list(EVALUATION_STATUSES)

    def test_it_is_not_the_covenant_domain_status_set(self) -> None:
        """covenant_domain grades a SINGLE covenant result OK/BREACH/NEAR_BREACH.
        This grades a whole period's evaluation. Two three-member status sets
        in one domain, neither a superset of the other, and both spelled on a
        field called `status` -- which is why the guard cannot watch that name.
        """
        assert "NEAR_BREACH" not in EVALUATION_STATUSES

        with pytest.raises(JSONTypeError, match="got 'NEAR_BREACH'"):
            as_evaluation_status("NEAR_BREACH", "status")


class TestNarrowingAString:
    @pytest.mark.parametrize("status", ["OK", "BREACH", "WARNING"])
    def test_each_status_is_accepted(self, status: str) -> None:
        assert as_evaluation_status(status, "evaluation_status") == status

    def test_an_undeclared_status_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="got 'PENDING'"):
            as_evaluation_status("PENDING", "evaluation_status")

    def test_the_refusal_names_the_field_and_the_accepted_set(self) -> None:
        with pytest.raises(JSONTypeError) as raised:
            as_evaluation_status("PENDING", "period_status")

        message = str(raised.value)
        assert "period_status" in message
        assert "OK, BREACH, WARNING" in message

    def test_matching_is_case_sensitive(self) -> None:
        """These are wire values, not user input; accepting "ok" would mean the
        encoder and decoder disagree about the payload."""
        with pytest.raises(JSONTypeError, match="got 'ok'"):
            as_evaluation_status("ok", "evaluation_status")

    def test_an_empty_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="got ''"):
            as_evaluation_status("", "evaluation_status")


class TestReadingAField:
    def test_it_reads_and_narrows(self) -> None:
        payload: JSONObject = {"status": "BREACH"}

        assert require_evaluation_status(payload, "status") == "BREACH"

    def test_a_missing_field_is_refused(self) -> None:
        payload: JSONObject = {}

        with pytest.raises(JSONTypeError):
            require_evaluation_status(payload, "status")

    def test_a_non_string_field_is_refused(self) -> None:
        payload: JSONObject = {"status": 1}

        with pytest.raises(JSONTypeError):
            require_evaluation_status(payload, "status")

    def test_an_undeclared_status_in_a_field_is_refused(self) -> None:
        payload: JSONObject = {"status": "PENDING"}

        with pytest.raises(JSONTypeError, match="Field 'status' must be one of"):
            require_evaluation_status(payload, "status")


__all__ = ["TestNarrowingAString", "TestReadingAField", "TestTheDeclaredSet"]
