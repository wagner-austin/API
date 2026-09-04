"""The one narrowing from a string to a risk tier.

Three packages reach this: the covenant metrics event decoder here, the
streaming prediction decoder and the Google AI response reader in
covenant-radar-api. Each had its own copy before, so this is the only place
that decides what a tier is, and it is tested directly rather than only
through whichever decoder happens to call it.
"""

from __future__ import annotations

import pytest

from platform_core.json_utils import JSONObject, JSONTypeError
from platform_core.risk_tiers import RISK_TIERS, as_risk_tier, require_risk_tier


class TestTheDeclaredSet:
    def test_it_names_the_four_tiers_in_ascending_order(self) -> None:
        assert RISK_TIERS == ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_every_declared_tier_survives_the_narrowing(self) -> None:
        """The tuple and the chain are two constructs holding one set, for the
        reasons the module docstring gives. This is what keeps them honest: a
        tier added to the tuple and not to the chain fails here rather than at
        a caller."""
        assert [as_risk_tier(tier, "risk_tier") for tier in RISK_TIERS] == list(RISK_TIERS)


class TestNarrowingAString:
    @pytest.mark.parametrize("tier", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    def test_each_tier_is_accepted(self, tier: str) -> None:
        assert as_risk_tier(tier, "risk_tier") == tier

    def test_an_undeclared_tier_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="got 'SEVERE'"):
            as_risk_tier("SEVERE", "risk_tier")

    def test_the_refusal_names_the_field_and_the_accepted_set(self) -> None:
        """A decoder reads several fields; "invalid risk tier" alone left the
        reader to find both which field and what would have been accepted."""
        with pytest.raises(JSONTypeError) as raised:
            as_risk_tier("SEVERE", "predicted_tier")

        message = str(raised.value)
        assert "predicted_tier" in message
        assert "LOW, MEDIUM, HIGH, CRITICAL" in message

    def test_matching_is_case_sensitive(self) -> None:
        """The tiers are wire values, not user input; accepting "low" here
        would mean the encoder and decoder disagree about the payload."""
        with pytest.raises(JSONTypeError, match="got 'low'"):
            as_risk_tier("low", "risk_tier")

    def test_an_empty_string_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="got ''"):
            as_risk_tier("", "risk_tier")


class TestReadingAField:
    def test_it_reads_and_narrows(self) -> None:
        payload: JSONObject = {"risk_tier": "HIGH"}

        assert require_risk_tier(payload, "risk_tier") == "HIGH"

    def test_a_missing_field_is_refused(self) -> None:
        payload: JSONObject = {}

        with pytest.raises(JSONTypeError):
            require_risk_tier(payload, "risk_tier")

    def test_a_non_string_field_is_refused(self) -> None:
        payload: JSONObject = {"risk_tier": 3}

        with pytest.raises(JSONTypeError):
            require_risk_tier(payload, "risk_tier")

    def test_an_undeclared_tier_in_a_field_is_refused(self) -> None:
        payload: JSONObject = {"risk_tier": "SEVERE"}

        with pytest.raises(JSONTypeError, match="Field 'risk_tier' must be one of"):
            require_risk_tier(payload, "risk_tier")


__all__ = ["TestNarrowingAString", "TestReadingAField", "TestTheDeclaredSet"]
