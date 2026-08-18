"""Tests for the shared failure base.

Every module's error type derives from :class:`navprobe.NavProbeError`, and
every test in this suite asserts on ``.code``. That makes the code attribute and
its message format load-bearing across the whole package, so they are pinned
here rather than assumed everywhere.
"""

from __future__ import annotations

import pytest

from navprobe import NavProbeError
from navprobe.canonical import CanonicalEncodingError
from navprobe.comparison import ComparisonError
from navprobe.experiment import TrialError
from navprobe.rollout import RolloutError
from navprobe.wireformat import WireFormatError

#: Every concrete failure type in the package, with the code prefix it owns.
#: A new error type absent from this table has no prefix reserved for it, which
#: is how two areas end up sharing one code range.
ERROR_TYPES: tuple[tuple[type[NavProbeError], str], ...] = (
    (CanonicalEncodingError, "NP-CANON-"),
    (WireFormatError, "NP-WIRE-"),
    (RolloutError, "NP-ROLLOUT-"),
    (ComparisonError, "NP-COMPARE-"),
    (TrialError, "NP-TRIAL-"),
)


class TestNavProbeError:
    """Tests for the base failure type."""

    def test_carries_the_code_it_was_given(self) -> None:
        """The code is readable off the exception, which is what tests branch on."""
        assert NavProbeError("NP-TEST-001", "something went wrong").code == "NP-TEST-001"

    def test_carries_the_message_it_was_given(self) -> None:
        """The human-readable message is preserved separately from the code."""
        error = NavProbeError("NP-TEST-001", "something went wrong")
        assert error.message == "something went wrong"

    def test_string_form_leads_with_the_bracketed_code(self) -> None:
        """A logged failure names its code without the reader parsing anything."""
        assert str(NavProbeError("NP-TEST-001", "boom")) == "[NP-TEST-001] boom"

    def test_is_raisable_and_catchable_as_itself(self) -> None:
        """The base is a real exception, not a wrapper around one."""
        with pytest.raises(NavProbeError) as caught:
            raise NavProbeError("NP-TEST-002", "boom")
        assert caught.value.code == "NP-TEST-002"


class TestErrorTypes:
    """Every concrete failure type shares the base's contract."""

    @staticmethod
    def test_all_derive_from_the_shared_base() -> None:
        """A caller can catch every failure this package raises with one type."""
        assert [issubclass(error_type, NavProbeError) for error_type, _ in ERROR_TYPES] == [
            True,
            True,
            True,
            True,
            True,
        ]

    @staticmethod
    def test_each_owns_a_distinct_code_prefix() -> None:
        """No two areas share a code range, so a code names exactly one area."""
        prefixes = [prefix for _, prefix in ERROR_TYPES]
        assert sorted(prefixes) == sorted(set(prefixes))

    @staticmethod
    def test_each_preserves_the_base_constructor() -> None:
        """A subclass that overrode the constructor would break every assertion."""
        raised = [error_type(f"{prefix}999", "boom").code for error_type, prefix in ERROR_TYPES]
        assert raised == [
            "NP-CANON-999",
            "NP-WIRE-999",
            "NP-ROLLOUT-999",
            "NP-COMPARE-999",
            "NP-TRIAL-999",
        ]
