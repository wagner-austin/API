"""Tests for the shared command-line parsing."""

from __future__ import annotations

import pytest
from scripts.arguments import (
    DEFAULT_DEVICE,
    ScriptArgumentError,
    require_count,
    require_positive_count,
    split_device,
)


class TestSplitDevice:
    """Tests for :func:`split_device`."""

    def test_defaults_when_the_flag_is_absent(self) -> None:
        """Every command line documented before the flag existed still parses."""
        assert split_device(["RUN_TO_RUN", "cache"]) == (DEFAULT_DEVICE, ["RUN_TO_RUN", "cache"])

    def test_removes_the_flag_and_its_value_from_the_middle(self) -> None:
        """The positionals either side close up around it."""
        assert split_device(["a", "--device", "cuda:1", "b"]) == ("cuda:1", ["a", "b"])

    def test_accepts_the_flag_first(self) -> None:
        """Flag-first is the form a variadic trailing list forces."""
        assert split_device(["--device", "cuda:1", "a", "b"]) == ("cuda:1", ["a", "b"])

    def test_accepts_the_flag_last(self) -> None:
        """Flag-last is the form a reader appends without counting positions."""
        assert split_device(["a", "b", "--device", "cuda:1"]) == ("cuda:1", ["a", "b"])

    def test_rejects_a_dangling_flag(self) -> None:
        """A flag with nothing after it must not silently mean the default.

        Defaulting here would run a whole sweep on ``cuda:0`` after the
        operator asked for another card, and label the record accordingly.
        """
        with pytest.raises(ScriptArgumentError) as caught:
            split_device(["a", "b", "--device"])
        assert caught.value.code == "NP-ARGS-002"

    def test_does_not_mutate_the_argument_list(self) -> None:
        """The caller's list is untouched, so a retry sees what it passed."""
        args = ["a", "--device", "cuda:1", "b"]
        split_device(args)
        assert args == ["a", "--device", "cuda:1", "b"]


class TestRequireCount:
    """Tests for :func:`require_count`."""

    def test_accepts_zero(self) -> None:
        """Zero means Warp's own code-generated bound, not an absent value."""
        assert require_count("0", "MAX_RECORDS") == 0

    def test_accepts_a_positive_count(self) -> None:
        """The bound that cleared the 32-body overflow parses."""
        assert require_count("64", "MAX_RECORDS") == 64

    def test_rejects_a_negative_count(self) -> None:
        """A leading minus is not a digit string."""
        with pytest.raises(ScriptArgumentError) as caught:
            require_count("-1", "MAX_RECORDS")
        assert caught.value.code == "NP-ARGS-003"

    def test_rejects_text(self) -> None:
        """A mistyped flag landing in a numeric slot is caught here."""
        with pytest.raises(ScriptArgumentError) as caught:
            require_count("--device", "MAX_RECORDS")
        assert caught.value.code == "NP-ARGS-003"

    def test_names_the_argument_it_rejected(self) -> None:
        """The message says which argument was wrong, not just that one was."""
        with pytest.raises(ScriptArgumentError) as caught:
            require_count("x", "CAPACITY")
        assert "CAPACITY" in caught.value.message


class TestRequirePositiveCount:
    """Tests for :func:`require_positive_count`."""

    def test_accepts_one(self) -> None:
        """A single world is a legitimate rung."""
        assert require_positive_count("1", "WORLDS[0]") == 1

    def test_rejects_zero(self) -> None:
        """A ladder rung of zero worlds simulates nothing."""
        with pytest.raises(ScriptArgumentError) as caught:
            require_positive_count("0", "WORLDS[0]")
        assert caught.value.code == "NP-ARGS-004"

    def test_rejects_text_through_the_shared_check(self) -> None:
        """The digit check runs first, so text reports the count code."""
        with pytest.raises(ScriptArgumentError) as caught:
            require_positive_count("many", "CAPACITY")
        assert caught.value.code == "NP-ARGS-003"
