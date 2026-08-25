"""Tests for command-line argument reading."""

from __future__ import annotations

import pytest

from platform_core.cli_args import parse_single_flags, require_flag, take_value

_FLAGS = ("--host", "--spec")


class TestTakeValue:
    def test_it_reads_the_next_token(self) -> None:
        assert take_value(["--host", "hpc3"], 1, "--host") == "hpc3"

    def test_a_missing_value_at_the_end_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a value"):
            take_value(["--host"], 1, "--host")

    def test_a_following_flag_is_not_a_value(self) -> None:
        """--host --verbose would otherwise bind '--verbose' as a hostname."""
        with pytest.raises(ValueError, match="got the flag"):
            take_value(["--host", "--verbose"], 1, "--host")


class TestParseSingleFlags:
    def test_it_reads_every_flag(self) -> None:
        parsed = parse_single_flags(["--host", "hpc3", "--spec", "s.json"], _FLAGS)
        assert parsed == {"--host": "hpc3", "--spec": "s.json"}

    def test_no_arguments_yields_no_flags(self) -> None:
        assert parse_single_flags([], _FLAGS) == {}

    def test_an_absent_flag_is_absent_not_empty(self) -> None:
        assert "--spec" not in parse_single_flags(["--host", "hpc3"], _FLAGS)

    def test_an_unknown_flag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown argument"):
            parse_single_flags(["--turbo", "yes"], _FLAGS)

    def test_a_bare_positional_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown argument"):
            parse_single_flags(["hpc3"], _FLAGS)

    def test_a_repeated_flag_is_refused(self) -> None:
        """Keeping the last would silently discard a value the caller typed."""
        with pytest.raises(ValueError, match="more than once"):
            parse_single_flags(["--host", "a", "--host", "b"], _FLAGS)

    def test_a_missing_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a value"):
            parse_single_flags(["--host"], _FLAGS)


class TestRequireFlag:
    def test_it_returns_a_present_value(self) -> None:
        assert require_flag({"--host": "hpc3"}, "--host") == "hpc3"

    def test_an_absent_flag_is_refused(self) -> None:
        """No default: a defaulted host sends work somewhere unnamed."""
        with pytest.raises(ValueError, match="--host is required"):
            require_flag({}, "--host")
