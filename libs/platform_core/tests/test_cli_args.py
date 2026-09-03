"""Tests for command-line argument reading."""

from __future__ import annotations

import pytest

from platform_core.cli_args import (
    HELP_FLAGS,
    HelpRequestedError,
    parse_single_flags,
    require_flag,
    take_value,
    usage_text,
)

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


class TestHelp:
    """Asking what the flags are is the one thing a caller who does not know
    them can type, so it must not be answered with 'unknown argument'.
    """

    @pytest.mark.parametrize("flag", HELP_FLAGS)
    def test_a_help_flag_is_recognised(self, flag: str) -> None:
        with pytest.raises(HelpRequestedError) as raised:
            parse_single_flags([flag], _FLAGS)

        assert raised.value.allowed == _FLAGS

    def test_help_wins_over_an_unknown_flag_beside_it(self) -> None:
        """`cmd --turbo yes --help` is a caller who already guessed wrong."""
        with pytest.raises(HelpRequestedError):
            parse_single_flags(["--help", "--turbo", "yes"], _FLAGS)

    def test_help_after_a_valid_flag_is_still_help(self) -> None:
        with pytest.raises(HelpRequestedError):
            parse_single_flags(["--host", "hpc3", "--help"], _FLAGS)

    def test_it_is_a_value_error_so_untaught_boundaries_still_refuse(self) -> None:
        """The compatibility property. A boundary that has not been taught
        about help must degrade to the refusal it gave before, not to a
        traceback.
        """
        assert issubclass(HelpRequestedError, ValueError)

        with pytest.raises(ValueError, match="expected one of"):
            parse_single_flags(["--help"], _FLAGS)

    def test_usage_names_every_flag_and_that_each_takes_a_value(self) -> None:
        assert usage_text(_FLAGS) == "usage: --host <value> --spec <value>"


class TestRequireFlag:
    def test_it_returns_a_present_value(self) -> None:
        assert require_flag({"--host": "hpc3"}, "--host") == "hpc3"

    def test_an_absent_flag_is_refused(self) -> None:
        """No default: a defaulted host sends work somewhere unnamed."""
        with pytest.raises(ValueError, match="--host is required"):
            require_flag({}, "--host")
