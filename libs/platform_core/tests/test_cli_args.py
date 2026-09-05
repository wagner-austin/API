"""Tests for command-line argument reading."""

from __future__ import annotations

import argparse

import pytest

from platform_core.cli_args import (
    HELP_FLAGS,
    HelpRequestedError,
    namespace_bool,
    namespace_int,
    namespace_str,
    namespace_str_or_none,
    namespace_str_tuple,
    parse_single_flags,
    require_flag,
    run_subcommand_cli,
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

    def test_it_is_not_a_value_error_because_it_is_not_a_bad_command_line(self) -> None:
        """The other raises here mean the command line is wrong. This one means
        it is a question, and a boundary that cannot tell them apart prints a
        refusal at a non-zero status for a caller who typed something valid.
        """
        assert not issubclass(HelpRequestedError, ValueError)

        with pytest.raises(HelpRequestedError):
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


class TestNamespaceReaders:
    """Reading an argparse namespace without the silence it used to carry.

    Two libraries had a private copy of these -- `platform_calendar` and
    `platform_email` -- with annotations that had already drifted apart
    (`str | None` against `str | int | bool | None`) and the same function
    living under two names. Both returned the DEFAULT when the attribute was
    present with the wrong type, and both carried a test pinning that.

    Absent and wrong-typed are different events. The first is a caller
    declining a flag. The second is a parser declaration contradicting the
    reader that consumes it, which no run can be correct under.
    """

    def test_a_declared_string_is_read(self) -> None:
        assert namespace_str(argparse.Namespace(folder="inbox"), "folder", "sent") == "inbox"

    def test_an_absent_flag_takes_the_default(self) -> None:
        assert namespace_str(argparse.Namespace(), "folder", "inbox") == "inbox"

    def test_a_string_flag_holding_a_number_is_refused(self) -> None:
        """The case both copies answered with the default, and both pinned
        with a test. Nothing in a correct parser reaches it."""
        with pytest.raises(ValueError, match="parsed as int, expected str"):
            namespace_str(argparse.Namespace(folder=123), "folder", "inbox")

    def test_the_refusal_names_the_flag_as_it_was_typed(self) -> None:
        """The cause is a declaration in another function, so the message has
        to be greppable from the command line the operator ran."""
        with pytest.raises(ValueError, match=r"--dry-run"):
            namespace_str(argparse.Namespace(dry_run=1), "dry_run", "")

    def test_a_declared_but_unsupplied_string_takes_the_default(self) -> None:
        """argparse sets a declared option it did not receive to None rather
        than leaving the attribute off. Reading that as a type error is what
        the first run of this caught: `main` with no subcommand reported
        "--command parsed as NoneType", which is argparse's normal encoding of
        absence, not a defect."""
        assert namespace_str(argparse.Namespace(folder=None), "folder", "inbox") == "inbox"

    def test_a_declared_but_unsupplied_integer_takes_the_default(self) -> None:
        assert namespace_int(argparse.Namespace(count=None), "count", 10) == 10

    def test_a_declared_but_unsupplied_boolean_takes_the_default(self) -> None:
        assert namespace_bool(argparse.Namespace(force=None), "force", True) is True

    def test_an_optional_string_reads_none_when_absent(self) -> None:
        assert namespace_str_or_none(argparse.Namespace(), "query") is None

    def test_an_optional_string_is_read_when_present(self) -> None:
        assert namespace_str_or_none(argparse.Namespace(query="a"), "query") == "a"

    def test_an_optional_string_holding_a_list_is_refused(self) -> None:
        given: list[str] = ["a"]

        with pytest.raises(ValueError, match="expected str or None"):
            namespace_str_or_none(argparse.Namespace(query=given), "query")

    def test_an_integer_is_read(self) -> None:
        assert namespace_int(argparse.Namespace(count=42), "count", 10) == 42

    def test_an_absent_integer_takes_the_default(self) -> None:
        assert namespace_int(argparse.Namespace(), "count", 10) == 10

    def test_a_boolean_is_not_accepted_as_an_integer(self) -> None:
        """`bool` is an `int` to Python, so a `store_true` flag read as a
        count would silently become 1 and the command would run once."""
        with pytest.raises(ValueError, match="parsed as bool, expected int"):
            namespace_int(argparse.Namespace(count=True), "count", 10)

    def test_an_integer_flag_holding_a_string_is_refused(self) -> None:
        with pytest.raises(ValueError, match="parsed as str, expected int"):
            namespace_int(argparse.Namespace(count="10"), "count", 10)

    def test_a_boolean_is_read(self) -> None:
        assert namespace_bool(argparse.Namespace(force=True), "force", False) is True

    def test_an_absent_boolean_takes_the_default(self) -> None:
        assert namespace_bool(argparse.Namespace(), "force", True) is True

    def test_a_boolean_flag_holding_an_integer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="parsed as int, expected bool"):
            namespace_bool(argparse.Namespace(force=1), "force", False)

    def test_a_repeatable_flag_reads_every_value(self) -> None:
        given: list[str] = ["a.txt", "b.txt"]
        namespace = argparse.Namespace(attach=given)

        assert namespace_str_tuple(namespace, "attach") == ("a.txt", "b.txt")

    def test_an_absent_repeatable_flag_is_empty(self) -> None:
        assert namespace_str_tuple(argparse.Namespace(), "attach") == ()

    def test_a_repeatable_flag_holding_one_bad_element_is_refused(self) -> None:
        """The version this replaces dropped offending elements silently, so a
        mistyped attachment shortened the list and the mail went without it."""
        given: list[str | int] = ["a.txt", 7]

        with pytest.raises(ValueError, match="parsed as int, expected str"):
            namespace_str_tuple(argparse.Namespace(attach=given), "attach")

    def test_a_repeatable_flag_holding_a_bare_string_is_refused(self) -> None:
        """`--attach a.txt` declared without `action="append"` arrives as a
        string, and iterating it would attach one file per character."""
        with pytest.raises(ValueError, match="expected list of str"):
            namespace_str_tuple(argparse.Namespace(attach="a.txt"), "attach")


class TestRunSubcommandCli:
    """The four lines two entry points each spelled for themselves."""

    @staticmethod
    def _build() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="demo")
        subparsers = parser.add_subparsers(dest="command")
        listing = subparsers.add_parser("list", aliases=["ls"])
        listing.add_argument("-n", "--count", type=int, default=10)
        return parser

    def test_the_named_subcommand_reaches_the_dispatcher(self) -> None:
        seen: list[str] = []

        run_subcommand_cli(
            ["list"],
            build_parser=self._build,
            dispatch=lambda command, _args: seen.append(command),
        )

        assert seen == ["list"]

    def test_an_alias_arrives_as_itself_not_as_its_target(self) -> None:
        """Both commands dispatch on membership in a tuple of aliases, so the
        alias must not be resolved on the way in."""
        seen: list[str] = []

        run_subcommand_cli(
            ["ls"],
            build_parser=self._build,
            dispatch=lambda command, _args: seen.append(command),
        )

        assert seen == ["ls"]

    def test_no_subcommand_at_all_dispatches_the_empty_string(self) -> None:
        """argparse leaves `command` None, and both commands rely on reaching
        their default view rather than crashing on the attribute."""
        seen: list[str] = []

        run_subcommand_cli(
            [],
            build_parser=self._build,
            dispatch=lambda command, _args: seen.append(command),
        )

        assert seen == [""]

    def test_the_parsed_namespace_reaches_the_dispatcher_too(self) -> None:
        """The dispatcher decodes the subcommand's own flags out of it."""
        counts: list[int] = []

        run_subcommand_cli(
            ["list", "-n", "3"],
            build_parser=self._build,
            dispatch=lambda _command, args: counts.append(namespace_int(args, "count", 10)),
        )

        assert counts == [3]

    def test_the_parser_is_built_once(self) -> None:
        builds: list[argparse.ArgumentParser] = []

        def _counting_build() -> argparse.ArgumentParser:
            parser = self._build()
            builds.append(parser)
            return parser

        run_subcommand_cli(["list"], build_parser=_counting_build, dispatch=lambda _c, _a: None)

        assert len(builds) == 1

    def test_an_unknown_subcommand_is_refused_by_argparse(self) -> None:
        """The dispatcher never sees it: a mistyped subcommand exits rather
        than falling through to whichever branch the dispatcher ends on."""
        with pytest.raises(SystemExit):
            run_subcommand_cli(
                ["lst"],
                build_parser=self._build,
                dispatch=lambda _c, _a: None,
            )
