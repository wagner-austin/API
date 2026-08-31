"""The command line one headless match is started by.

A sweep composes this per job, so these are the launch surface the whole
harness goes through. Most of what is checked here is refusal: a match played
under a mistyped flag is a DIFFERENT match and would still file a scorecard.
"""

from __future__ import annotations

import runpy

import pytest

from rw_bot.harness.launch import CATALOGUE, FROZEN_CATALOGUE, FROZEN_TYPE_DUMP, TYPE_DUMP
from rw_bot.harness.play_match_cli import (
    ALLOWED_FLAGS,
    EXIT_HELP,
    FLAG_HELP,
    NUMERIC_FLAGS,
    OPTIONAL_FLAGS,
    REQUIRED_FLAGS,
    LaunchCommandError,
    decode_launch,
    main,
    render_usage,
)
from rw_bot.validation import DecodeError
from tests.harness_fakes import FakeHost

MINIMAL = ("--port", "27511", "--game-dir", ".game-w1", "--play-log", "runs/a.log")


class TestReadingALaunch:
    def test_the_minimal_command_is_the_three_that_cannot_be_guessed(self) -> None:
        config = decode_launch(MINIMAL)
        assert (config["port"], config["game_dir"], config["play_log"]) == (
            27511,
            ".game-w1",
            "runs/a.log",
        )

    def test_everything_else_defaults_to_off(self) -> None:
        config = decode_launch(MINIMAL)
        assert (config["seed"], config["lockstep"], config["pin_delta"]) == (0, 0, 0)
        assert (config["map"], config["tree"], config["extra_agent_args"]) == ("", "", "")

    def test_the_artifacts_default_to_the_modules_own_constants(self) -> None:
        """One owner: the Makefile no longer carries its own copy of either."""
        config = decode_launch(MINIMAL)
        assert config["catalogue"] == CATALOGUE
        assert config["type_dump"] == TYPE_DUMP

    def test_a_tree_re_roots_both_registry_dumps(self) -> None:
        """A frozen tree carries both dumps under fixed names, and a launch
        against one must read THOSE -- the repository-relative defaults do
        not exist where a frozen tree is the only code present. This is the
        rule the sweep composes per member, on the direct path; job 55663569
        died on the difference."""
        config = decode_launch((*MINIMAL, "--tree", "/pub/wagnera3/rusted/payload"))
        assert config["catalogue"] == f"/pub/wagnera3/rusted/payload/{FROZEN_CATALOGUE}"
        assert config["type_dump"] == f"/pub/wagnera3/rusted/payload/{FROZEN_TYPE_DUMP}"

    def test_an_explicit_dump_beats_the_tree_derived_one(self) -> None:
        config = decode_launch(
            (
                *MINIMAL,
                "--tree",
                "/pub/wagnera3/rusted/payload",
                "--catalogue",
                "probe/printunits.log",
                "--type-dump",
                "probe/type-flags.ndjson",
            )
        )
        assert config["catalogue"] == "probe/printunits.log"
        assert config["type_dump"] == "probe/type-flags.ndjson"

    def test_a_mistyped_flag_is_refused_and_the_valid_ones_named(self) -> None:
        """Ignoring it would play a different match and still file a card."""
        with pytest.raises(ValueError, match="unknown argument"):
            decode_launch((*MINIMAL, "--fastforward", "10"))

    @pytest.mark.parametrize("missing", REQUIRED_FLAGS)
    def test_each_required_flag_is_required(self, missing: str) -> None:
        tokens = list(MINIMAL)
        index = tokens.index(missing)
        del tokens[index : index + 2]
        with pytest.raises(ValueError, match=f"{missing} is required"):
            decode_launch(tokens)

    def test_a_flag_given_twice_is_refused_rather_than_last_wins(self) -> None:
        with pytest.raises(ValueError, match="given more than once"):
            decode_launch((*MINIMAL, "--seed", "1", "--seed", "2"))

    def test_a_flag_with_no_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a value"):
            decode_launch((*MINIMAL, "--seed"))

    def test_a_non_numeric_value_names_the_flag_that_carried_it(self) -> None:
        """int() alone reports only the text it choked on, which across nine
        numeric flags leaves the caller to guess which one they mistyped."""
        with pytest.raises(LaunchCommandError) as caught:
            decode_launch((*MINIMAL, "--seed", "soon"))
        assert caught.value.code == "RW-LAUNCH-001"
        assert "--seed must be a whole number, got 'soon'" in caught.value.message

    def test_a_negative_seed_survives_because_the_engine_takes_one(self) -> None:
        assert decode_launch((*MINIMAL, "--seed", "-3"))["seed"] == -3

    def test_a_port_of_zero_is_refused_by_the_shared_decoder(self) -> None:
        """Zero is what the old recipe meant by "draw one at random", and
        there is no draw left: an invented port collides with a live match's
        lease. Refused by the SAME decoder a sweep's launch goes through, so
        neither caller can start a match the other could not."""
        with pytest.raises(DecodeError) as caught:
            decode_launch(("--port", "0", "--game-dir", ".g", "--play-log", "l"))
        assert caught.value.code == "RW-DECODE-004"

    def test_a_blank_game_directory_is_refused(self) -> None:
        with pytest.raises(DecodeError) as caught:
            decode_launch(("--port", "1", "--game-dir", "  ", "--play-log", "l"))
        assert caught.value.code == "RW-DECODE-003"

    def test_every_flag_reaches_the_field_it_names(self) -> None:
        """The field name is derived from the flag rather than tabulated, so
        this is what proves the derivation covers all eighteen."""
        config = decode_launch(
            (
                *MINIMAL,
                "--pin-delta",
                "3",
                "--fast-forward",
                "10",
                "--rng-tap",
                "1",
                "--extra-agent-args",
                "x=1",
                "--type-dump",
                "t.ndjson",
            )
        )
        assert config["pin_delta"] == 3
        assert config["fast_forward"] == 10
        assert config["rng_tap"] == 1
        assert config["extra_agent_args"] == "x=1"
        assert config["type_dump"] == "t.ndjson"


class TestTheFlagTables:
    def test_every_flag_is_either_required_or_defaulted(self) -> None:
        assert set(ALLOWED_FLAGS) == set(REQUIRED_FLAGS) | set(OPTIONAL_FLAGS)

    def test_no_flag_is_both(self) -> None:
        assert set(REQUIRED_FLAGS).isdisjoint(OPTIONAL_FLAGS)

    def test_every_numeric_flag_is_a_flag(self) -> None:
        assert set(NUMERIC_FLAGS) <= set(ALLOWED_FLAGS)

    def test_every_numeric_default_is_written_as_a_plain_number(self) -> None:
        """A default that does not parse would fail every launch that relied
        on it, and only the launches that relied on it -- so the ones that
        never mention the flag would pass and hide it. Round-tripping through
        int() also rejects a default written as ``"00"`` or ``" 0"``, which
        parse but are not what a reader of the help would be told."""
        defaults = {flag: OPTIONAL_FLAGS[flag] for flag in NUMERIC_FLAGS if flag in OPTIONAL_FLAGS}
        assert {flag: str(int(value)) for flag, value in defaults.items()} == defaults

    def test_every_flag_is_described(self) -> None:
        """The help is generated from these tables, so a flag with no
        description would render a blank line rather than fail here."""
        assert sorted(FLAG_HELP) == sorted(ALLOWED_FLAGS)


class TestTheHelp:
    def test_it_names_every_flag(self) -> None:
        text = "\n".join(render_usage())
        for flag in ALLOWED_FLAGS:
            assert flag in text

    def test_it_states_what_each_optional_flag_falls_back_to(self) -> None:
        text = "\n".join(render_usage())
        assert "[default: 22]" in text
        assert "[default: (empty)]" in text

    def test_it_separates_what_must_be_given_from_what_need_not(self) -> None:
        text = "\n".join(render_usage())
        assert "required:" in text
        assert "optional:" in text

    def test_asking_for_help_is_not_a_failed_match(self) -> None:
        """A launcher that exits non-zero on --help makes every wrapper treat
        a puzzled human as a failure."""
        with FakeHost() as host:
            assert main(["--help"]) == EXIT_HELP
            assert any("usage:" in line for line in host.printed)

    def test_a_bare_invocation_explains_itself_rather_than_erroring(self) -> None:
        """Someone who types the module name and presses enter gets the help,
        not a stack trace about a missing --port."""
        with FakeHost() as host:
            assert main([]) == EXIT_HELP
            assert any("--port" in line for line in host.printed)

    def test_help_is_offered_even_beside_other_flags(self) -> None:
        with FakeHost():
            assert main(["--port", "1", "--help"]) == EXIT_HELP


class TestRunningIt:
    def test_it_reads_the_process_arguments_when_given_none(self) -> None:
        with FakeHost() as host:
            host.argv = ["--help"]
            assert main(None) == EXIT_HELP

    def test_a_malformed_launch_propagates_rather_than_becoming_a_status(self) -> None:
        """A sweep that swallowed it would file the failure as a match
        result, which is a measurement nobody took."""
        with FakeHost(), pytest.raises(ValueError):
            main(["--port", "27511"])

    def test_a_complete_launch_reaches_the_match(self) -> None:
        with FakeHost(platform="linux") as host:
            host.files["agent/src/rwbot/agent/Agent.java"] = ()
            assert main(list(MINIMAL)) == 0
            assert len(host.spawned) == 1

    def test_the_module_guard_runs_main(self) -> None:
        """``python -m rw_bot.harness.play_match_cli`` is how a sweep and the
        Makefile both start a match, so the guard that makes that work is part
        of the launcher rather than boilerplate."""
        with FakeHost() as host:
            host.argv = ["--help"]
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("rw_bot.harness.play_match_cli", run_name="__main__")
            assert caught.value.code == EXIT_HELP
