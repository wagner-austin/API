"""A match a human watches, or joins: ``make watch`` and ``make host``.

Driven against :class:`tests.harness_fakes.FakeHost`, so the build, the wait,
the planner and the teardown are the real control flow.
"""

from __future__ import annotations

import runpy

import pytest
from scripts.interactive import (
    HOST,
    OPTIONAL_FLAGS,
    WAIT_SECONDS,
    WATCH,
    InteractiveCommandError,
    announce,
    interactive_config,
    main,
    play_interactive,
)

from rw_bot.harness.launch import (
    CATALOGUE,
    HEAP,
    LIBRARY_PATH,
    MAIN_CLASS,
    TYPE_DUMP,
    agent_arguments,
)
from rw_bot.harness.play_match import EXIT_AGENT_BUILD_FAILED, EXIT_NO_CHANNEL
from tests.harness_fakes import FakeHost

MAP = "maps/skirmish/[p2]duel_lake.tmx"
BASE = ["--port", "27612", "--game-dir", ".game", "--play-log", "runs/w.log", "--map", MAP]


def _values(**overrides: str) -> dict[str, str]:
    """Build the flag values a launch is read from.

    Args:
        **overrides: Flag values by field name, e.g. ``difficulty="5"``.

    Returns:
        Values by flag, defaults applied.
    """
    values = {**OPTIONAL_FLAGS, "--port": "27612", "--game-dir": ".game"}
    values.update({"--play-log": "runs/w.log", "--map": MAP})
    values.update({f"--{key.replace('_', '-')}": value for key, value in overrides.items()})
    return values


def _host() -> FakeHost:
    """Build a host with an agent source and a machine address.

    Returns:
        The host.
    """
    host = FakeHost()
    host.files["agent/src/rwbot/agent/Premain.java"] = ()
    host.command_results["hostname"] = (0, ("192.168.1.20 100.64.0.7 ",))
    return host


class TestTheLaunchEachModeIs:
    def test_a_watched_game_names_its_skirmish_to_the_agent(self) -> None:
        config = interactive_config(WATCH, _values(difficulty="5"))
        assert agent_arguments(config) == (
            f"channelPort=27612;matchMap={MAP};matchOpponents=1;matchDifficulty=5"
        )

    def test_a_hosted_game_names_only_the_map_it_hosts(self) -> None:
        """The human is the opponent, so there is no skirmish to describe."""
        config = interactive_config(HOST, _values())
        assert agent_arguments(config) == f"channelPort=27612;hostMap={MAP}"

    def test_the_reproducibility_machinery_is_off(self) -> None:
        """A person watches the wall clock; a pinned delta or a lockstep would
        stop the simulation tracking it."""
        config = interactive_config(WATCH, _values())
        assert (config["seed"], config["lockstep"], config["pin_delta"]) == (0, 0, 0)
        assert (config["catalogue"], config["type_dump"]) == (CATALOGUE, TYPE_DUMP)

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(InteractiveCommandError) as caught:
            interactive_config("spectate", _values())
        assert caught.value.code == "RW-INTERACTIVE-001"


class TestAnnouncing:
    def test_a_hosted_game_gives_the_human_every_address_to_join(self) -> None:
        with _host() as host:
            announce(HOST)
            assert any("192.168.1.20 or 100.64.0.7 (port 5123)" in line for line in host.printed)

    def test_a_machine_that_cannot_name_an_address_is_not_hosted(self) -> None:
        with _host() as host:
            host.command_results["hostname"] = (1, ())
            with pytest.raises(InteractiveCommandError) as caught:
                announce(HOST)
        assert caught.value.code == "RW-INTERACTIVE-002"

    def test_a_watched_game_runs_no_command_to_announce_itself(self) -> None:
        with _host() as host:
            announce(WATCH)
            assert host.commands == []


class TestPlayingIt:
    def test_a_watched_game_opens_a_window_and_runs_the_planner(self) -> None:
        with _host() as host:
            assert play_interactive(WATCH, interactive_config(WATCH, _values())) == 0
            command = host.spawned[0][0]
            assert "-nodisplay" not in command
            assert command[command.index("-width") + 1] == "1280"
            assert "-sandbox" not in command
            assert host.inherited[0][0] == (
                "/venv/bin/python",
                "-m",
                "scripts.play",
                "27612",
                CATALOGUE,
                TYPE_DUMP,
            )
            assert host.felled == [4242]
            assert "agent/build/watch-stamp001" in host.removed
            assert "agent/build/rw-agent-watch-stamp001.jar" in host.removed

    def test_a_hosted_game_stays_headless(self) -> None:
        with _host() as host:
            play_interactive(HOST, interactive_config(HOST, _values()))
            assert host.spawned[0][0] == (
                "/repo/.game/jvm-linux/bin/java",
                HEAP,
                LIBRARY_PATH,
                f"-javaagent:/repo/agent/build/rw-agent-host-stamp001.jar=channelPort=27612;hostMap={MAP}",
                "-cp",
                "game-lib.jar:libs/*",
                MAIN_CLASS,
                "-nodisplay",
                "-nosound",
                "-width",
                "800",
                "-height",
                "600",
                "-log",
                "/repo/runs/w.log",
            )

    def test_a_failed_build_starts_no_game(self) -> None:
        with _host() as host:
            host.command_results["javac"] = (1, ("error",))
            assert play_interactive(WATCH, interactive_config(WATCH, _values())) == (
                EXIT_AGENT_BUILD_FAILED
            )
            assert host.spawned == []

    def test_a_lobby_nobody_joins_ends_at_its_deadline(self) -> None:
        """A quiet lobby is not a hung engine, so the wait is a deadline."""
        with _host() as host:
            host.channel_opens = False
            assert play_interactive(HOST, interactive_config(HOST, _values())) == EXIT_NO_CHANNEL
            assert sum(host.slept) >= WAIT_SECONDS[HOST]
            assert any("nobody joined the lobby" in line for line in host.printed)
            assert host.felled == [4242]
            assert host.inherited == []

    def test_an_engine_that_dies_ends_the_wait_at_once(self) -> None:
        with _host() as host:
            host.channel_opens = False
            host.engine_exit_status = 1
            assert play_interactive(WATCH, interactive_config(WATCH, _values())) == EXIT_NO_CHANNEL
            assert host.slept == []
            assert any("exited with status 1" in line for line in host.printed)

    def test_a_channel_that_opens_late_still_plays(self) -> None:
        with _host() as host:
            host.channel_opens = False
            host.channel_opens_after = 3
            assert play_interactive(WATCH, interactive_config(WATCH, _values())) == 0
            assert len(host.inherited) == 1


class TestTheCommandLine:
    def test_a_complete_command_plays(self) -> None:
        with _host() as host:
            assert main(["--mode", "watch", *BASE]) == 0
            assert len(host.spawned) == 1

    def test_the_map_is_required(self) -> None:
        with _host(), pytest.raises(ValueError, match="--map is required"):
            main(["--mode", "watch", *BASE[:-2]])

    @pytest.mark.parametrize("flag", ["--port", "--opponents", "--difficulty"])
    def test_a_numeric_flag_must_be_a_whole_number(self, flag: str) -> None:
        values = dict(zip(BASE[::2], BASE[1::2], strict=True))
        values[flag] = "hard"
        argv = ["--mode", "watch", *(token for pair in values.items() for token in pair)]
        with _host() as host, pytest.raises(InteractiveCommandError) as caught:
            main(argv)
        assert caught.value.code == "RW-INTERACTIVE-003"
        assert caught.value.message == f"{flag} must be a whole number, got 'hard'"
        assert host.spawned == []

    def test_the_module_guard_runs_main_on_the_process_arguments(self) -> None:
        with _host() as host:
            host.argv = ["--mode", "watch", *BASE]
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.interactive", run_name="__main__")
            assert caught.value.code == 0
