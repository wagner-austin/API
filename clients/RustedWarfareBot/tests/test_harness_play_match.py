"""Playing one match: build, start, wait, plan, tear down.

Driven against :class:`tests.harness_fakes.FakeHost`, so these exercise the
real control flow rather than a rehearsal of it -- the real teardown ordering,
the real orphan check, the real frozen-tree branch.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.launch import LaunchConfig
from rw_bot.harness.play_match import (
    EXIT_AGENT_BUILD_FAILED,
    EXIT_NO_CHANNEL,
    build_agent,
    clear_orphaned_engine,
    play,
)
from rw_bot.platform_id import WINDOWS
from tests.harness_fakes import FakeHost

LINUX = "linux"


def _config(
    tree: str = "", map_path: str = "", port: int = 27511, display: int = 0
) -> LaunchConfig:
    """Build a launch for the tests below.

    Args:
        tree: Frozen snapshot, or empty to compile the agent.
        map_path: Skirmish map, or empty for the sandbox.
        port: Channel port.
        display: X display to start a server on, or 0 for none.

    Returns:
        The launch.
    """
    return LaunchConfig(
        port=port,
        game_dir=".game-w1",
        seed=42,
        lockstep=75,
        settle=22,
        display=display,
        play_log="runs/sweeps/demo/logs/a-s1.log",
        map=map_path,
        opponents=1,
        difficulty=0,
        tree=tree,
        pin_delta=0,
        fast_forward=0,
        rng_tap=0,
        extra_agent_args="",
        module="scripts.play",
        catalogue="cat.log",
        type_dump="types.ndjson",
        play_args="1500 doctrines/a.doctrine -",
    )


def _host(platform: str = LINUX) -> FakeHost:
    """Build a host with the agent sources planted.

    Args:
        platform: Which platform to pretend to be.

    Returns:
        The host.
    """
    host = FakeHost(platform=platform)
    host.files["agent/src/rwbot/agent/Agent.java"] = ()
    host.files["agent/src/rwbot/agent/Targets.java"] = ()
    return host


class TestBuildingTheAgent:
    def test_a_failing_compiler_stops_before_packaging(self) -> None:
        """Packaging a failed compile produces a jar of whatever survived,
        which attaches without error and never opens the channel."""
        with _host() as host:
            host.command_results["javac"] = (1, ("error: cannot find symbol",))
            assert build_agent(_config(), "a.jar", "build/x", LINUX) is False
            assert not [c for c in host.commands if c[0].rsplit("/", 1)[-1] == "jar"]

    def test_a_failing_compiler_reports_what_it_said(self) -> None:
        with _host() as host:
            host.command_results["javac"] = (1, ("error: cannot find symbol",))
            build_agent(_config(), "a.jar", "build/x", LINUX)
            assert "error: cannot find symbol" in host.printed
            assert "[play] javac failed" in host.printed

    def test_a_failing_packager_is_reported_as_itself(self) -> None:
        with _host() as host:
            host.command_results["jar"] = (1, ("no manifest",))
            assert build_agent(_config(), "a.jar", "build/x", LINUX) is False
            assert "[play] jar failed" in host.printed

    def test_a_clean_build_reports_success(self) -> None:
        with _host():
            assert build_agent(_config(), "a.jar", "build/x", LINUX) is True

    def test_the_tools_come_out_of_the_clone(self) -> None:
        """The clone's JDK is the pinned one; the system compiler is a
        different toolchain."""
        with _host() as host:
            build_agent(_config(), "a.jar", "build/x", LINUX)
            assert host.commands[0][0] == ".game-w1/jvm-linux/bin/javac"
            assert host.commands[1][0] == ".game-w1/jvm-linux/bin/jar"


class TestClearingAnOrphanedEngine:
    def test_a_free_port_is_left_alone(self) -> None:
        with _host() as host:
            clear_orphaned_engine(27511, LINUX)
            assert host.felled == []

    def test_a_jvm_holding_the_port_is_felled(self) -> None:
        """A worker killed mid-match leaves its engine alive; the job requeues
        onto the same clone and dies at the bind (vhdoom96b, 2026-08-09)."""
        with _host() as host:
            host.command_results["ss"] = (
                0,
                ('LISTEN 0 1 127.0.0.1:27511 0.0.0.0:* users:(("java",pid=999,fd=5))',),
            )
            host.command_results["ps"] = (0, ("java",))
            clear_orphaned_engine(27511, LINUX)
            assert host.felled == [999]

    def test_a_bystander_on_the_port_is_not_felled(self) -> None:
        """Felling it is worse than failing to start: the match that does not
        start is reported, and the process that vanishes is not."""
        with _host() as host:
            host.command_results["ss"] = (
                0,
                ('LISTEN 0 1 127.0.0.1:27511 0.0.0.0:* users:(("code",pid=999,fd=5))',),
            )
            host.command_results["ps"] = (0, ("code",))
            clear_orphaned_engine(27511, LINUX)
            assert host.felled == []
            assert any("leaving it alone" in line for line in host.printed)


class TestPlayingAMatch:
    def test_a_frozen_batch_does_not_compile(self) -> None:
        """The jar was built once when the snapshot was taken; recompiling per
        match would be a different jar attached to a frozen experiment."""
        with _host() as host:
            play(_config(tree="runs/sweeps/demo/.tree"))
            assert not [c for c in host.commands if c[0].endswith("javac")]
            assert "[play] frozen tree: runs/sweeps/demo/.tree" in host.printed

    def test_a_build_failure_never_starts_an_engine(self) -> None:
        with _host() as host:
            host.command_results["javac"] = (1, ("boom",))
            assert play(_config()) == EXIT_AGENT_BUILD_FAILED
            assert host.spawned == []

    def test_the_engine_runs_in_the_game_directory(self) -> None:
        """It writes three fixed-name paths inside its own directory, so the
        working directory IS the isolation between concurrent matches."""
        with _host() as host:
            play(_config(tree="t"))
            assert host.spawned[0][1] == "/repo/.game-w1"

    def test_the_engine_is_given_the_dynamic_linker_search_path(self) -> None:
        """Its native GUI stack links against itself: librocketConnector.so
        needs libRocketCore.so.1 beside it, and the JVM's own library path
        does not tell the linker that."""
        with _host() as host:
            play(_config(tree="t"))
            assert host.engine_environment["LD_LIBRARY_PATH"] == "."

    def test_the_engine_still_inherits_the_rest_of_the_environment(self) -> None:
        with _host() as host:
            play(_config(tree="t"))
            assert host.engine_environment["PATH"] == "/usr/bin"

    def test_the_two_engine_streams_are_kept_apart(self) -> None:
        """A crash interleaved into a merged stream is a crash nobody found."""
        with _host() as host:
            play(_config(tree="t"))
            _, _, out, err = host.spawned[0]
            assert out.endswith("a-s1.log.agent")
            assert err.endswith("a-s1.log.err")

    def test_a_channel_that_never_opens_is_reported_and_not_planned(self) -> None:
        with _host() as host:
            host.channel_opens = False
            assert play(_config(tree="t")) == EXIT_NO_CHANNEL
            assert host.inherited == []

    def test_the_reason_the_channel_never_opened_is_carried_out(self) -> None:
        """A refused connection means the engine is up and the agent never
        bound; a timeout with no route means the engine died during boot.
        Ninety seconds of silence followed by "never opened port N" reported
        neither, and they are fixed in different files."""
        with _host() as host:
            host.channel_opens = False
            play(_config(tree="t"))
            reported = [line for line in host.printed if "never opened port 27511" in line]
            assert len(reported) == 1
            assert "ConnectionRefusedError" in reported[0]

    def test_the_engine_is_felled_even_when_the_channel_never_opened(self) -> None:
        """An engine left holding a leased port kills the NEXT match."""
        with _host() as host:
            host.channel_opens = False
            play(_config(tree="t"))
            assert host.felled == [4242]

    def test_the_engine_is_felled_after_a_normal_match(self) -> None:
        with _host() as host:
            play(_config(tree="t"))
            assert host.felled == [4242]

    def test_the_sandbox_is_left_to_settle_and_a_named_map_is_not(self) -> None:
        """Under a named map the world is held at its first frame, so settling
        would only burn wall clock."""
        with _host() as sandbox:
            play(_config(tree="t"))
            assert 22 in sandbox.slept
        with _host() as named:
            play(_config(tree="t", map_path="maps/x.tmx"))
            assert 22 not in named.slept

    def test_the_planner_status_is_the_matchs_status(self) -> None:
        with _host() as host:
            host.planner_status = 7
            assert play(_config(tree="t")) == 7

    def test_a_frozen_tree_reaches_the_planner_as_a_path(self) -> None:
        with _host() as host:
            play(_config(tree="runs/sweeps/demo/.tree"))
            _, environment = host.inherited[0]
            assert environment["PYTHONPATH"] == (
                "/repo/runs/sweeps/demo/.tree:/repo/runs/sweeps/demo/.tree/src"
            )

    def test_a_working_tree_match_sets_no_path(self) -> None:
        """A blank PYTHONPATH is not an unset one."""
        with _host() as host:
            play(_config())
            _, environment = host.inherited[0]
            assert "PYTHONPATH" not in environment

    def test_the_planner_inherits_the_rest_of_the_environment(self) -> None:
        with _host() as host:
            play(_config(tree="t"))
            _, environment = host.inherited[0]
            assert environment["PATH"] == "/usr/bin"

    def test_a_compiled_jar_and_its_classes_are_removed(self) -> None:
        with _host() as host:
            play(_config())
            assert "agent/build/play-stamp001" in host.removed
            assert "agent/build/rw-agent-play-stamp001.jar" in host.removed

    def test_a_snapshots_jar_is_never_removed(self) -> None:
        """It belongs to the batch, not to this match, and every later match
        in the batch attaches the same one."""
        with _host() as host:
            play(_config(tree="runs/sweeps/demo/.tree"))
            assert host.removed == []

    @pytest.mark.parametrize("platform", [WINDOWS, LINUX])
    def test_a_match_plays_the_same_way_on_either_platform(self, platform: str) -> None:
        """The point of the whole lift: one launcher, two platforms, and the
        only differences are the ones the platform genuinely forces."""
        with _host(platform=platform) as host:
            assert play(_config(tree="t")) == 0
            assert len(host.spawned) == 1
            assert len(host.inherited) == 1
