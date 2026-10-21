"""Building the agent jar a batch freezes, and certifying it against the game.

Driven against :class:`tests.harness_fakes.FakeHost`, whose ``jar`` writes the
jar it was asked for and whose rename moves it, so a publish is judged by the
files it leaves rather than by the calls it made.
"""

from __future__ import annotations

import runpy

import pytest
from scripts.agent import (
    EXIT_SELFTEST_FAILED,
    AgentCommandError,
    check_published,
    main,
    publish_agent,
    run_selftest,
)

from rw_bot.harness.agent_build import BUILT_AGENT_JAR, selftest_command
from rw_bot.harness.play_match import EXIT_AGENT_BUILD_FAILED, EXIT_OK, TOOL_WALL_SECONDS
from tests.harness_fakes import FakeHost

#: What the fake host's stamp makes the publish's stamped paths.
STAGED = f"{BUILT_AGENT_JAR}.stamp001.new"
CLASSES = "agent/build/classes-stamp001"
VERIFY_CLASSES = "agent/build/verify-stamp001"

#: The bundled ``java`` inside the pinned game the self-test runs on.
GAME_JAVA = ".game/jvm-linux/bin/java"


def _host() -> FakeHost:
    """Build a host with an agent source planted and an older jar published.

    Returns:
        The host.
    """
    host = FakeHost()
    host.files["agent/src/rwbot/agent/Premain.java"] = ()
    host.files[BUILT_AGENT_JAR] = ("the previous build",)
    return host


class TestPublishingTheJar:
    def test_a_clean_build_replaces_the_published_jar(self) -> None:
        with _host() as host:
            assert publish_agent() == EXIT_OK
            assert host.replaced == [(STAGED, BUILT_AGENT_JAR)]
            assert host.files[BUILT_AGENT_JAR] == ()
            assert STAGED not in host.files
            assert f"[agent] built {BUILT_AGENT_JAR}" in host.printed

    def test_it_compiles_into_a_stamped_directory_and_removes_it(self) -> None:
        """Two builds at once must not interleave class files."""
        with _host() as host:
            publish_agent()
            assert host.commands[0][host.commands[0].index("-d") + 1] == CLASSES
            assert host.commands[1][2] == STAGED
            assert CLASSES in host.removed
            assert STAGED in host.removed

    def test_a_failed_compile_leaves_the_published_jar_as_it_was(self) -> None:
        with _host() as host:
            host.command_results["javac"] = (1, ("error: cannot find symbol",))
            assert publish_agent() == EXIT_AGENT_BUILD_FAILED
            assert host.files[BUILT_AGENT_JAR] == ("the previous build",)
            assert host.replaced == []
            assert CLASSES in host.removed

    def test_a_failed_package_leaves_the_published_jar_as_it_was(self) -> None:
        with _host() as host:
            host.command_results["jar"] = (1, ("no manifest",))
            assert publish_agent() == EXIT_AGENT_BUILD_FAILED
            assert host.files[BUILT_AGENT_JAR] == ("the previous build",)


class TestReadingThePublishBack:
    def test_a_published_jar_with_no_staged_copy_passes(self) -> None:
        with _host():
            check_published(STAGED)

    def test_an_absent_published_jar_is_refused(self) -> None:
        with _host() as host:
            del host.files[BUILT_AGENT_JAR]
            with pytest.raises(AgentCommandError) as caught:
                check_published(STAGED)
        assert caught.value.code == "RW-AGENT-001"

    def test_a_staged_jar_that_survived_the_rename_is_refused(self) -> None:
        """The PowerShell recipe once reported a build over a jar it never
        replaced; a staged copy still standing is that, read off the disk."""
        with _host() as host:
            host.files[STAGED] = ()
            with pytest.raises(AgentCommandError) as caught:
                check_published(STAGED)
        assert caught.value.code == "RW-AGENT-001"
        assert STAGED in caught.value.message


class TestTheSelfTest:
    def test_it_runs_the_fresh_classes_on_the_games_own_runtime(self) -> None:
        with _host() as host:
            host.command_results[GAME_JAVA] = (0, ("PASS Targets 14/14",))
            assert run_selftest(".game") == EXIT_OK
            assert host.commands[1] == selftest_command(".game", VERIFY_CLASSES)
            assert host.walls[1] == TOOL_WALL_SECONDS
            assert "PASS Targets 14/14" in host.printed
            assert "[agent-selftest] passed" in host.printed
            assert VERIFY_CLASSES in host.removed

    def test_the_command_is_relative_to_the_root_it_runs_from(self) -> None:
        assert selftest_command(".game", VERIFY_CLASSES) == (
            GAME_JAVA,
            "-cp",
            f"{VERIFY_CLASSES}:.game/game-lib.jar:.game/libs/*",
            "rwbot.agent.SelfTest",
            ".game/game-lib.jar",
        )

    def test_a_refused_patch_is_its_own_exit_and_names_the_check(self) -> None:
        with _host() as host:
            host.command_results[GAME_JAVA] = (1, ("FAIL Targets: VerifyError in xq",))
            assert run_selftest(".game") == EXIT_SELFTEST_FAILED
            assert "FAIL Targets: VerifyError in xq" in host.printed
            assert "[agent-selftest] failed: the self-test exited 1" in host.printed
            assert VERIFY_CLASSES in host.removed

    def test_sources_that_do_not_compile_never_reach_the_game(self) -> None:
        with _host() as host:
            host.command_results["javac"] = (1, ("error",))
            assert run_selftest(".game") == EXIT_AGENT_BUILD_FAILED
            assert len(host.commands) == 1
            assert VERIFY_CLASSES in host.removed

    def test_it_never_touches_the_published_jar(self) -> None:
        """So it runs while a game holds that jar attached."""
        with _host() as host:
            run_selftest(".game")
            assert host.files[BUILT_AGENT_JAR] == ("the previous build",)
            assert host.replaced == []


class TestTheCommandLine:
    def test_build_publishes(self) -> None:
        with _host() as host:
            assert main(["build"]) == EXIT_OK
            assert host.replaced == [(STAGED, BUILT_AGENT_JAR)]

    def test_build_takes_no_flags(self) -> None:
        with _host(), pytest.raises(ValueError, match="unknown argument"):
            main(["build", "--game-dir", ".game"])

    def test_selftest_needs_the_game(self) -> None:
        with _host(), pytest.raises(ValueError, match="--game-dir is required"):
            main(["selftest"])

    def test_selftest_runs_against_the_named_game(self) -> None:
        with _host() as host:
            assert main(["selftest", "--game-dir", ".game-w1"]) == EXIT_OK
            assert host.commands[1][0] == ".game-w1/jvm-linux/bin/java"

    @pytest.mark.parametrize("argv", [[], ["publish"]])
    def test_a_missing_or_unknown_command_is_refused(self, argv: list[str]) -> None:
        with _host(), pytest.raises(AgentCommandError) as caught:
            main(argv)
        assert caught.value.code == "RW-AGENT-002"

    def test_the_module_guard_runs_main_on_the_process_arguments(self) -> None:
        """``python -m scripts.agent build`` is the Makefile's."""
        with _host() as host:
            host.argv = ["build"]
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.agent", run_name="__main__")
            assert caught.value.code == EXIT_OK
