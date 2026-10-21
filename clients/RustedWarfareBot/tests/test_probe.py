"""Running the game once for what it reports, rather than for a match.

The command is pinned whole per kind, because a flag moved across the main
class is read by the engine rather than the JVM and the boot dies; the run is
driven against :class:`tests.harness_fakes.FakeHost`.
"""

from __future__ import annotations

import runpy
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath, PurePosixPath

import pytest
from scripts.probe import (
    EXIT_PROBE_FAILED,
    EXIT_PROBE_TIMED_OUT,
    PROBE_WALL_SECONDS,
    ProbeCommandError,
    agent_options,
    main,
    probe_command,
    run_probe,
)

from rw_bot.harness.launch import HEAP, LIBRARY_PATH, MAIN_CLASS
from rw_bot.harness.play_match import EXIT_OK
from tests.harness_fake_game import FakeGame
from tests.harness_fakes import FakeHost

ROOT = PurePosixPath("/repo")
JAVA = "/repo/.game/jvm-linux/bin/java"
CLASSPATH = ("-cp", "game-lib.jar:libs/*", MAIN_CLASS)


class WritingHost(FakeHost):
    """A host whose engine writes the agent's output file when it starts.

    Attributes:
        writes: The file the engine writes, and its lines.
    """

    def __init__(self, writes: tuple[str, tuple[str, ...]]) -> None:
        super().__init__()
        self.writes = writes

    def spawn_game(
        self,
        argv: Sequence[str],
        cwd: PurePath,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str],
    ) -> FakeGame:
        """Start the engine, which writes its output as the real agent does.

        Args:
            argv: The engine command.
            cwd: Its working directory.
            stdout_path: Where its output goes.
            stderr_path: Where its errors go.
            env: Its environment.

        Returns:
            The engine handle.
        """
        path, lines = self.writes
        self.files[path] = lines
        return super().spawn_game(argv, cwd, stdout_path, stderr_path, env)


class TestTheCommandEachKindRuns:
    def test_a_boot_probe_runs_the_engine_alone(self) -> None:
        assert probe_command("boot", ".game", "runs/boot.log", "", "", 0, ROOT) == (
            JAVA,
            HEAP,
            LIBRARY_PATH,
            *CLASSPATH,
            "-nodisplay",
            "-nosound",
            "-width",
            "800",
            "-height",
            "600",
            "-log",
            "/repo/runs/boot.log",
        )

    def test_a_unit_dump_asks_for_the_catalogue(self) -> None:
        assert probe_command("units", ".game", "runs/u.log", "", "", 0, ROOT) == (
            JAVA,
            HEAP,
            LIBRARY_PATH,
            *CLASSPATH,
            "-nodisplay",
            "-nosound",
            "-printunits",
            "-log",
            "/repo/runs/u.log",
        )

    def test_a_sandbox_probe_attaches_the_published_agent(self) -> None:
        command = probe_command("sandbox", ".game", "runs/s.log", "", "", 0, ROOT)
        assert command[3] == "-javaagent:/repo/agent/build/rw-agent.jar"
        assert command[4:] == (
            *CLASSPATH,
            "-nodisplay",
            "-nosound",
            "-sandbox",
            "-width",
            "800",
            "-height",
            "600",
            "-log",
            "/repo/runs/s.log",
        )

    def test_a_discovery_probe_snapshots_at_its_offsets_and_exits(self) -> None:
        command = probe_command("discover", ".game", "l", "5,25", "", 0, ROOT)
        assert command[3] == (
            "-javaagent:/repo/agent/build/rw-agent.jar="
            "discoverAtSeconds=5,25;exitAfterDiscovery=true"
        )

    def test_a_leased_display_brings_its_own_server(self) -> None:
        command = probe_command("boot", ".game", "l", "", "", 91, ROOT)
        assert command[0] == "xvfb-run"
        assert command[5] == JAVA

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ProbeCommandError) as caught:
            probe_command("match", ".game", "l", "", "", 0, ROOT)
        assert caught.value.code == "RW-PROBE-001"

    @pytest.mark.parametrize(("kind", "at"), [("boot", "5"), ("sandbox", "5"), ("discover", "")])
    def test_offsets_go_to_the_discovery_kinds_and_only_there(self, kind: str, at: str) -> None:
        """Given elsewhere they would be dropped silently, and the probe would
        report on a run that was not the one asked for."""
        with pytest.raises(ProbeCommandError) as caught:
            probe_command(kind, ".game", "l", at, "", 0, ROOT)
        assert caught.value.code == "RW-PROBE-002"

    @pytest.mark.parametrize(("kind", "out"), [("discover", "x"), ("wire", ""), ("units", "x")])
    def test_an_output_goes_to_the_writing_kinds_and_only_there(self, kind: str, out: str) -> None:
        at = "5" if kind != "units" else ""
        with pytest.raises(ProbeCommandError) as caught:
            probe_command(kind, ".game", "l", at, out, 0, ROOT)
        assert caught.value.code == "RW-PROBE-003"


class TestWhatTheAgentIsTold:
    def test_the_world_stream_goes_to_the_absolute_output(self) -> None:
        """The engine runs from the game directory, so a relative path would
        land inside the clone."""
        assert agent_options("wire", "16,17,18", "runs/w.ndjson", ROOT) == (
            "discoverAtSeconds=16,17,18;stateOutPath=/repo/runs/w.ndjson;exitAfterDiscovery=true"
        )

    def test_the_registry_goes_to_its_own_option(self) -> None:
        assert agent_options("type-flags", "20", "runs/t.ndjson", ROOT) == (
            "discoverAtSeconds=20;typeFlagsPath=/repo/runs/t.ndjson;exitAfterDiscovery=true"
        )

    def test_a_plain_sandbox_is_told_nothing(self) -> None:
        assert agent_options("sandbox", "", "", ROOT) == ""


class TestRunningAProbe:
    def test_a_clean_exit_reports_the_log(self) -> None:
        with FakeHost() as host:
            host.engine_exit_status = 0
            assert run_probe("boot", ".game", "runs/boot.log", "", "", 0) == EXIT_OK
            command, cwd, stdout, stderr = host.spawned[0]
            assert command[-1] == "/repo/runs/boot.log"
            assert cwd == "/repo/.game"
            assert (stdout, stderr) == ("runs/boot.log.agent", "runs/boot.log.err")
            assert host.engine_environment["LD_LIBRARY_PATH"] == "."
            assert host.felled == []
            assert "[probe] boot done, log runs/boot.log" in host.printed

    def test_a_failed_engine_names_its_error_stream(self) -> None:
        with FakeHost() as host:
            host.engine_exit_status = 1
            assert run_probe("units", ".game", "runs/u.log", "", "", 0) == EXIT_PROBE_FAILED
            assert "[probe] the engine exited 1; see runs/u.log.err" in host.printed

    def test_an_engine_that_never_exits_is_felled_at_the_wall(self) -> None:
        with FakeHost() as host:
            assert run_probe("sandbox", ".game", "l", "", "", 0) == EXIT_PROBE_TIMED_OUT
            assert host.felled == [4242]
            assert sum(host.slept) >= PROBE_WALL_SECONDS

    def test_the_output_is_this_runs_and_is_counted(self) -> None:
        """Removed first, so a count cannot include a previous capture."""
        host = WritingHost(("runs/w.ndjson", ("{}", "{}", "{}")))
        with host:
            host.engine_exit_status = 0
            host.files["runs/w.ndjson"] = ("stale",)
            status = run_probe("wire", ".game", "runs/w.log", "16", "runs/w.ndjson", 0)
            assert status == EXIT_OK
            assert "runs/w.ndjson" in host.removed
            assert "[probe] 3 line(s) written to runs/w.ndjson" in host.printed
            assert "stateOutPath=/repo/runs/w.ndjson" in host.spawned[0][0][3]


class TestTheCommandLine:
    def test_the_required_flags_are_required(self) -> None:
        with FakeHost(), pytest.raises(ValueError, match="--log is required"):
            main(["--kind", "boot", "--game-dir", ".game"])

    def test_a_display_that_is_not_a_number_is_refused(self) -> None:
        with FakeHost(), pytest.raises(ProbeCommandError) as caught:
            main(["--kind", "boot", "--game-dir", ".game", "--log", "l", "--display", "x"])
        assert caught.value.code == "RW-PROBE-004"

    def test_a_complete_command_runs_the_probe(self) -> None:
        with FakeHost() as host:
            host.engine_exit_status = 0
            argv = ["--kind", "boot", "--game-dir", ".g", "--log", "l", "--display", "91"]
            assert main(argv) == EXIT_OK
            assert host.spawned[0][0][:3] == ("xvfb-run", "-n", "91")

    def test_the_module_guard_runs_main_on_the_process_arguments(self) -> None:
        with FakeHost() as host:
            host.engine_exit_status = 0
            host.argv = ["--kind", "boot", "--game-dir", ".game", "--log", "l"]
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.probe", run_name="__main__")
            assert caught.value.code == EXIT_OK
