"""Tests for operating the sim server on the game host.

Every command the script runs goes through ``scripts._test_hooks.run_command``,
here a :class:`tests.scripts._recorder.Recorder` that answers by the command's
words and keeps every argv, so each test asserts the exact commands sent to
the host. The scratch directory is real, under ``tmp_path``.
"""

from __future__ import annotations

import runpy
import shlex
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.fleet_release import ARCHIVED_PATHS
from scripts.fleet_remote import GAME_HOST_SSH, FleetHostError, on_host
from scripts.sim_host import (
    COMPOSE_FILE,
    SECRETS_FILE,
    SIM_DIR,
    SMOKE_ATTEMPTS,
    SMOKE_PAUSE_SECONDS,
    down,
    ensure_image,
    head_sha,
    image_tag,
    run,
    smoke,
    stage,
    up,
)

from scripts import _test_hooks as script_hooks
from scripts import sim_host
from tankpit_bot import _test_hooks as core_hooks
from tests.scripts._recorder import Recorder, fail, ok

COMPOSE = ["docker", "compose", "-f", f"{SIM_DIR}/{COMPOSE_FILE}"]
COMPOSE += ["--env-file", f"{SIM_DIR}/sim.env", "--env-file", f"{SIM_DIR}/{SECRETS_FILE}"]
SMOKE = on_host(
    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}:%{size_download}",
     "http://127.0.0.1:8765/terrain/1"]
)  # fmt: skip
SERVING = ok("200:65536")
CONTEXT = f"{SIM_DIR}/context-1a2b3c4d"


class Archiving(Recorder):
    """A recorder whose ``git archive`` writes the archive it was asked for."""

    def __call__(self, argv: list[str], cwd: Path) -> script_hooks.CommandResult:
        """Write the archive a ``git archive -o`` names, then answer as the recorder does.

        Args:
            argv: The command.
            cwd: Ignored.

        Returns:
            The scripted answer.
        """
        if argv[:4] == ["git", "-C", "../..", "archive"]:
            Path(argv[argv.index("-o") + 1]).write_text("tar bytes", encoding="utf-8")
        return super().__call__(argv, cwd)


class Pauses:
    """A sleep that records each pause instead of sleeping."""

    def __init__(self) -> None:
        """Start with no pauses."""
        self.seconds: list[float] = []

    def __call__(self, seconds: float) -> None:
        """Record one pause.

        Args:
            seconds: How long the caller asked to sleep.
        """
        self.seconds.append(seconds)


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Put back every hook a test replaced.

    Yields:
        None, with the original hooks restored after.
    """
    run_command = script_hooks.run_command
    sleep_seconds = script_hooks.sleep_seconds
    get_argv = core_hooks.get_argv
    yield
    script_hooks.run_command = run_command
    script_hooks.sleep_seconds = sleep_seconds
    core_hooks.get_argv = get_argv


def test_image_tag_names_the_commit() -> None:
    """The sim's image is tagged by the commit it was built from."""
    assert image_tag("1a2b3c4d") == "tankpit-sim:1a2b3c4d"


def test_head_sha_is_heads_short_sha(tmp_path: Path) -> None:
    """HEAD's short sha, stripped of git's newline."""
    script_hooks.run_command = Recorder({"rev-parse": ok("1a2b3c4d\n")})
    assert head_sha(tmp_path) == "1a2b3c4d"


def test_head_sha_refuses_outside_a_checkout(tmp_path: Path) -> None:
    """No checkout means no commit to build, refused with git's words."""
    script_hooks.run_command = Recorder({"rev-parse": fail("fatal: not a git repository")})
    with pytest.raises(FleetHostError) as raised:
        head_sha(tmp_path)
    assert str(raised.value) == (
        "SIM_NOT_COMMITTED: git rev-parse --short HEAD exited 1: fatal: not a git repository"
    )


def test_stage_checks_the_password_then_sends_heads_compose_file(tmp_path: Path) -> None:
    """Stage asks for the host's password file, then sends HEAD's compose file and sim.env."""
    recorder = Recorder({"git show": ok('{"name": "tankpit-sim"}\n')})
    script_hooks.run_command = recorder
    stage(tmp_path, "1a2b3c4d", tmp_path)
    assert recorder.calls == [
        on_host(["ls", "-1", f"{SIM_DIR}/{SECRETS_FILE}"]),
        ["git", "show", f"HEAD:./{COMPOSE_FILE}"],
        ["scp", "-q", str(tmp_path / COMPOSE_FILE), str(tmp_path / "sim.env"),
         f"{GAME_HOST_SSH}:{SIM_DIR}/"],
    ]  # fmt: skip
    assert recorder.staged == {
        COMPOSE_FILE: '{"name": "tankpit-sim"}\n',
        "sim.env": "SIM_IMAGE=tankpit-sim:1a2b3c4d\n",
    }


def test_stage_refuses_a_host_without_the_password(tmp_path: Path) -> None:
    """A host with no secrets file is refused before HEAD is read."""
    recorder = Recorder({"ls -1": fail("ls: cannot access: No such file or directory")})
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, "1a2b3c4d", tmp_path)
    assert str(raised.value) == (
        f"SIM_SECRETS_MISSING: ssh {GAME_HOST_SSH} ls -1 {SIM_DIR}/{SECRETS_FILE} exited 1: "
        "ls: cannot access: No such file or directory"
    )
    assert len(recorder.calls) == 1


def test_stage_refuses_a_compose_file_head_does_not_hold(tmp_path: Path) -> None:
    """A compose file missing from HEAD is refused with git's own words."""
    script_hooks.run_command = Recorder({"git show": fail("fatal: path does not exist")})
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, "1a2b3c4d", tmp_path)
    assert str(raised.value) == (
        f"SIM_NOT_COMMITTED: git show HEAD:./{COMPOSE_FILE} exited 1: fatal: path does not exist"
    )


def test_stage_refuses_when_the_host_refuses_a_copy(tmp_path: Path) -> None:
    """A copy the host refuses stops the stage by name."""
    script_hooks.run_command = Recorder({"scp": fail("scp: permission denied")})
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, "1a2b3c4d", tmp_path)
    assert str(raised.value).startswith("SIM_STAGE_FAILED: scp -q ")
    assert str(raised.value).endswith("exited 1: scp: permission denied")


def test_ensure_image_keeps_a_tag_the_host_holds(tmp_path: Path) -> None:
    """A tag the host has is not rebuilt, and HEAD is not archived."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    assert ensure_image(tmp_path, "1a2b3c4d", tmp_path) is False
    assert recorder.calls == [on_host(["docker", "image", "inspect", "tankpit-sim:1a2b3c4d"])]


def test_ensure_image_builds_heads_tree_on_the_host_and_removes_it(tmp_path: Path) -> None:
    """A missing tag is built on the host from HEAD's unpacked tree, which is removed after."""
    recorder = Archiving({"image inspect": fail("No such image")})
    script_hooks.run_command = recorder
    assert ensure_image(tmp_path, "1a2b3c4d", tmp_path) is True
    archive = tmp_path / "context-1a2b3c4d.tar"
    assert recorder.calls[1:] == [
        ["git", "-C", "../..", "archive", "--format=tar", "-o", str(archive), "HEAD", "--",
         *ARCHIVED_PATHS],
        on_host(["mkdir", "-p", SIM_DIR]),
        ["scp", "-q", str(archive), f"{GAME_HOST_SSH}:{SIM_DIR}/"],
        on_host(
            ["sh", "-c",
             'set -e; rm -rf "$2"; mkdir "$2"; tar -xf "$1" -C "$2"; rm "$1"; mv "$2" "$3"',
             "sh", f"{CONTEXT}.tar", f"{CONTEXT}.partial", CONTEXT]
        ),
        on_host(
            ["docker", "build", "-f", f"{CONTEXT}/clients/TankpitBot/Dockerfile",
             "-t", "tankpit-sim:1a2b3c4d", "--build-arg", "BUILD_REF=sim-1a2b3c4d", CONTEXT]
        ),
        on_host(["rm", "-rf", CONTEXT]),
    ]  # fmt: skip


def test_ensure_image_refuses_a_failed_build(tmp_path: Path) -> None:
    """A build that fails is refused with the build's error, and the tree is left to read."""
    recorder = Archiving({"image inspect": fail("No such image"), " build ": fail("no space")})
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError) as raised:
        ensure_image(tmp_path, "1a2b3c4d", tmp_path)
    assert str(raised.value).startswith(f"SIM_IMAGE_BUILD_FAILED: ssh {GAME_HOST_SSH} docker build")
    assert str(raised.value).endswith("exited 1: no space")
    assert shlex.split(recorder.calls[-1][2])[:2] == ["docker", "build"]


def test_smoke_waits_for_the_whole_field(tmp_path: Path) -> None:
    """Smoke asks until the sim serves all 65536 tiles, pausing between asks."""
    recorder = Recorder({"curl": [ok("000:0"), ok("200:12"), SERVING]})
    script_hooks.run_command = recorder
    pauses = Pauses()
    script_hooks.sleep_seconds = pauses
    smoke(tmp_path)
    assert recorder.calls == [SMOKE, SMOKE, SMOKE]
    assert pauses.seconds == [SMOKE_PAUSE_SECONDS, SMOKE_PAUSE_SECONDS]


def test_smoke_hands_the_curl_format_to_the_host_as_one_word() -> None:
    """The format's braces and percent signs reach curl as one argument."""
    assert shlex.split(SMOKE[2])[5] == "%{http_code}:%{size_download}"


def test_smoke_gives_up_naming_the_last_answer(tmp_path: Path) -> None:
    """A sim that never serves the whole field is refused with what it last answered."""
    script_hooks.run_command = Recorder({"curl": ok("502:0")})
    script_hooks.sleep_seconds = Pauses()
    with pytest.raises(FleetHostError) as raised:
        smoke(tmp_path)
    assert str(raised.value) == (
        f"SIM_SMOKE_FAILED: http://127.0.0.1:8765/terrain/1 on {GAME_HOST_SSH} after "
        f"{SMOKE_ATTEMPTS} attempts answered '502:0', not '200:65536'"
    )


def test_up_stages_pulls_builds_composes_and_smokes(tmp_path: Path) -> None:
    """Up runs HEAD's sim on the host and reports what it did."""
    recorder = Archiving(
        {"rev-parse": ok("1a2b3c4d\n"), "image inspect": fail("No such image"), "curl": SERVING}
    )
    script_hooks.run_command = recorder
    lines = up(tmp_path)
    remote = [shlex.split(call[2]) if call[0] == "ssh" else call for call in recorder.calls]
    assert remote[4] == [*COMPOSE, "pull", "sim-db"]
    assert remote[-2] == [*COMPOSE, "up", "-d", "--no-build"]
    assert recorder.calls[-1] == SMOKE
    assert lines == [
        f"{COMPOSE_FILE} at 1a2b3c4d staged into {GAME_HOST_SSH}:{SIM_DIR}",
        "image tankpit-sim:1a2b3c4d built on the host",
        "sim up; http://127.0.0.1:8765/terrain/1 on the host serves the whole field",
    ]


def test_up_refuses_a_failed_pull(tmp_path: Path) -> None:
    """A database image the registry refuses stops up by name."""
    script_hooks.run_command = Recorder({"rev-parse": ok("1a2b3c4d\n"), " pull ": fail("denied")})
    with pytest.raises(FleetHostError) as raised:
        up(tmp_path)
    assert str(raised.value).startswith(f"SIM_PULL_FAILED: ssh {GAME_HOST_SSH} docker compose")
    assert str(raised.value).endswith("exited 1: denied")


def test_up_refuses_when_compose_refuses(tmp_path: Path) -> None:
    """Compose refusing on the host is refused by name, before any smoke."""
    script_hooks.run_command = Recorder(
        {"rev-parse": ok("1a2b3c4d\n"), " up -d ": fail("port is already allocated")}
    )
    with pytest.raises(FleetHostError) as raised:
        up(tmp_path)
    assert str(raised.value).startswith(f"SIM_COMPOSE_FAILED: ssh {GAME_HOST_SSH} docker compose")
    assert str(raised.value).endswith("exited 1: port is already allocated")


def test_down_stops_the_sim_and_its_database(tmp_path: Path) -> None:
    """Down stops both services and keeps the accounts' volume."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    assert down(tmp_path) == [
        f"sim and sim-db stopped on {GAME_HOST_SSH}; the accounts stay in the sim-db volume"
    ]
    assert recorder.calls == [on_host([*COMPOSE, "stop"])]


def test_main_runs_down_and_prints_its_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """``down`` from the command line prints what it did."""
    script_hooks.run_command = Recorder({})
    core_hooks.get_argv = lambda: ["sim_host", "down"]
    sim_host.main()
    assert capsys.readouterr().out == (
        f"sim and sim-db stopped on {GAME_HOST_SSH}; the accounts stay in the sim-db volume\n"
    )


def test_run_up_reports_a_tag_the_host_already_holds(tmp_path: Path) -> None:
    """``up`` on the command line runs HEAD's sim, reusing an image the host has."""
    script_hooks.run_command = Recorder({"rev-parse": ok("1a2b3c4d\n"), "curl": SERVING})
    lines = run(["up"], tmp_path)
    assert lines[1] == "image tankpit-sim:1a2b3c4d already on the host"


@pytest.mark.parametrize("arguments", [[], ["sideways"], ["up", "down"]])
def test_run_refuses_anything_else_with_the_usage(
    tmp_path: Path, arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Any other argument list prints the usage and exits 2, running nothing."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    with pytest.raises(SystemExit) as exited:
        run(arguments, tmp_path)
    assert exited.value.code == 2
    assert capsys.readouterr().out == "usage: python -m scripts.sim_host {up|down}\n"
    assert recorder.calls == []


def test_module_runs_as_a_program(capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m scripts.sim_host`` reaches main."""
    core_hooks.get_argv = lambda: ["sim_host"]
    with pytest.raises(SystemExit):
        runpy.run_module("scripts.sim_host", run_name="__main__")
    assert capsys.readouterr().out.startswith("usage:")
