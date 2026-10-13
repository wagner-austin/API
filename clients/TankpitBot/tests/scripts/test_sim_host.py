"""Tests for operating the sim server on sedona from the hub.

Every command the script runs goes through ``scripts._test_hooks.run_command``,
here a :class:`tests.scripts._recorder.Recorder` that answers by the command's
words and keeps every argv, so each test asserts the exact commands sent to
sedona. The secrets file and the scratch directory are real files under
``tmp_path``.
"""

from __future__ import annotations

import runpy
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.fleet_remote import SEDONA_SSH, FleetHostError
from scripts.sim_host import (
    COMPOSE_FILE,
    SEDONA_DOCKER_HOST,
    SIM_DIR,
    SMOKE_ATTEMPTS,
    SMOKE_PAUSE_SECONDS,
    down,
    ensure_image,
    head_sha,
    image_tag,
    pull_database,
    run,
    smoke,
    stage,
    up,
)

from scripts import _test_hooks as script_hooks
from scripts import sim_host
from tankpit_bot import _test_hooks as core_hooks
from tests.scripts._recorder import Recorder, fail, ok

COMPOSE = f"{SIM_DIR}/{COMPOSE_FILE}"
SIM_ENV = f"{SIM_DIR}/sim.env"
SMOKE = ["ssh", SEDONA_SSH, "curl.exe", "-s", "-o", "NUL", "-w", "%{http_code} %{size_download}"]
SERVING = ok("200 65536")


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


def _secrets(tmp_path: Path) -> Path:
    secrets = tmp_path / "secrets" / "sim.env"
    secrets.parent.mkdir()
    secrets.write_text("SIM_DB_PASSWORD=p\n", encoding="utf-8")
    return secrets


def _scratch(tmp_path: Path) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return scratch


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
    """The sim's image is tagged by the commit it was built from, not the fleet's ladder."""
    assert image_tag("abc1234") == "tankpit-sim:abc1234"


def test_head_sha_is_heads_short_sha(tmp_path: Path) -> None:
    """HEAD's short sha, trimmed of git's newline."""
    script_hooks.run_command = Recorder({"rev-parse": ok("abc1234\n")})
    assert head_sha(tmp_path) == "abc1234"


def test_head_sha_refuses_outside_a_checkout(tmp_path: Path) -> None:
    """No HEAD is refused by name."""
    script_hooks.run_command = Recorder({"rev-parse": fail("not a git repository")})
    with pytest.raises(FleetHostError) as raised:
        head_sha(tmp_path)
    assert str(raised.value) == (
        "SIM_NOT_COMMITTED: git rev-parse --short HEAD exited 1: not a git repository"
    )


def test_stage_sends_heads_compose_file_and_the_env(tmp_path: Path) -> None:
    """Stage sends HEAD's compose file, and sim.env naming the image with the password."""
    recorder = Recorder({f"git show HEAD:./{COMPOSE_FILE}": ok('{"name": "tankpit-sim"}\n')})
    script_hooks.run_command = recorder
    scratch = _scratch(tmp_path)
    stage(tmp_path, "abc1234", _secrets(tmp_path), scratch)
    assert recorder.calls == [
        ["git", "show", f"HEAD:./{COMPOSE_FILE}"],
        [
            "ssh",
            SEDONA_SSH,
            "powershell",
            "-NoProfile",
            "-Command",
            f"$null = New-Item -ItemType Directory -Force -Path {SIM_DIR}",
        ],
        [
            "scp",
            "-q",
            str(scratch / COMPOSE_FILE),
            str(scratch / "sim.env"),
            f"{SEDONA_SSH}:{SIM_DIR}/",
        ],
    ]
    assert recorder.staged == {
        COMPOSE_FILE: '{"name": "tankpit-sim"}\n',
        "sim.env": "SIM_IMAGE=tankpit-sim:abc1234\nSIM_DB_PASSWORD=p\n",
    }


def test_stage_refuses_without_the_secrets(tmp_path: Path) -> None:
    """No password file, no stage, and nothing is sent."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    absent = tmp_path / "absent.env"
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, "abc1234", absent, _scratch(tmp_path))
    assert str(raised.value) == (
        f"SIM_SECRETS_MISSING: {absent} is absent; it holds the sim's SIM_DB_PASSWORD"
    )
    assert recorder.calls == []


def test_stage_refuses_a_compose_file_head_does_not_hold(tmp_path: Path) -> None:
    """A compose file HEAD lacks is refused before sedona is touched."""
    recorder = Recorder({"git show": fail("does not exist in 'HEAD'")})
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError, match=r"^SIM_NOT_COMMITTED: git show"):
        stage(tmp_path, "abc1234", _secrets(tmp_path), _scratch(tmp_path))
    assert len(recorder.calls) == 1


def test_stage_refuses_when_sedona_refuses_a_copy(tmp_path: Path) -> None:
    """A refused copy is refused by name."""
    script_hooks.run_command = Recorder({"scp": fail("permission denied")})
    with pytest.raises(FleetHostError, match=r"^SIM_STAGE_FAILED: scp -q .* permission denied$"):
        stage(tmp_path, "abc1234", _secrets(tmp_path), _scratch(tmp_path))


def test_pull_database_pulls_through_the_hubs_client(tmp_path: Path) -> None:
    """The database image is pulled by the hub's docker client against sedona's daemon."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    pull_database(tmp_path, tmp_path)
    assert recorder.calls == [
        [
            "docker",
            "--host",
            SEDONA_DOCKER_HOST,
            "compose",
            "-f",
            str(tmp_path / COMPOSE_FILE),
            "--env-file",
            str(tmp_path / "sim.env"),
            "pull",
            "sim-db",
        ]
    ]


def test_pull_database_refuses_a_failed_pull(tmp_path: Path) -> None:
    """A failed pull is refused by name."""
    script_hooks.run_command = Recorder({"pull": fail("manifest unknown")})
    with pytest.raises(FleetHostError, match=r"^SIM_PULL_FAILED: .* manifest unknown$"):
        pull_database(tmp_path, tmp_path)


def test_ensure_image_keeps_a_tag_sedona_holds(tmp_path: Path) -> None:
    """A tag sedona already holds is not built again."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    assert ensure_image(tmp_path / "clients" / "TankpitBot", "abc1234", _scratch(tmp_path)) is False
    assert recorder.calls == [
        ["docker", "--host", SEDONA_DOCKER_HOST, "image", "inspect", "tankpit-sim:abc1234"]
    ]


def test_ensure_image_builds_heads_archive_when_the_tag_is_missing(tmp_path: Path) -> None:
    """A missing tag is built on sedona from an archive of HEAD, never the working tree."""
    recorder = Recorder({"image inspect": fail("No such image")})
    script_hooks.run_command = recorder
    scratch = _scratch(tmp_path)
    context = scratch / "context"
    assert ensure_image(tmp_path / "clients" / "TankpitBot", "abc1234", scratch) is True
    assert recorder.calls[1:] == [
        [
            "git",
            "archive",
            "--format=tar",
            "-o",
            str(scratch / "snapshot.tar"),
            "HEAD",
            "--",
            "clients/TankpitBot",
            "libs/platform_core",
            "libs/monorepo_guards",
        ],
        ["tar", "-xf", str(scratch / "snapshot.tar"), "-C", str(context)],
        [
            "docker",
            "--host",
            SEDONA_DOCKER_HOST,
            "build",
            "-f",
            str(context / "clients" / "TankpitBot" / "Dockerfile"),
            "-t",
            "tankpit-sim:abc1234",
            "--build-arg",
            "BUILD_REF=sim-abc1234",
            str(context),
        ],
    ]
    assert context.is_dir()


def test_ensure_image_refuses_a_failed_build(tmp_path: Path) -> None:
    """A failed build is refused by name."""
    script_hooks.run_command = Recorder(
        {"image inspect": fail("No such image"), " build ": fail("COPY failed")}
    )
    with pytest.raises(FleetHostError, match=r"^SIM_IMAGE_BUILD_FAILED: .* COPY failed$"):
        ensure_image(tmp_path / "clients" / "TankpitBot", "abc1234", _scratch(tmp_path))


def test_smoke_waits_for_the_whole_field(tmp_path: Path) -> None:
    """Smoke asks sedona's loopback until room 1's terrain arrives whole."""
    recorder = Recorder({"curl.exe": [ok("000 0"), ok("200 512"), SERVING]})
    script_hooks.run_command = recorder
    pauses = Pauses()
    script_hooks.sleep_seconds = pauses
    smoke(tmp_path)
    assert recorder.calls == [[*SMOKE, "http://127.0.0.1:8765/terrain/1"]] * 3
    assert pauses.seconds == [SMOKE_PAUSE_SECONDS, SMOKE_PAUSE_SECONDS]


def test_smoke_gives_up_naming_the_last_answer(tmp_path: Path) -> None:
    """A sim that never serves is refused after every attempt, naming what it last said."""
    script_hooks.run_command = Recorder({"curl.exe": ok("000 0\r\n")})
    pauses = Pauses()
    script_hooks.sleep_seconds = pauses
    with pytest.raises(FleetHostError) as raised:
        smoke(tmp_path)
    assert str(raised.value) == (
        f"SIM_SMOKE_FAILED: http://127.0.0.1:8765/terrain/1 on sedona after {SMOKE_ATTEMPTS} "
        "attempts answered '000 0', not '200 65536'"
    )
    assert len(pauses.seconds) == SMOKE_ATTEMPTS - 1


def test_up_stages_pulls_builds_composes_and_smokes(tmp_path: Path) -> None:
    """Up runs HEAD's sim on sedona, composed with no build there, and sees it serve."""
    recorder = Recorder(
        {
            "rev-parse": ok("abc1234\n"),
            "image inspect": fail("No such image"),
            "curl.exe": SERVING,
        }
    )
    script_hooks.run_command = recorder
    script_hooks.sleep_seconds = Pauses()
    lines = up(tmp_path / "clients" / "TankpitBot", _secrets(tmp_path))
    assert [
        "ssh",
        SEDONA_SSH,
        "docker",
        "compose",
        "-f",
        COMPOSE,
        "--env-file",
        SIM_ENV,
        "up",
        "-d",
        "--no-build",
    ] in recorder.calls
    assert lines == [
        f"{COMPOSE_FILE} at abc1234 staged into {SEDONA_SSH}:{SIM_DIR}",
        "image tankpit-sim:abc1234 built on sedona",
        "sim up; http://127.0.0.1:8765/terrain/1 on sedona serves the whole field",
    ]


def test_up_refuses_when_compose_refuses(tmp_path: Path) -> None:
    """A refused compose up is refused by name, and nothing is smoked."""
    recorder = Recorder({"rev-parse": ok("abc1234\n"), " up -d": fail("port is already allocated")})
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError, match=r"^SIM_COMPOSE_FAILED: .* port is already allocated$"):
        up(tmp_path / "clients" / "TankpitBot", _secrets(tmp_path))
    assert not any("curl.exe" in call for call in recorder.calls)


def test_down_stops_the_sim_and_its_database(tmp_path: Path) -> None:
    """Down stops every service of the sim's project; the volume is untouched."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    assert down(tmp_path) == [
        f"sim and sim-db stopped on {SEDONA_SSH}; the accounts stay in the sim-db volume"
    ]
    assert recorder.calls == [
        ["ssh", SEDONA_SSH, "docker", "compose", "-f", COMPOSE, "--env-file", SIM_ENV, "stop"]
    ]


def test_main_runs_down_and_prints_its_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """``down`` from the command line prints what it did."""
    script_hooks.run_command = Recorder({})
    core_hooks.get_argv = lambda: ["sim_host", "down"]
    sim_host.main()
    assert capsys.readouterr().out == (
        f"sim and sim-db stopped on {SEDONA_SSH}; the accounts stay in the sim-db volume\n"
    )


def test_run_up_uses_the_secrets_it_is_given(tmp_path: Path) -> None:
    """``up`` on the command line stages the secrets file it is handed."""
    recorder = Recorder({"rev-parse": ok("abc1234\n"), "curl.exe": SERVING})
    script_hooks.run_command = recorder
    lines = run(["up"], tmp_path / "clients" / "TankpitBot", _secrets(tmp_path))
    assert lines[1] == "image tankpit-sim:abc1234 already on sedona"
    assert recorder.staged["sim.env"].endswith("SIM_DB_PASSWORD=p\n")


@pytest.mark.parametrize("arguments", [[], ["sideways"], ["up", "down"]])
def test_run_refuses_anything_else_with_the_usage(
    tmp_path: Path, arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Any other argument list prints the usage and exits 2, running nothing."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    with pytest.raises(SystemExit) as exited:
        run(arguments, tmp_path, tmp_path)
    assert exited.value.code == 2
    assert capsys.readouterr().out == "usage: python -m scripts.sim_host {up|down}\n"
    assert recorder.calls == []


def test_module_runs_as_a_program(capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m scripts.sim_host`` reaches main."""
    core_hooks.get_argv = lambda: ["sim_host"]
    with pytest.raises(SystemExit):
        runpy.run_module("scripts.sim_host", run_name="__main__")
    assert capsys.readouterr().out.startswith("usage:")
