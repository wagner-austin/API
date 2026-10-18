"""Tests for operating the fleet on the game host.

Every command the script runs goes through ``scripts._test_hooks.run_command``,
here a recorder that answers by the command's words and keeps every argv, so
each test asserts the exact commands sent to the host.
"""

from __future__ import annotations

import runpy
import shlex
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.fleet_gate import FLEET_URL, GATE_INSTANCE
from scripts.fleet_host import (
    HOST_DIR,
    HOST_FILES,
    PUBLIC_ORIGIN,
    REGISTRY_FILE,
    RELEASES_DIR,
    SMOKE_ATTEMPTS,
    SMOKE_PAUSE_SECONDS,
    down,
    ensure_image,
    image_tag,
    newest_release,
    run,
    smoke,
    stage,
    up,
)
from scripts.fleet_remote import GAME_HOST_SSH, FleetHostError, on_host

from scripts import _test_hooks as script_hooks
from scripts import fleet_host
from tankpit_bot import _test_hooks as core_hooks
from tests.scripts._fleet_answers import passing_gate
from tests.scripts._recorder import Recorder, fail, ok

COMPOSE = ["docker", "compose", "-f", f"{HOST_DIR}/docker-compose.yml"]
COMPOSE += ["--env-file", f"{HOST_DIR}/host.env"]

#: The ladder listing ``find`` prints: two releases, the older first.
LADDER = "1000.5 v0.1.0-aaaaaaaa\n2000.25 v0.1.0-bbbbbbbb\n"


class Answer:
    """One HTTP answer with a status and no body."""

    def __init__(self, status_code: int) -> None:
        """Hold the status.

        Args:
            status_code: The answer's status.
        """
        self.status_code = status_code
        self.content = b""


def _remote(call: list[str]) -> list[str]:
    """The words one ssh call ran on the host.

    Args:
        call: A recorded ``ssh <host> <line>`` argv.

    Returns:
        The line, split back into its words.
    """
    assert call[:2] == ["ssh", GAME_HOST_SSH]
    return shlex.split(call[2])


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Put back every hook a test replaced.

    Yields:
        None, with the original hooks restored after.
    """
    run_command = script_hooks.run_command
    http_get = script_hooks.http_get
    sleep_seconds = script_hooks.sleep_seconds
    get_argv = core_hooks.get_argv
    yield
    script_hooks.run_command = run_command
    script_hooks.http_get = http_get
    script_hooks.sleep_seconds = sleep_seconds
    core_hooks.get_argv = get_argv


def test_newest_release_is_the_last_written(tmp_path: Path) -> None:
    """The ladder's newest directory by write time is the release, whatever order find prints."""
    recorder = Recorder({"find": ok("2000.25 v0.1.0-bbbbbbbb\n1000.5 v0.1.0-aaaaaaaa\n")})
    script_hooks.run_command = recorder
    assert newest_release(tmp_path) == "v0.1.0-bbbbbbbb"
    assert _remote(recorder.calls[0]) == [
        "find", RELEASES_DIR, "-mindepth", "1", "-maxdepth", "1", "-type", "d",
        "-printf", "%T@ %f\n",
    ]  # fmt: skip
    assert image_tag("v0.1.0-bbbbbbbb") == "tankpit-fleet:v0.1.0-bbbbbbbb"


def test_newest_release_refuses_an_empty_ladder(tmp_path: Path) -> None:
    """A ladder holding no release is refused by name."""
    script_hooks.run_command = Recorder({"find": ok("")})
    with pytest.raises(FleetHostError) as raised:
        newest_release(tmp_path)
    assert str(raised.value) == (
        f"FLEET_NO_RELEASE: {GAME_HOST_SSH}:{RELEASES_DIR} holds no release; run make release first"
    )


def test_newest_release_refuses_a_ladder_it_cannot_read(tmp_path: Path) -> None:
    """A missing ladder is refused with find's own words."""
    script_hooks.run_command = Recorder({"find": fail("find: No such file or directory")})
    with pytest.raises(FleetHostError) as raised:
        newest_release(tmp_path)
    assert str(raised.value).startswith(f"FLEET_NO_RELEASE: ssh {GAME_HOST_SSH} find ")
    assert str(raised.value).endswith("exited 1: find: No such file or directory")


def test_stage_seeds_the_registry_checks_the_hosts_files_then_sends_heads_files(
    tmp_path: Path,
) -> None:
    """Stage seeds a missing registry, asks for the host's files, then sends HEAD's and host.env."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recorder = Recorder(
        {
            "git show HEAD:./docker-compose.yml": ok("name: tankpitbot\n"),
            "git show HEAD:./edge/nginx.conf": ok("server {}\n"),
        }
    )
    script_hooks.run_command = recorder
    stage(tmp_path, "v0.1.0-cccccccc", scratch)
    held = [f"{HOST_DIR}/{name}" for name in HOST_FILES]
    registry = f"{RELEASES_DIR}/v0.1.0-cccccccc/clients/TankpitBot/{REGISTRY_FILE}"
    assert recorder.calls == [
        on_host(["mkdir", "-p", f"{HOST_DIR}/edge", f"{HOST_DIR}/data", f"{HOST_DIR}/runs"]),
        on_host(["cp", "--update=none", registry, f"{HOST_DIR}/{REGISTRY_FILE}"]),
        on_host(["ls", "-1", *held]),
        ["git", "show", "HEAD:./docker-compose.yml"],
        ["git", "show", "HEAD:./edge/nginx.conf"],
        [
            "scp",
            "-q",
            str(scratch / "docker-compose.yml"),
            str(scratch / "host.env"),
            f"{GAME_HOST_SSH}:{HOST_DIR}/",
        ],
        ["scp", "-q", str(scratch / "edge" / "nginx.conf"), f"{GAME_HOST_SSH}:{HOST_DIR}/edge/"],
    ]
    assert recorder.staged == {
        "docker-compose.yml": "name: tankpitbot\n",
        "host.env": "FLEET_IMAGE=tankpit-fleet:v0.1.0-cccccccc\n",
        "nginx.conf": "server {}\n",
    }


def test_stage_refuses_a_host_without_its_own_files(tmp_path: Path) -> None:
    """A host lacking the token, the accounts or the registry is refused, naming the file.

    The registry was seeded first, so a host refused for it is one whose
    release carries none either.
    """
    recorder = Recorder(
        {"ls -1": fail(f"ls: cannot access '{HOST_DIR}/edge.env': No such file or directory")}
    )
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, "v0.1.0-dddddddd", tmp_path)
    assert str(raised.value).startswith(f"FLEET_HOST_FILES_MISSING: ssh {GAME_HOST_SSH} ls -1 ")
    assert str(raised.value).endswith(
        f"exited 1: ls: cannot access '{HOST_DIR}/edge.env': No such file or directory"
    )
    assert len(recorder.calls) == 3


def test_stage_refuses_a_file_head_does_not_hold(tmp_path: Path) -> None:
    """A compose file missing from HEAD is refused with git's own words."""
    script_hooks.run_command = Recorder(
        {"git show": fail("fatal: path 'docker-compose.yml' does not exist in 'HEAD'\n")}
    )
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, "v0.1.0-dddddddd", tmp_path)
    assert str(raised.value) == (
        "FLEET_NOT_COMMITTED: git show HEAD:./docker-compose.yml exited 1: "
        "fatal: path 'docker-compose.yml' does not exist in 'HEAD'"
    )


def test_stage_refuses_when_the_host_refuses_a_copy(tmp_path: Path) -> None:
    """A copy the host refuses stops the stage by name."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    script_hooks.run_command = Recorder({"scp": fail("scp: permission denied")})
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, "v0.1.0-eeeeeeee", scratch)
    assert str(raised.value) == (
        f"FLEET_STAGE_FAILED: scp -q {scratch / 'docker-compose.yml'} {scratch / 'host.env'} "
        f"{GAME_HOST_SSH}:{HOST_DIR}/ exited 1: scp: permission denied"
    )


def test_ensure_image_builds_only_a_missing_tag(tmp_path: Path) -> None:
    """A tag the host holds is not rebuilt; a missing one is built there from the release."""
    held = Recorder({})
    script_hooks.run_command = held
    assert ensure_image(tmp_path, "v0.1.0-ffffffff") is False
    assert held.calls == [on_host(["docker", "image", "inspect", "tankpit-fleet:v0.1.0-ffffffff"])]
    missing = Recorder({"image inspect": fail("No such image")})
    script_hooks.run_command = missing
    assert ensure_image(tmp_path, "v0.1.0-ffffffff") is True
    context = f"{RELEASES_DIR}/v0.1.0-ffffffff"
    assert _remote(missing.calls[1]) == [
        "docker", "build", "-f", f"{context}/clients/TankpitBot/Dockerfile",
        "-t", "tankpit-fleet:v0.1.0-ffffffff",
        "--build-arg", "BUILD_REF=v0.1.0-ffffffff", context,
    ]  # fmt: skip


def test_ensure_image_refuses_a_failed_build(tmp_path: Path) -> None:
    """A build that fails is refused with the build's error."""
    script_hooks.run_command = Recorder(
        {"image inspect": fail("No such image"), " build ": fail("no space left")}
    )
    with pytest.raises(FleetHostError) as raised:
        ensure_image(tmp_path, "v0.1.0-00000000")
    message = str(raised.value)
    assert message.startswith(f"FLEET_IMAGE_BUILD_FAILED: ssh {GAME_HOST_SSH} docker build")
    assert message.endswith("exited 1: no space left")


class Pauses:
    """A sleep that records how long it was asked to block, and returns at once."""

    def __init__(self) -> None:
        """Start with no pauses."""
        self.taken: list[float] = []

    def __call__(self, seconds: float) -> None:
        """Record one pause.

        Args:
            seconds: How long the caller asked to block.
        """
        self.taken.append(seconds)


def _serving(url: str) -> Answer:
    return Answer(204 if url.endswith("/healthz") else 200)


def _cloudflare_down(url: str) -> Answer:
    return Answer(530)


def test_smoke_waits_for_the_filter_and_the_roster() -> None:
    """Smoke asks until both answer, pausing between rounds."""
    answers = iter([502, 530, 204, 502, 204, 200])
    asked: list[str] = []
    pauses = Pauses()

    def get(url: str) -> Answer:
        asked.append(url)
        return Answer(next(answers))

    script_hooks.http_get = get
    script_hooks.sleep_seconds = pauses
    smoke(PUBLIC_ORIGIN)
    assert asked == [f"{PUBLIC_ORIGIN}/healthz", f"{PUBLIC_ORIGIN}/demo/fleet"] * 3
    assert pauses.taken == [SMOKE_PAUSE_SECONDS, SMOKE_PAUSE_SECONDS]


def test_smoke_gives_up_naming_the_last_answers() -> None:
    """An origin that never serves both is refused with what it last said."""
    script_hooks.http_get = _cloudflare_down
    pauses = Pauses()
    script_hooks.sleep_seconds = pauses
    with pytest.raises(FleetHostError) as raised:
        smoke(PUBLIC_ORIGIN)
    assert str(raised.value) == (
        f"FLEET_SMOKE_FAILED: {PUBLIC_ORIGIN} after {SMOKE_ATTEMPTS} attempts: "
        "/healthz answered 530, /demo/fleet answered 530"
    )
    assert pauses.taken == [SMOKE_PAUSE_SECONDS] * (SMOKE_ATTEMPTS - 1)


def test_up_stages_pulls_builds_composes_smokes_and_gates(tmp_path: Path) -> None:
    """Up runs the newest release with its edge on the host, sees a bot tick, and reports it all."""
    recorder = Recorder(
        {"find": ok(LADDER), "image inspect": fail("No such image"), **passing_gate()}
    )
    script_hooks.run_command = recorder
    script_hooks.http_get = _serving
    script_hooks.sleep_seconds = Pauses()
    lines = up(tmp_path)
    remote = [_remote(call) if call[0] == "ssh" else call for call in recorder.calls]
    verbs = [words[0] if words[0] != "docker" else words[1] for words in remote]
    assert verbs[:12] == [
        "find",
        "mkdir",
        "cp",
        "ls",
        "git",
        "git",
        "scp",
        "scp",
        "compose",
        "image",
        "build",
        "compose",
    ]
    assert remote[8] == [*COMPOSE, "--profile", "edge", "pull", "public", "tunnel"]
    assert remote[11] == [
        *COMPOSE, "--profile", "edge", "up", "-d", "--no-build", "fleet", "public", "tunnel",
    ]  # fmt: skip
    gate_requests = [
        "spawn" if "POST" in words else words[-1].removeprefix(FLEET_URL) for words in remote[12:]
    ]
    assert gate_requests == [
        "/accounts",
        "/bots",
        "spawn",
        f"/bots/{GATE_INSTANCE}/stats",
        "/bots",
        f"/bots/{GATE_INSTANCE}/activity",
    ]
    assert lines == [
        f"release v0.1.0-bbbbbbbb staged into {GAME_HOST_SSH}:{HOST_DIR}",
        "image tankpit-fleet:v0.1.0-bbbbbbbb built on the host",
        f"fleet and edge up; {PUBLIC_ORIGIN} serves /healthz and /demo/fleet",
        "gate bot 'gate' on Arterial reached tick 2 at fuel 942 in Practice",
    ]


def test_up_refuses_a_failed_pull(tmp_path: Path) -> None:
    """An edge image the registry refuses stops up by name, before any build."""
    recorder = Recorder({"find": ok(LADDER), " pull ": fail("manifest unknown")})
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError) as raised:
        up(tmp_path)
    assert str(raised.value).startswith(f"FLEET_PULL_FAILED: ssh {GAME_HOST_SSH} docker compose")
    assert str(raised.value).endswith("exited 1: manifest unknown")
    assert "build" not in " ".join(recorder.calls[-1])


def test_up_refuses_when_compose_refuses(tmp_path: Path) -> None:
    """Compose refusing on the host is refused by name, before any smoke."""
    script_hooks.run_command = Recorder(
        {"find": ok(LADDER), " up -d ": fail("edge.env: no such file")}
    )
    with pytest.raises(FleetHostError) as raised:
        up(tmp_path)
    assert str(raised.value).startswith(f"FLEET_COMPOSE_FAILED: ssh {GAME_HOST_SSH} docker compose")
    assert str(raised.value).endswith("exited 1: edge.env: no such file")


def test_down_stops_the_fleet_only(tmp_path: Path) -> None:
    """Down drains the fleet and leaves the edge up."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    assert down(tmp_path) == [
        f"fleet stopped on {GAME_HOST_SSH}; the edge still answers, and the demo shows it offline"
    ]
    assert recorder.calls == [on_host([*COMPOSE, "stop", "fleet"])]


def test_main_runs_down_and_prints_its_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """``down`` from the command line prints what it did."""
    script_hooks.run_command = Recorder({})
    core_hooks.get_argv = lambda: ["fleet_host", "down"]
    fleet_host.main()
    assert capsys.readouterr().out == (
        f"fleet stopped on {GAME_HOST_SSH}; the edge still answers, and the demo shows it offline\n"
    )


def test_run_up_runs_the_hosts_newest_release(tmp_path: Path) -> None:
    """``up`` on the command line runs the newest release on the host's ladder."""
    script_hooks.run_command = Recorder({"find": ok(LADDER), **passing_gate()})
    script_hooks.http_get = _serving
    script_hooks.sleep_seconds = Pauses()
    lines = run(["up"], tmp_path)
    assert lines[1] == "image tankpit-fleet:v0.1.0-bbbbbbbb already on the host"


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
    assert capsys.readouterr().out == "usage: python -m scripts.fleet_host {up|down}\n"
    assert recorder.calls == []


def test_module_runs_as_a_program(capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m scripts.fleet_host`` reaches main."""
    core_hooks.get_argv = lambda: ["fleet_host"]
    with pytest.raises(SystemExit):
        runpy.run_module("scripts.fleet_host", run_name="__main__")
    assert capsys.readouterr().out.startswith("usage:")
