"""Tests for operating the fleet on sedona from the hub.

Every command the script runs goes through ``scripts._test_hooks.run_command``,
here a recorder that answers by the command's words and keeps every argv, so
each test asserts the exact commands sent to sedona. The release ladder and
the secrets file are real files under ``tmp_path``.
"""

from __future__ import annotations

import os
import runpy
from collections.abc import Generator
from pathlib import Path

import pytest
from scripts.fleet_host import (
    HOST_DIR,
    PUBLIC_ORIGIN,
    SEDONA_DOCKER_HOST,
    SEDONA_SSH,
    SMOKE_ATTEMPTS,
    SMOKE_PAUSE_SECONDS,
    FleetHostError,
    down,
    ensure_image,
    image_tag,
    newest_release,
    pull_edge,
    run,
    smoke,
    stage,
    up,
)

from scripts import _test_hooks as script_hooks
from scripts import fleet_host
from tankpit_bot import _test_hooks as core_hooks

COMPOSE = f"{HOST_DIR}/docker-compose.yml"
HOST_ENV = f"{HOST_DIR}/host.env"


class Recorder:
    """A command runner that answers from a script and records every argv."""

    def __init__(self, answers: dict[str, script_hooks.CommandResult]) -> None:
        """Hold the answers.

        Args:
            answers: Keyed by a word the command line contains; the first
                key found in the joined argv answers. Unmatched commands
                succeed with empty output.
        """
        self.answers = answers
        self.calls: list[list[str]] = []
        self.staged: dict[str, str] = {}

    def __call__(self, argv: list[str], cwd: Path) -> script_hooks.CommandResult:
        """Record one command and answer it.

        Args:
            argv: The command.
            cwd: Ignored; every command runs from the package directory.

        Returns:
            The scripted answer.
        """
        self.calls.append(argv)
        if argv[0] == "scp":
            for source in argv[2:-1]:
                self.staged[Path(source).name] = Path(source).read_text(encoding="utf-8")
        line = " ".join(argv)
        for word, answer in self.answers.items():
            if word in line:
                return answer
        return script_hooks.CommandResult(returncode=0, stdout="", stderr="")


class Answer:
    """One HTTP answer with a status and no body."""

    def __init__(self, status_code: int) -> None:
        """Hold the status.

        Args:
            status_code: The answer's status.
        """
        self.status_code = status_code
        self.content = b""


def _ok(stdout: str) -> script_hooks.CommandResult:
    return script_hooks.CommandResult(returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str) -> script_hooks.CommandResult:
    return script_hooks.CommandResult(returncode=1, stdout="", stderr=stderr)


def _release(ladder: Path, name: str, *, registry: bool) -> Path:
    inputs = ladder / name / "clients" / "TankpitBot"
    (inputs / "data").mkdir(parents=True)
    (inputs / ".env").write_text("TANKPIT_URL=x\n", encoding="utf-8")
    (inputs / "accounts.json").write_text("[]\n", encoding="utf-8")
    if registry:
        (inputs / "data" / "tank_registry.json").write_text("{}\n", encoding="utf-8")
    return ladder / name


def _secrets(tmp_path: Path) -> Path:
    secrets = tmp_path / "secrets" / "edge.env"
    secrets.parent.mkdir()
    secrets.write_text("TUNNEL_TOKEN=t\n", encoding="utf-8")
    return secrets


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
    """The ladder's newest directory by write time is the release, and a stray file is not."""
    older = _release(tmp_path, "v0.1.0-aaaaaaaa", registry=False)
    newer = _release(tmp_path, "v0.1.0-bbbbbbbb", registry=False)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    assert newest_release(tmp_path) == newer
    assert image_tag(newer) == "tankpit-fleet:v0.1.0-bbbbbbbb"


@pytest.mark.parametrize("ladder", ["missing", "empty"])
def test_newest_release_refuses_a_ladder_without_one(tmp_path: Path, ladder: str) -> None:
    """No release, or no ladder at all, is refused by name."""
    root = tmp_path / ladder
    if ladder == "empty":
        root.mkdir()
    with pytest.raises(FleetHostError) as raised:
        newest_release(root)
    assert str(raised.value) == f"FLEET_NO_RELEASE: {root} holds no release; run make release first"


@pytest.mark.parametrize(("sedona_has_registry", "copied"), [("False", True), ("True", False)])
def test_stage_sends_committed_files_release_inputs_and_the_token(
    tmp_path: Path, sedona_has_registry: str, copied: bool
) -> None:
    """Stage sends HEAD's files, the release's inputs and the token.

    The registry goes only to a host that has none.
    """
    release = _release(tmp_path / "ladder", "v0.1.0-cccccccc", registry=True)
    secrets = _secrets(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recorder = Recorder(
        {
            "git show HEAD:./docker-compose.yml": _ok("name: tankpitbot\n"),
            "git show HEAD:./edge/nginx.conf": _ok("server {}\n"),
            "Test-Path": _ok(f"{sedona_has_registry}\r\n"),
        }
    )
    script_hooks.run_command = recorder
    stage(tmp_path, release, secrets, scratch)
    inputs = release / "clients" / "TankpitBot"
    expected = [
        ["git", "show", "HEAD:./docker-compose.yml"],
        ["git", "show", "HEAD:./edge/nginx.conf"],
        [
            "ssh",
            SEDONA_SSH,
            "powershell",
            "-NoProfile",
            "-Command",
            "$null = New-Item -ItemType Directory -Force -Path "
            f"{HOST_DIR}/edge, {HOST_DIR}/data, {HOST_DIR}/runs",
        ],
        [
            "scp",
            "-q",
            str(scratch / "docker-compose.yml"),
            str(scratch / "host.env"),
            str(scratch / "edge.env"),
            str(scratch / ".env"),
            str(scratch / "accounts.json"),
            f"{SEDONA_SSH}:{HOST_DIR}/",
        ],
        ["scp", "-q", str(scratch / "edge" / "nginx.conf"), f"{SEDONA_SSH}:{HOST_DIR}/edge/"],
        [
            "ssh",
            SEDONA_SSH,
            "powershell",
            "-NoProfile",
            "-Command",
            f"Test-Path {HOST_DIR}/data/tank_registry.json",
        ],
    ]
    if copied:
        expected.append(
            [
                "scp",
                "-q",
                str(inputs / "data" / "tank_registry.json"),
                f"{SEDONA_SSH}:{HOST_DIR}/data/",
            ]
        )
    assert recorder.calls == expected
    assert recorder.staged["docker-compose.yml"] == "name: tankpitbot\n"
    assert recorder.staged["nginx.conf"] == "server {}\n"
    assert recorder.staged["host.env"] == "FLEET_IMAGE=tankpit-fleet:v0.1.0-cccccccc\n"
    assert recorder.staged["edge.env"] == "TUNNEL_TOKEN=t\n"
    assert recorder.staged[".env"] == "TANKPIT_URL=x\n"
    assert recorder.staged["accounts.json"] == "[]\n"
    assert secrets.read_text(encoding="utf-8") == "TUNNEL_TOKEN=t\n"


def test_stage_skips_a_registry_the_release_does_not_carry(tmp_path: Path) -> None:
    """A release without a registry sends none, even to a host without one."""
    release = _release(tmp_path / "ladder", "v0.1.0-dddddddd", registry=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recorder = Recorder({"Test-Path": _ok("False\r\n")})
    script_hooks.run_command = recorder
    stage(tmp_path, release, _secrets(tmp_path), scratch)
    assert recorder.calls[-1][-1] == f"Test-Path {HOST_DIR}/data/tank_registry.json"


def test_stage_refuses_without_the_token(tmp_path: Path) -> None:
    """No token file means no tunnel, refused before anything is sent."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    missing = tmp_path / "edge.env"
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, tmp_path, missing, tmp_path)
    assert str(raised.value) == (
        f"FLEET_SECRETS_MISSING: {missing} is absent; it holds the tankpit tunnel's TUNNEL_TOKEN"
    )
    assert recorder.calls == []


def test_stage_refuses_a_file_head_does_not_hold(tmp_path: Path) -> None:
    """A compose file missing from HEAD is refused with git's own words."""
    script_hooks.run_command = Recorder(
        {"git show": _fail("fatal: path 'docker-compose.yml' does not exist in 'HEAD'\n")}
    )
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, tmp_path, _secrets(tmp_path), tmp_path)
    assert str(raised.value) == (
        "FLEET_NOT_COMMITTED: git show HEAD:./docker-compose.yml exited 1: "
        "fatal: path 'docker-compose.yml' does not exist in 'HEAD'"
    )


def test_stage_refuses_when_sedona_refuses_a_copy(tmp_path: Path) -> None:
    """A copy sedona refuses stops the stage by name."""
    release = _release(tmp_path / "ladder", "v0.1.0-eeeeeeee", registry=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    script_hooks.run_command = Recorder({"scp": _fail("scp: permission denied")})
    with pytest.raises(FleetHostError) as raised:
        stage(tmp_path, release, _secrets(tmp_path), scratch)
    sent = [scratch / name for name in ("docker-compose.yml", "host.env", "edge.env", ".env")]
    sent.append(scratch / "accounts.json")
    assert str(raised.value) == (
        f"FLEET_STAGE_FAILED: scp -q {' '.join(str(path) for path in sent)} "
        f"{SEDONA_SSH}:{HOST_DIR}/ exited 1: scp: permission denied"
    )


def test_pull_edge_pulls_through_the_hubs_client(tmp_path: Path) -> None:
    """The edge's images are pulled by the staged compose file, against sedona's daemon."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    pull_edge(tmp_path, tmp_path / "scratch")
    assert recorder.calls == [
        ["docker", "--host", SEDONA_DOCKER_HOST, "compose",
         "-f", str(tmp_path / "scratch" / "docker-compose.yml"),
         "--env-file", str(tmp_path / "scratch" / "host.env"),
         "--profile", "edge", "pull", "public", "tunnel"],
    ]  # fmt: skip


def test_pull_edge_refuses_a_failed_pull(tmp_path: Path) -> None:
    """A pull the registry refuses is refused with the pull's error."""
    script_hooks.run_command = Recorder({" pull ": _fail("manifest unknown")})
    with pytest.raises(FleetHostError) as raised:
        pull_edge(tmp_path, tmp_path)
    assert str(raised.value) == (
        f"FLEET_PULL_FAILED: docker --host {SEDONA_DOCKER_HOST} compose "
        f"-f {tmp_path / 'docker-compose.yml'} --env-file {tmp_path / 'host.env'} "
        "--profile edge pull public tunnel exited 1: manifest unknown"
    )


def test_ensure_image_builds_only_a_missing_tag(tmp_path: Path) -> None:
    """A tag sedona holds is not rebuilt; a missing one is built from the snapshot."""
    release = tmp_path / "v0.1.0-ffffffff"
    held = Recorder({})
    script_hooks.run_command = held
    assert ensure_image(tmp_path, release) is False
    assert held.calls == [
        [
            "docker",
            "--host",
            SEDONA_DOCKER_HOST,
            "image",
            "inspect",
            "tankpit-fleet:v0.1.0-ffffffff",
        ]
    ]
    missing = Recorder({"image inspect": _fail("No such image")})
    script_hooks.run_command = missing
    assert ensure_image(tmp_path, release) is True
    assert missing.calls[1] == [
        "docker", "--host", SEDONA_DOCKER_HOST, "build",
        "-f", str(release / "clients" / "TankpitBot" / "Dockerfile"),
        "-t", "tankpit-fleet:v0.1.0-ffffffff",
        "--build-arg", "BUILD_REF=v0.1.0-ffffffff", str(release),
    ]  # fmt: skip


def test_ensure_image_refuses_a_failed_build(tmp_path: Path) -> None:
    """A build that fails is refused with the build's error."""
    script_hooks.run_command = Recorder(
        {"image inspect": _fail("No such image"), " build ": _fail("no space left")}
    )
    release = tmp_path / "v0.1.0-00000000"
    with pytest.raises(FleetHostError) as raised:
        ensure_image(tmp_path, release)
    dockerfile = release / "clients" / "TankpitBot" / "Dockerfile"
    assert str(raised.value) == (
        f"FLEET_IMAGE_BUILD_FAILED: docker --host {SEDONA_DOCKER_HOST} build -f {dockerfile} "
        f"-t tankpit-fleet:v0.1.0-00000000 --build-arg BUILD_REF=v0.1.0-00000000 {release} "
        "exited 1: no space left"
    )


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


def test_up_stages_builds_composes_and_smokes(tmp_path: Path) -> None:
    """Up runs the newest release with its edge on sedona and reports each step."""
    release = _release(tmp_path / "ladder", "v0.1.0-12345678", registry=False)
    recorder = Recorder({"image inspect": _fail("No such image"), "Test-Path": _ok("True\r\n")})
    script_hooks.run_command = recorder
    script_hooks.http_get = _serving
    lines = up(tmp_path, tmp_path / "ladder", _secrets(tmp_path))
    verbs = [call[3] if call[0] == "docker" else call[0] for call in recorder.calls]
    assert verbs == ["git", "git", "ssh", "scp", "scp", "ssh", "compose", "image", "build", "ssh"]
    assert recorder.calls[6][-5:] == ["--profile", "edge", "pull", "public", "tunnel"]
    assert recorder.calls[-1] == [
        "ssh",
        SEDONA_SSH,
        "docker",
        "compose",
        "-f",
        COMPOSE,
        "--env-file",
        HOST_ENV,
        "--profile",
        "edge",
        "up",
        "-d",
        "--no-build",
        "fleet",
        "public",
        "tunnel",
    ]
    assert lines == [
        f"release {release.name} staged into {SEDONA_SSH}:{HOST_DIR}",
        "image tankpit-fleet:v0.1.0-12345678 built on sedona",
        f"fleet and edge up; {PUBLIC_ORIGIN} serves /healthz and /demo/fleet",
    ]


def test_up_refuses_when_compose_refuses(tmp_path: Path) -> None:
    """Compose refusing on sedona is refused by name, before any smoke."""
    _release(tmp_path / "ladder", "v0.1.0-87654321", registry=False)
    script_hooks.run_command = Recorder(
        {"docker compose": _fail("edge.env: no such file"), "Test-Path": _ok("True\r\n")}
    )
    with pytest.raises(FleetHostError) as raised:
        up(tmp_path, tmp_path / "ladder", _secrets(tmp_path))
    assert str(raised.value) == (
        f"FLEET_COMPOSE_FAILED: ssh {SEDONA_SSH} docker compose -f {COMPOSE} --env-file {HOST_ENV} "
        "--profile edge up -d --no-build fleet public tunnel exited 1: edge.env: no such file"
    )


def test_down_stops_the_fleet_only(tmp_path: Path) -> None:
    """Down drains the fleet and leaves the edge up."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    assert down(tmp_path) == [
        f"fleet stopped on {SEDONA_SSH}; the edge still answers, and the demo shows it offline"
    ]
    assert recorder.calls == [
        [
            "ssh",
            SEDONA_SSH,
            "docker",
            "compose",
            "-f",
            COMPOSE,
            "--env-file",
            HOST_ENV,
            "stop",
            "fleet",
        ]
    ]


def test_main_runs_down_and_prints_its_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """``down`` from the command line prints what it did."""
    script_hooks.run_command = Recorder({})
    core_hooks.get_argv = lambda: ["fleet_host", "down"]
    fleet_host.main()
    assert (
        capsys.readouterr().out
        == f"fleet stopped on {SEDONA_SSH}; the edge still answers, and the demo shows it offline\n"
    )


def test_run_up_reads_the_ladder_and_secrets_it_is_given(tmp_path: Path) -> None:
    """``up`` on the command line runs the newest release of the ladder it is handed."""
    _release(tmp_path / "ladder", "v0.1.0-abcdef12", registry=False)
    script_hooks.run_command = Recorder({"Test-Path": _ok("True\r\n")})
    script_hooks.http_get = _serving
    lines = run(["up"], tmp_path, tmp_path / "ladder", _secrets(tmp_path))
    assert lines[1] == "image tankpit-fleet:v0.1.0-abcdef12 already on sedona"


@pytest.mark.parametrize("arguments", [[], ["sideways"], ["up", "down"]])
def test_run_refuses_anything_else_with_the_usage(
    tmp_path: Path, arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Any other argument list prints the usage and exits 2, running nothing."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    with pytest.raises(SystemExit) as exited:
        run(arguments, tmp_path, tmp_path, tmp_path)
    assert exited.value.code == 2
    assert capsys.readouterr().out == "usage: python -m scripts.fleet_host {up|down}\n"
    assert recorder.calls == []


def test_module_runs_as_a_program(capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m scripts.fleet_host`` reaches main."""
    core_hooks.get_argv = lambda: ["fleet_host"]
    with pytest.raises(SystemExit):
        runpy.run_module("scripts.fleet_host", run_name="__main__")
    assert capsys.readouterr().out.startswith("usage:")
