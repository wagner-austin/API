"""Operate the fleet on the game host, colossus, over plain ssh.

The game host owns the fleet and its public edge (MCPs board task 077204e8
gave them to sedona on 2026-09-28; board task c481c259 moved them to
colossus, a Linux machine, on 2026-10-18). ``make up`` and ``make down`` run
this from any Linux checkout that can ssh to it.

THE HOST HOLDS EVERYTHING. Its directory, :data:`HOST_DIR`, is the compose
project and every mount. Three of its files are the host's own and no
repository's: ``.env``, ``accounts.json`` and ``edge.env`` (the tunnel
token), copied once from sedona; ``up`` refuses a host that lacks one,
naming it. Nothing is read from the hub any more: austinpc, which held
the release ladder, the token and the gitignored accounts, is being
retired.

WHAT ``up`` STAGES. ``data/tank_registry.json`` from the release only while
the host has none, because the bots write measured ranks back into it at
runtime and the host's copy is then the newer one; the compose file and
``edge/nginx.conf`` as COMMITTED at HEAD, read with ``git show`` so an
uncommitted edit never reaches the host; and ``host.env``, naming the
release's image for compose to run.

WHICH IMAGE. The newest release on the host's ladder,
:data:`RELEASES_DIR`, which :mod:`scripts.fleet_release` fills. Its image
is built on the host from that directory, only when the tag is not there
yet. Compose then pulls the edge's images and runs the fleet and the edge
on the host. The public origin is asked for the filter's health and the
fleet's demo roster, and last :func:`scripts.fleet_gate.gate` spawns one
bounded Practice bot and waits for its first tick, so an ``up`` that exits
0 has been seen serving from the internet AND playing the game.

``down`` stops the fleet only. The edge keeps answering, and the demo page
renders the fleet's absence as "offline", which is the truth.

Usage::

    python -m scripts.fleet_host up
    python -m scripts.fleet_host down
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Final

from scripts import _test_hooks as script_hooks
from scripts.fleet_gate import gate
from scripts.fleet_remote import (
    GAME_HOST_HOME,
    GAME_HOST_SSH,
    FleetHostError,
    on_host,
    run_checked,
)
from tankpit_bot import _test_hooks as core_hooks

#: The fleet's directory on the host: compose project, mounts and runs.
HOST_DIR: Final[str] = f"{GAME_HOST_HOME}/tankpit"

#: The host's release ladder, which ``make release`` fills.
RELEASES_DIR: Final[str] = f"{GAME_HOST_HOME}/tankpit-releases"

#: Where the internet reaches the demo.
PUBLIC_ORIGIN: Final[str] = "https://tankpit.austinwagner.org"

#: How often, and how far apart, the public origin is asked before ``up``
#: gives up: a minute, enough for the tunnel to register its connections.
SMOKE_ATTEMPTS: Final[int] = 20
SMOKE_PAUSE_SECONDS: Final[float] = 3.0

#: The files staged from HEAD, relative to this package.
COMMITTED_FILES: Final[tuple[str, ...]] = ("docker-compose.yml", "edge/nginx.conf")

#: The runtime-written registry, relative to the package and to :data:`HOST_DIR`.
REGISTRY_FILE: Final[str] = "data/tank_registry.json"

#: The files the host must hold, relative to :data:`HOST_DIR`.
HOST_FILES: Final[tuple[str, ...]] = (".env", "accounts.json", "edge.env", REGISTRY_FILE)

_USAGE: Final[str] = "usage: python -m scripts.fleet_host {up|down}\n"


def newest_release(project_root: Path) -> str:
    """The release ``up`` runs: the most recently written one on the host's ladder.

    Args:
        project_root: Where to run ssh from.

    Returns:
        The newest release's directory name.

    Raises:
        FleetHostError: FLEET_NO_RELEASE when the ladder cannot be read or
            holds none.
    """
    listing = run_checked(
        on_host(
            ["find", RELEASES_DIR, "-mindepth", "1", "-maxdepth", "1", "-type", "d", "-printf",
             "%T@ %f\n"]
        ),
        project_root,
        "FLEET_NO_RELEASE",
    )  # fmt: skip
    releases = [line.split(" ", 1) for line in listing.splitlines() if line]
    if not releases:
        raise FleetHostError(
            f"FLEET_NO_RELEASE: {GAME_HOST_SSH}:{RELEASES_DIR} holds no release; "
            "run make release first"
        )
    return max(releases, key=_written_at)[1]


def _written_at(entry: list[str]) -> float:
    """When a release directory was last written, the ladder's order.

    Args:
        entry: ``[seconds since the epoch, name]`` as ``find`` printed them.

    Returns:
        The modification time.
    """
    return float(entry[0])


def image_tag(release: str) -> str:
    """The image a release runs as.

    Args:
        release: A release directory's name.

    Returns:
        ``tankpit-fleet:<release name>``.
    """
    return f"tankpit-fleet:{release}"


def _compose(*args: str) -> list[str]:
    """An ssh command line that runs compose on the host against the host directory.

    Args:
        args: The compose subcommand and its arguments.

    Returns:
        The argv.
    """
    return on_host(
        [
            "docker",
            "compose",
            "-f",
            f"{HOST_DIR}/docker-compose.yml",
            "--env-file",
            f"{HOST_DIR}/host.env",
            *args,
        ]
    )


def stage(project_root: Path, release: str, scratch: Path) -> None:
    """Seed the registry, refuse a host without its own files, then send HEAD's files.

    The tank registry is committed, so the release carries one; it is
    copied in only while the host has none, because the bots write their
    measured ranks back into the host's copy, which is then the newer.

    Args:
        project_root: This package's directory, inside the git checkout.
        release: The release whose image ``host.env`` names.
        scratch: An empty directory to assemble the files in.

    Raises:
        FleetHostError: FLEET_HOST_FILES_MISSING naming what the host lacks;
            FLEET_NOT_COMMITTED when HEAD lacks a committed file;
            FLEET_STAGE_FAILED when the host refuses a directory or a copy.
    """
    run_checked(
        on_host(["mkdir", "-p", *(f"{HOST_DIR}/{name}" for name in ("edge", "data", "runs"))]),
        project_root,
        "FLEET_STAGE_FAILED",
    )
    run_checked(
        on_host(
            ["cp", "--update=none",
             f"{RELEASES_DIR}/{release}/clients/TankpitBot/{REGISTRY_FILE}",
             f"{HOST_DIR}/{REGISTRY_FILE}"]
        ),
        project_root,
        "FLEET_STAGE_FAILED",
    )  # fmt: skip
    run_checked(
        on_host(["ls", "-1", *(f"{HOST_DIR}/{name}" for name in HOST_FILES)]),
        project_root,
        "FLEET_HOST_FILES_MISSING",
    )
    (scratch / "edge").mkdir()
    for name in COMMITTED_FILES:
        committed = run_checked(
            ["git", "show", f"HEAD:./{name}"], project_root, "FLEET_NOT_COMMITTED"
        )
        (scratch / name).write_text(committed, encoding="utf-8", newline="")
    (scratch / "host.env").write_text(
        f"FLEET_IMAGE={image_tag(release)}\n", encoding="utf-8", newline=""
    )
    run_checked(
        ["scp", "-q", str(scratch / "docker-compose.yml"), str(scratch / "host.env"),
         f"{GAME_HOST_SSH}:{HOST_DIR}/"],
        project_root,
        "FLEET_STAGE_FAILED",
    )  # fmt: skip
    run_checked(
        ["scp", "-q", str(scratch / "edge" / "nginx.conf"), f"{GAME_HOST_SSH}:{HOST_DIR}/edge/"],
        project_root,
        "FLEET_STAGE_FAILED",
    )


def ensure_image(project_root: Path, release: str) -> bool:
    """Build the release's image on the host unless the tag is already there.

    Args:
        project_root: Where to run ssh from.
        release: The release's directory name on the host's ladder.

    Returns:
        True when it was built now, False when the host already had it.

    Raises:
        FleetHostError: FLEET_IMAGE_BUILD_FAILED when the build fails.
    """
    tag = image_tag(release)
    held = script_hooks.run_command(on_host(["docker", "image", "inspect", tag]), project_root)
    if held["returncode"] == 0:
        return False
    context = f"{RELEASES_DIR}/{release}"
    build = [
        "docker",
        "build",
        "-f",
        f"{context}/clients/TankpitBot/Dockerfile",
        "-t",
        tag,
        "--build-arg",
        f"BUILD_REF={release}",
        context,
    ]
    run_checked(on_host(build), project_root, "FLEET_IMAGE_BUILD_FAILED")
    return True


def smoke(origin: str) -> None:
    """Wait for the public origin to serve the filter's health and the demo roster.

    Args:
        origin: The public origin.

    Raises:
        FleetHostError: FLEET_SMOKE_FAILED naming the last answers, when
            :data:`SMOKE_ATTEMPTS` asks never saw both.
    """
    last = ""
    for attempt in range(SMOKE_ATTEMPTS):
        if attempt > 0:
            script_hooks.sleep_seconds(SMOKE_PAUSE_SECONDS)
        health = script_hooks.http_get(f"{origin}/healthz").status_code
        roster = script_hooks.http_get(f"{origin}/demo/fleet").status_code
        if health == 204 and roster == 200:
            return
        last = f"/healthz answered {health}, /demo/fleet answered {roster}"
    raise FleetHostError(f"FLEET_SMOKE_FAILED: {origin} after {SMOKE_ATTEMPTS} attempts: {last}")


def up(project_root: Path) -> list[str]:
    """Run the newest release on the host with its public edge, and see it serve.

    Args:
        project_root: This package's directory, inside the git checkout.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: As :func:`newest_release`, :func:`stage`,
            :func:`ensure_image`, :func:`smoke` and
            :func:`scripts.fleet_gate.gate`; FLEET_PULL_FAILED when the
            edge's images cannot be pulled; FLEET_COMPOSE_FAILED when
            compose refuses.
    """
    release = newest_release(project_root)
    with tempfile.TemporaryDirectory() as scratch:
        stage(project_root, release, Path(scratch))
    run_checked(
        _compose("--profile", "edge", "pull", "public", "tunnel"), project_root, "FLEET_PULL_FAILED"
    )
    built = ensure_image(project_root, release)
    run_checked(
        _compose("--profile", "edge", "up", "-d", "--no-build", "fleet", "public", "tunnel"),
        project_root,
        "FLEET_COMPOSE_FAILED",
    )
    smoke(PUBLIC_ORIGIN)
    return [
        f"release {release} staged into {GAME_HOST_SSH}:{HOST_DIR}",
        f"image {image_tag(release)} {'built on the host' if built else 'already on the host'}",
        f"fleet and edge up; {PUBLIC_ORIGIN} serves /healthz and /demo/fleet",
        gate(project_root),
    ]


def down(project_root: Path) -> list[str]:
    """Drain and stop the fleet on the host, leaving the edge answering.

    Args:
        project_root: Where to run ssh from.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: FLEET_COMPOSE_FAILED when compose refuses.
    """
    run_checked(_compose("stop", "fleet"), project_root, "FLEET_COMPOSE_FAILED")
    return [
        f"fleet stopped on {GAME_HOST_SSH}; the edge still answers, and the demo shows it offline"
    ]


def run(arguments: list[str], project_root: Path) -> list[str]:
    """Do what the command line asks.

    Args:
        arguments: The arguments after the program name.
        project_root: This package's directory, inside the git checkout.

    Returns:
        The lines reporting what was done.

    Raises:
        SystemExit: 2 with the usage for anything but ``up`` or ``down``.
        FleetHostError: As :func:`up` and :func:`down`.
    """
    if arguments == ["up"]:
        return up(project_root)
    if arguments == ["down"]:
        return down(project_root)
    sys.stdout.write(_USAGE)
    raise SystemExit(2)


def main() -> None:
    """Entry point, run from this package's directory.

    Raises:
        SystemExit: As :func:`run`.
        FleetHostError: As :func:`run`.
    """
    for line in run(list(core_hooks.get_argv())[1:], Path.cwd()):
        sys.stdout.write(f"{line}\n")


if __name__ == "__main__":
    main()


__all__ = [
    "COMMITTED_FILES",
    "HOST_DIR",
    "HOST_FILES",
    "PUBLIC_ORIGIN",
    "REGISTRY_FILE",
    "RELEASES_DIR",
    "SMOKE_ATTEMPTS",
    "SMOKE_PAUSE_SECONDS",
    "down",
    "ensure_image",
    "image_tag",
    "main",
    "newest_release",
    "run",
    "smoke",
    "stage",
    "up",
]
