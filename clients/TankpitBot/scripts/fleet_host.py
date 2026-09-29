"""Operate the fleet on sedona, the game host, from the hub.

Sedona owns every game server (MCPs board task 077204e8, 2026-09-28): the
fleet left diphtheria, and with it the public edge that used to live in
the MCPs workspace. ``make up`` and ``make down`` on the hub run this.

WHAT ``up`` STAGES, AND FROM WHERE. The host directory on sedona,
:data:`HOST_DIR`, holds exactly what ``docker-compose.yml`` mounts:

* the compose file and ``edge/nginx.conf`` as COMMITTED at HEAD, read with
  ``git show`` so an uncommitted edit on the hub never reaches the host;
* ``.env`` and ``accounts.json`` from the newest release, overwritten on
  every ``up`` so the host's copies always hash equal to the release's;
* ``data/tank_registry.json`` from the release only while the host has
  none, because the bot writes measured ranks back into it at runtime
  and the host's copy is then the newer one;
* ``edge.env``, the tunnel token, from :data:`SECRETS_FILE` on the hub,
  which no repository holds;
* ``host.env``, naming the release's image for compose to run.

The edge's images are pulled onto sedona through the hub's docker client
(:func:`pull_edge` says why not by compose there). The fleet's image is
built on sedona, from the release snapshot on the hub, only when that tag
is not there yet. Compose then runs ON sedona over ssh,
since the mounts are paths on sedona. Then the public origin is asked
for the filter's health and the fleet's demo roster, and last
:func:`scripts.fleet_gate.gate` spawns one bounded Practice bot and waits
for its first tick, so an ``up`` that exits 0 has been seen serving from
the internet AND playing the game.

``down`` stops the fleet only. The edge keeps answering, and the demo page
renders the fleet's absence as "offline", which is the truth.

Usage::

    python -m scripts.fleet_host up
    python -m scripts.fleet_host down
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Final

from scripts import _test_hooks as script_hooks
from scripts.fleet_gate import gate
from scripts.fleet_remote import SEDONA_SSH, FleetHostError, on_sedona, run_checked
from tankpit_bot import _test_hooks as core_hooks

#: The same host as a docker endpoint, for the image build.
SEDONA_DOCKER_HOST: Final[str] = f"ssh://{SEDONA_SSH}"

#: The fleet's directory on sedona: compose project, mounts and runs.
HOST_DIR: Final[str] = "C:/fleet/tankpit"

#: The hub's release ladder, which ``make release`` writes.
RELEASES_DIR: Final[Path] = Path("C:/Users/Test/PROJECTS/tankpit-releases")

#: The tunnel token, on the hub and in no repository.
SECRETS_FILE: Final[Path] = Path.home() / ".tankpit-secrets" / "edge.env"

#: Where the internet reaches the demo.
PUBLIC_ORIGIN: Final[str] = "https://tankpit.austinwagner.org"

#: How often, and how far apart, the public origin is asked before ``up``
#: gives up: a minute, enough for the tunnel to register its connections.
SMOKE_ATTEMPTS: Final[int] = 20
SMOKE_PAUSE_SECONDS: Final[float] = 3.0

#: The files staged from HEAD, relative to this package.
COMMITTED_FILES: Final[tuple[str, ...]] = ("docker-compose.yml", "edge/nginx.conf")

#: The files staged from the release, overwritten on every ``up``.
RELEASE_FILES: Final[tuple[str, ...]] = (".env", "accounts.json")

#: The runtime-written registry, staged only while the host has none.
REGISTRY_FILE: Final[str] = "data/tank_registry.json"

_USAGE: Final[str] = "usage: python -m scripts.fleet_host {up|down}\n"


def newest_release(releases_dir: Path) -> Path:
    """The release ``up`` runs: the most recently written one, as before the move.

    Args:
        releases_dir: The release ladder.

    Returns:
        The newest release directory.

    Raises:
        FleetHostError: FLEET_NO_RELEASE when the ladder holds none.
    """
    releases = (
        [entry for entry in releases_dir.iterdir() if entry.is_dir()]
        if releases_dir.is_dir()
        else []
    )
    if not releases:
        raise FleetHostError(
            f"FLEET_NO_RELEASE: {releases_dir} holds no release; run make release first"
        )
    return max(releases, key=_written_at)


def _written_at(entry: Path) -> float:
    """When a release directory was last written, the ladder's order.

    Args:
        entry: A release directory.

    Returns:
        Its modification time, in seconds since the epoch.
    """
    return entry.stat().st_mtime


def image_tag(release: Path) -> str:
    """The image a release runs as, the tag the hub's ``make up`` always used.

    Args:
        release: A release directory.

    Returns:
        ``tankpit-fleet:<release name>``.
    """
    return f"tankpit-fleet:{release.name}"


def _compose(*args: str) -> list[str]:
    """An ssh command line that runs compose on sedona against the host directory.

    Args:
        args: The compose subcommand and its arguments.

    Returns:
        The argv.
    """
    return [
        "ssh",
        SEDONA_SSH,
        "docker",
        "compose",
        "-f",
        f"{HOST_DIR}/docker-compose.yml",
        "--env-file",
        f"{HOST_DIR}/host.env",
        *args,
    ]


def stage(project_root: Path, release: Path, secrets_file: Path, scratch: Path) -> None:
    """Assemble the host directory in ``scratch``, then copy it to sedona's.

    The scratch copy is the host directory without its runtime state, so
    :func:`pull_edge` can hand the same compose project to the hub's docker
    client.

    Args:
        project_root: This package's directory, inside the git checkout.
        release: The release whose inputs are staged.
        secrets_file: The hub's tunnel token file.
        scratch: An empty directory to assemble the files in.

    Raises:
        FleetHostError: FLEET_SECRETS_MISSING when the token file is absent;
            FLEET_NOT_COMMITTED when HEAD lacks a committed file;
            FLEET_STAGE_FAILED when sedona refuses a directory or a copy.
    """
    if not secrets_file.is_file():
        raise FleetHostError(
            f"FLEET_SECRETS_MISSING: {secrets_file} is absent; "
            "it holds the tankpit tunnel's TUNNEL_TOKEN"
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
    shutil.copyfile(secrets_file, scratch / "edge.env")
    inputs = release / "clients" / "TankpitBot"
    for name in RELEASE_FILES:
        shutil.copyfile(inputs / name, scratch / name)
    # Assigned to $null rather than piped to Out-Null: ssh hands the line
    # to sedona's cmd.exe first, which would take the pipe for its own.
    directories = ", ".join(f"{HOST_DIR}/{name}" for name in ("edge", "data", "runs"))
    run_checked(
        on_sedona(f"$null = New-Item -ItemType Directory -Force -Path {directories}"),
        project_root,
        "FLEET_STAGE_FAILED",
    )
    top = [str(scratch / name) for name in ("docker-compose.yml", "host.env", "edge.env")]
    top += [str(scratch / name) for name in RELEASE_FILES]
    run_checked(
        ["scp", "-q", *top, f"{SEDONA_SSH}:{HOST_DIR}/"], project_root, "FLEET_STAGE_FAILED"
    )
    run_checked(
        ["scp", "-q", str(scratch / "edge" / "nginx.conf"), f"{SEDONA_SSH}:{HOST_DIR}/edge/"],
        project_root,
        "FLEET_STAGE_FAILED",
    )
    registry = inputs / REGISTRY_FILE
    held = run_checked(
        on_sedona(f"Test-Path {HOST_DIR}/{REGISTRY_FILE}"), project_root, "FLEET_STAGE_FAILED"
    ).strip()
    if held == "False" and registry.is_file():
        run_checked(
            ["scp", "-q", str(registry), f"{SEDONA_SSH}:{HOST_DIR}/data/"],
            project_root,
            "FLEET_STAGE_FAILED",
        )


def pull_edge(project_root: Path, scratch: Path) -> None:
    """Pull the edge's images onto sedona through the hub's docker client.

    Not by compose on sedona: Docker Desktop's credential helper cannot
    reach the Windows credential store from an ssh logon session, so a
    pull there fails with "A specified logon session does not exist"
    (measured 2026-09-28). Through ``--host`` the daemon on sedona pulls
    while the hub answers for credentials, and the image names stay in
    the compose file alone.

    Args:
        project_root: Where to run docker from.
        scratch: The host directory :func:`stage` assembled.

    Raises:
        FleetHostError: FLEET_PULL_FAILED when a pull fails.
    """
    pull = [
        "docker",
        "--host",
        SEDONA_DOCKER_HOST,
        "compose",
        "-f",
        str(scratch / "docker-compose.yml"),
    ]
    pull += [
        "--env-file",
        str(scratch / "host.env"),
        "--profile",
        "edge",
        "pull",
        "public",
        "tunnel",
    ]
    run_checked(pull, project_root, "FLEET_PULL_FAILED")


def ensure_image(project_root: Path, release: Path) -> bool:
    """Build the release's image on sedona unless the tag is already there.

    Args:
        project_root: Where to run docker from.
        release: The release snapshot, the build context.

    Returns:
        True when it was built now, False when sedona already had it.

    Raises:
        FleetHostError: FLEET_IMAGE_BUILD_FAILED when the build fails.
    """
    tag = image_tag(release)
    held = script_hooks.run_command(
        ["docker", "--host", SEDONA_DOCKER_HOST, "image", "inspect", tag], project_root
    )
    if held["returncode"] == 0:
        return False
    dockerfile = release / "clients" / "TankpitBot" / "Dockerfile"
    build = [
        "docker",
        "--host",
        SEDONA_DOCKER_HOST,
        "build",
        "-f",
        str(dockerfile),
        "-t",
        tag,
        "--build-arg",
        f"BUILD_REF={release.name}",
        str(release),
    ]
    run_checked(build, project_root, "FLEET_IMAGE_BUILD_FAILED")
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


def up(project_root: Path, releases_dir: Path, secrets_file: Path) -> list[str]:
    """Run the newest release on sedona with its public edge, and see it serve.

    Args:
        project_root: This package's directory, inside the git checkout.
        releases_dir: The release ladder.
        secrets_file: The hub's tunnel token file.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: As :func:`newest_release`, :func:`stage`,
            :func:`pull_edge`, :func:`ensure_image`, :func:`smoke` and
            :func:`scripts.fleet_gate.gate`; FLEET_COMPOSE_FAILED when
            compose refuses.
    """
    release = newest_release(releases_dir)
    with tempfile.TemporaryDirectory() as scratch:
        stage(project_root, release, secrets_file, Path(scratch))
        pull_edge(project_root, Path(scratch))
    built = ensure_image(project_root, release)
    run_checked(
        _compose("--profile", "edge", "up", "-d", "--no-build", "fleet", "public", "tunnel"),
        project_root,
        "FLEET_COMPOSE_FAILED",
    )
    smoke(PUBLIC_ORIGIN)
    return [
        f"release {release.name} staged into {SEDONA_SSH}:{HOST_DIR}",
        f"image {image_tag(release)} {'built on sedona' if built else 'already on sedona'}",
        f"fleet and edge up; {PUBLIC_ORIGIN} serves /healthz and /demo/fleet",
        gate(project_root),
    ]


def down(project_root: Path) -> list[str]:
    """Drain and stop the fleet on sedona, leaving the edge answering.

    Args:
        project_root: Where to run ssh from.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: FLEET_COMPOSE_FAILED when compose refuses.
    """
    run_checked(_compose("stop", "fleet"), project_root, "FLEET_COMPOSE_FAILED")
    return [f"fleet stopped on {SEDONA_SSH}; the edge still answers, and the demo shows it offline"]


def run(
    arguments: list[str], project_root: Path, releases_dir: Path, secrets_file: Path
) -> list[str]:
    """Do what the command line asks.

    Args:
        arguments: The arguments after the program name.
        project_root: This package's directory, inside the git checkout.
        releases_dir: The release ladder.
        secrets_file: The hub's tunnel token file.

    Returns:
        The lines reporting what was done.

    Raises:
        SystemExit: 2 with the usage for anything but ``up`` or ``down``.
        FleetHostError: As :func:`up` and :func:`down`.
    """
    if arguments == ["up"]:
        return up(project_root, releases_dir, secrets_file)
    if arguments == ["down"]:
        return down(project_root)
    sys.stdout.write(_USAGE)
    raise SystemExit(2)


def main() -> None:
    """Entry point, run from this package's directory against the hub's ladder.

    Raises:
        SystemExit: As :func:`run`.
        FleetHostError: As :func:`run`.
    """
    for line in run(list(core_hooks.get_argv())[1:], Path.cwd(), RELEASES_DIR, SECRETS_FILE):
        sys.stdout.write(f"{line}\n")


if __name__ == "__main__":
    main()


__all__ = [
    "COMMITTED_FILES",
    "HOST_DIR",
    "PUBLIC_ORIGIN",
    "REGISTRY_FILE",
    "RELEASES_DIR",
    "RELEASE_FILES",
    "SECRETS_FILE",
    "SEDONA_DOCKER_HOST",
    "SMOKE_ATTEMPTS",
    "SMOKE_PAUSE_SECONDS",
    "down",
    "ensure_image",
    "image_tag",
    "main",
    "newest_release",
    "pull_edge",
    "run",
    "smoke",
    "stage",
    "up",
]
