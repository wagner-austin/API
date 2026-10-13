"""Operate the sim server on sedona, the game host, from the hub.

Sedona owns every game server (MCPs board task 077204e8; the operator,
2026-09-28: "sedona can own all the game servwrs. tankpit too"), and since
MCPs board task ff035b0d that includes the sim server, which until then ran
on the hub beside the API root compose's Traefik and platform-postgres.
``make sim-up`` and ``make sim-down`` on the hub run this.

WHAT ``up`` STAGES, AND FROM WHERE. The host directory on sedona,
:data:`SIM_DIR`, holds ``sim-server.compose.json`` as COMMITTED at HEAD and
``sim.env``, which names the image and carries the database password from
:data:`SECRETS_FILE` on the hub, which no repository holds.

WHICH IMAGE. The sim runs ``tankpit-sim:<HEAD's short sha>``, built on
sedona from a ``git archive`` of HEAD only when sedona lacks that tag, so an
uncommitted edit never reaches it. It is not the fleet's release image:
``make up`` runs the newest release, and a release cut for the sim would roll
the fleet at the next ``up``.

IT RUNS ONLY WHEN ASKED. The operator, 2026-10-13: "make sure it doesnt auro
stsrt or anything i csn maniallt stsrt it wheb needed". Both services carry
``restart: "no"``, so a reboot of sedona or of its Docker leaves them off,
and ``down`` stops both; ``up`` is the one way they start.

WHERE IT ANSWERS. The sim publishes on sedona's loopback alone, as it was
reached on the hub's before the move, and accounts live in the ``sim-db``
volume beside it, which outlives every ``down``. ``up`` ends by asking that
loopback, over ssh, for room 1's terrain, which the server reads off the
field it plays: an ``up`` that exits 0 has been seen serving.

Usage::

    python -m scripts.sim_host up
    python -m scripts.sim_host down
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Final

from scripts import _test_hooks as script_hooks
from scripts.fleet_remote import SEDONA_SSH, FleetHostError, on_sedona, run_checked
from tankpit_bot import _test_hooks as core_hooks

#: The same host as a docker endpoint, for the image build and the pull.
SEDONA_DOCKER_HOST: Final[str] = f"ssh://{SEDONA_SSH}"

#: The sim's directory on sedona.
SIM_DIR: Final[str] = "C:/fleet/tankpit-sim"

#: The compose file, relative to this package, staged from HEAD.
COMPOSE_FILE: Final[str] = "sim-server.compose.json"

#: The database password, on the hub and in no repository.
SECRETS_FILE: Final[Path] = Path.home() / ".tankpit-secrets" / "sim.env"

#: What the archive the image is built from holds, relative to the repo root.
ARCHIVE_PATHS: Final[tuple[str, ...]] = (
    "clients/TankpitBot",
    "libs/platform_core",
    "libs/monorepo_guards",
)

#: Where the sim answers on sedona, and what ``up`` asks it for: room 1's
#: terrain, one class byte per tile of a 256 by 256 field.
SIM_URL: Final[str] = "http://127.0.0.1:8765"
SMOKE_PATH: Final[str] = "/terrain/1"
TERRAIN_BYTES: Final[int] = 65536

#: How often, and how far apart, ``up`` asks before giving up.
SMOKE_ATTEMPTS: Final[int] = 20
SMOKE_PAUSE_SECONDS: Final[float] = 3.0

_USAGE: Final[str] = "usage: python -m scripts.sim_host {up|down}\n"


def image_tag(sha: str) -> str:
    """The image the sim runs as for one commit.

    Args:
        sha: HEAD's short sha.

    Returns:
        ``tankpit-sim:<sha>``.
    """
    return f"tankpit-sim:{sha}"


def head_sha(project_root: Path) -> str:
    """HEAD's short sha, the commit the image is built from.

    Args:
        project_root: This package's directory, inside the git checkout.

    Returns:
        The short sha.

    Raises:
        FleetHostError: SIM_NOT_COMMITTED when git cannot name HEAD.
    """
    return run_checked(
        ["git", "rev-parse", "--short", "HEAD"], project_root, "SIM_NOT_COMMITTED"
    ).strip()


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
        f"{SIM_DIR}/{COMPOSE_FILE}",
        "--env-file",
        f"{SIM_DIR}/sim.env",
        *args,
    ]


def stage(project_root: Path, sha: str, secrets_file: Path, scratch: Path) -> None:
    """Assemble the host directory in ``scratch``, then copy it to sedona's.

    Args:
        project_root: This package's directory, inside the git checkout.
        sha: HEAD's short sha, which names the image.
        secrets_file: The hub's file holding ``SIM_DB_PASSWORD``.
        scratch: An empty directory to assemble the files in.

    Raises:
        FleetHostError: SIM_SECRETS_MISSING when the secrets file is absent;
            SIM_NOT_COMMITTED when HEAD lacks the compose file;
            SIM_STAGE_FAILED when sedona refuses the directory or a copy.
    """
    if not script_hooks.path_exists(secrets_file):
        raise FleetHostError(
            f"SIM_SECRETS_MISSING: {secrets_file} is absent; it holds the sim's SIM_DB_PASSWORD"
        )
    committed = run_checked(
        ["git", "show", f"HEAD:./{COMPOSE_FILE}"], project_root, "SIM_NOT_COMMITTED"
    )
    (scratch / COMPOSE_FILE).write_text(committed, encoding="utf-8", newline="")
    secrets = script_hooks.read_text(secrets_file)
    (scratch / "sim.env").write_text(
        f"SIM_IMAGE={image_tag(sha)}\n{secrets}", encoding="utf-8", newline=""
    )
    run_checked(
        on_sedona(f"$null = New-Item -ItemType Directory -Force -Path {SIM_DIR}"),
        project_root,
        "SIM_STAGE_FAILED",
    )
    run_checked(
        [
            "scp",
            "-q",
            str(scratch / COMPOSE_FILE),
            str(scratch / "sim.env"),
            f"{SEDONA_SSH}:{SIM_DIR}/",
        ],
        project_root,
        "SIM_STAGE_FAILED",
    )


def pull_database(project_root: Path, scratch: Path) -> None:
    """Pull the database image onto sedona through the hub's docker client.

    Not by compose on sedona, for the reason :func:`scripts.fleet_host.pull_edge`
    gives: Docker Desktop's credential helper fails in an ssh logon session.

    Args:
        project_root: Where to run docker from.
        scratch: The host directory :func:`stage` assembled.

    Raises:
        FleetHostError: SIM_PULL_FAILED when the pull fails.
    """
    run_checked(
        [
            "docker",
            "--host",
            SEDONA_DOCKER_HOST,
            "compose",
            "-f",
            str(scratch / COMPOSE_FILE),
            "--env-file",
            str(scratch / "sim.env"),
            "pull",
            "sim-db",
        ],
        project_root,
        "SIM_PULL_FAILED",
    )


def ensure_image(project_root: Path, sha: str, scratch: Path) -> bool:
    """Build HEAD's image on sedona unless the tag is already there.

    Args:
        project_root: This package's directory, inside the git checkout.
        sha: HEAD's short sha.
        scratch: A directory the archive is unpacked under.

    Returns:
        True when it was built now, False when sedona already had it.

    Raises:
        FleetHostError: SIM_IMAGE_BUILD_FAILED when the archive, its
            unpacking or the build fails.
    """
    tag = image_tag(sha)
    held = script_hooks.run_command(
        ["docker", "--host", SEDONA_DOCKER_HOST, "image", "inspect", tag], project_root
    )
    if held["returncode"] == 0:
        return False
    repo_root = project_root.parents[1]
    context = scratch / "context"
    context.mkdir()
    tarball = scratch / "snapshot.tar"
    run_checked(
        ["git", "archive", "--format=tar", "-o", str(tarball), "HEAD", "--", *ARCHIVE_PATHS],
        repo_root,
        "SIM_IMAGE_BUILD_FAILED",
    )
    run_checked(
        ["tar", "-xf", str(tarball), "-C", str(context)], repo_root, "SIM_IMAGE_BUILD_FAILED"
    )
    run_checked(
        [
            "docker",
            "--host",
            SEDONA_DOCKER_HOST,
            "build",
            "-f",
            str(context / "clients" / "TankpitBot" / "Dockerfile"),
            "-t",
            tag,
            "--build-arg",
            f"BUILD_REF=sim-{sha}",
            str(context),
        ],
        project_root,
        "SIM_IMAGE_BUILD_FAILED",
    )
    return True


def smoke(project_root: Path) -> None:
    """Wait for the sim to serve room 1's terrain on sedona's loopback.

    Args:
        project_root: Where to run ssh from.

    Raises:
        FleetHostError: SIM_SMOKE_FAILED naming the last answer, when
            :data:`SMOKE_ATTEMPTS` asks never read the whole field.
    """
    expected = f"200 {TERRAIN_BYTES}"
    last = ""
    for attempt in range(SMOKE_ATTEMPTS):
        if attempt > 0:
            script_hooks.sleep_seconds(SMOKE_PAUSE_SECONDS)
        result = script_hooks.run_command(
            [
                "ssh",
                SEDONA_SSH,
                "curl.exe",
                "-s",
                "-o",
                "NUL",
                "-w",
                "%{http_code} %{size_download}",
                f"{SIM_URL}{SMOKE_PATH}",
            ],
            project_root,
        )
        last = result["stdout"].strip()
        if last == expected:
            return
    raise FleetHostError(
        f"SIM_SMOKE_FAILED: {SIM_URL}{SMOKE_PATH} on sedona after {SMOKE_ATTEMPTS} "
        f"attempts answered {last!r}, not {expected!r}"
    )


def up(project_root: Path, secrets_file: Path) -> list[str]:
    """Run HEAD's sim on sedona and see it serve.

    Args:
        project_root: This package's directory, inside the git checkout.
        secrets_file: The hub's file holding ``SIM_DB_PASSWORD``.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: As :func:`head_sha`, :func:`stage`,
            :func:`pull_database`, :func:`ensure_image` and :func:`smoke`;
            SIM_COMPOSE_FAILED when compose refuses.
    """
    sha = head_sha(project_root)
    with tempfile.TemporaryDirectory() as scratch:
        stage(project_root, sha, secrets_file, Path(scratch))
        pull_database(project_root, Path(scratch))
        built = ensure_image(project_root, sha, Path(scratch))
    run_checked(_compose("up", "-d", "--no-build"), project_root, "SIM_COMPOSE_FAILED")
    smoke(project_root)
    return [
        f"{COMPOSE_FILE} at {sha} staged into {SEDONA_SSH}:{SIM_DIR}",
        f"image {image_tag(sha)} {'built on sedona' if built else 'already on sedona'}",
        f"sim up; {SIM_URL}{SMOKE_PATH} on sedona serves the whole field",
    ]


def down(project_root: Path) -> list[str]:
    """Stop the sim and its database on sedona.

    Compose stops the sim before the database it depends on, and the sim
    records every seat still in play as it stops. The ``sim-db`` volume
    keeps every account for the next ``up``.

    Args:
        project_root: Where to run ssh from.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: SIM_COMPOSE_FAILED when compose refuses.
    """
    run_checked(_compose("stop"), project_root, "SIM_COMPOSE_FAILED")
    return [f"sim and sim-db stopped on {SEDONA_SSH}; the accounts stay in the sim-db volume"]


def run(arguments: list[str], project_root: Path, secrets_file: Path) -> list[str]:
    """Do what the command line asks.

    Args:
        arguments: The arguments after the program name.
        project_root: This package's directory, inside the git checkout.
        secrets_file: The hub's file holding ``SIM_DB_PASSWORD``.

    Returns:
        The lines reporting what was done.

    Raises:
        SystemExit: 2 with the usage for anything but ``up`` or ``down``.
        FleetHostError: As :func:`up` and :func:`down`.
    """
    if arguments == ["up"]:
        return up(project_root, secrets_file)
    if arguments == ["down"]:
        return down(project_root)
    sys.stdout.write(_USAGE)
    raise SystemExit(2)


def main() -> None:
    """Entry point, run from this package's directory against the hub's secrets.

    Raises:
        SystemExit: As :func:`run`.
        FleetHostError: As :func:`run`.
    """
    for line in run(list(core_hooks.get_argv())[1:], Path.cwd(), SECRETS_FILE):
        sys.stdout.write(f"{line}\n")


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHIVE_PATHS",
    "COMPOSE_FILE",
    "SECRETS_FILE",
    "SEDONA_DOCKER_HOST",
    "SIM_DIR",
    "SIM_URL",
    "SMOKE_ATTEMPTS",
    "SMOKE_PATH",
    "SMOKE_PAUSE_SECONDS",
    "TERRAIN_BYTES",
    "down",
    "ensure_image",
    "head_sha",
    "image_tag",
    "main",
    "pull_database",
    "run",
    "smoke",
    "stage",
    "up",
]
