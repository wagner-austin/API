"""Operate the sim server on the game host, colossus, over plain ssh.

The game host owns every game server (MCPs board task 077204e8; the operator,
2026-09-28: "sedona can own all the game servwrs. tankpit too"), and since
MCPs board task ff035b0d that includes the sim server. Board task c481c259
moved the game host from sedona to colossus, a Linux machine.
``make sim-up`` and ``make sim-down`` run this from any Linux checkout that
can ssh to it.

WHAT ``up`` STAGES, AND FROM WHERE. The host directory, :data:`SIM_DIR`,
holds ``sim-server.compose.json`` as COMMITTED at HEAD and ``sim.env``,
which names the image. The database password is the host's own, in
``secrets.env`` beside them, which no repository holds and ``up`` refuses
to run without.

WHICH IMAGE. The sim runs ``tankpit-sim:<HEAD's short sha>``, built on the
host from HEAD's tree, which :func:`scripts.fleet_release.send_head` unpacks
under :data:`SIM_DIR`, only when the host lacks that tag, so an uncommitted
edit never reaches it. It is not the fleet's release image: ``make up`` runs
the newest release, and a release cut for the sim would roll the fleet at
the next ``up``.

IT RUNS ONLY WHEN ASKED. The operator, 2026-10-13: "make sure it doesnt auro
stsrt or anything i csn maniallt stsrt it wheb needed". Both services carry
``restart: "no"``, so a reboot of the host or of its Docker leaves them off,
and ``down`` stops both; ``up`` is the one way they start.

WHERE IT ANSWERS. The sim publishes on the host's loopback alone, and
accounts live in the ``sim-db`` volume beside it, which outlives every
``down``. ``up`` ends by asking that loopback, over ssh, for room 1's
terrain, which the server reads off the field it plays: an ``up`` that exits
0 has been seen serving.

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
from scripts.fleet_release import send_head
from scripts.fleet_remote import (
    GAME_HOST_HOME,
    GAME_HOST_SSH,
    FleetHostError,
    on_host,
    run_checked,
)
from tankpit_bot import _test_hooks as core_hooks

#: The sim's directory on the host.
SIM_DIR: Final[str] = f"{GAME_HOST_HOME}/tankpit-sim"

#: The compose file, relative to this package, staged from HEAD.
COMPOSE_FILE: Final[str] = "sim-server.compose.json"

#: The host's own file holding ``SIM_DB_PASSWORD``, under :data:`SIM_DIR`.
SECRETS_FILE: Final[str] = "secrets.env"

#: Where the sim answers on the host, and what ``up`` asks it for: room 1's
#: terrain, one class byte per tile of a 256 by 256 field.
SIM_URL: Final[str] = "http://127.0.0.1:8765"
SMOKE_PATH: Final[str] = "/terrain/1"
TERRAIN_BYTES: Final[int] = 65536

#: What curl prints for one ask: the status and the bytes read. The braces
#: and percent signs reach the host's shell quoted, as one word.
SMOKE_FORMAT: Final[str] = "%{http_code}:%{size_download}"

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
    """An ssh command line that runs compose on the host against the sim's directory.

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
            f"{SIM_DIR}/{COMPOSE_FILE}",
            "--env-file",
            f"{SIM_DIR}/sim.env",
            "--env-file",
            f"{SIM_DIR}/{SECRETS_FILE}",
            *args,
        ]
    )


def stage(project_root: Path, sha: str, scratch: Path) -> None:
    """Refuse a host without the password, then copy HEAD's compose file and ``sim.env`` to it.

    Args:
        project_root: This package's directory, inside the git checkout.
        sha: HEAD's short sha, which names the image.
        scratch: An empty directory to assemble the files in.

    Raises:
        FleetHostError: SIM_SECRETS_MISSING when the host has no secrets
            file; SIM_NOT_COMMITTED when HEAD lacks the compose file;
            SIM_STAGE_FAILED when the host refuses a copy.
    """
    run_checked(
        on_host(["ls", "-1", f"{SIM_DIR}/{SECRETS_FILE}"]), project_root, "SIM_SECRETS_MISSING"
    )
    committed = run_checked(
        ["git", "show", f"HEAD:./{COMPOSE_FILE}"], project_root, "SIM_NOT_COMMITTED"
    )
    (scratch / COMPOSE_FILE).write_text(committed, encoding="utf-8", newline="")
    (scratch / "sim.env").write_text(f"SIM_IMAGE={image_tag(sha)}\n", encoding="utf-8", newline="")
    run_checked(
        ["scp", "-q", str(scratch / COMPOSE_FILE), str(scratch / "sim.env"),
         f"{GAME_HOST_SSH}:{SIM_DIR}/"],
        project_root,
        "SIM_STAGE_FAILED",
    )  # fmt: skip


def ensure_image(project_root: Path, sha: str, scratch: Path) -> bool:
    """Build HEAD's image on the host unless the tag is already there.

    Args:
        project_root: This package's directory, inside the git checkout.
        sha: HEAD's short sha.
        scratch: An empty local directory for HEAD's archive.

    Returns:
        True when it was built now, False when the host already had it.

    Raises:
        FleetHostError: As :func:`scripts.fleet_release.send_head` with
            SIM_IMAGE_BUILD_FAILED; SIM_IMAGE_BUILD_FAILED when the build
            fails.
    """
    tag = image_tag(sha)
    held = script_hooks.run_command(on_host(["docker", "image", "inspect", tag]), project_root)
    if held["returncode"] == 0:
        return False
    name = f"context-{sha}"
    send_head(project_root, scratch, SIM_DIR, name, "SIM_IMAGE_BUILD_FAILED")
    context = f"{SIM_DIR}/{name}"
    build = [
        "docker",
        "build",
        "-f",
        f"{context}/clients/TankpitBot/Dockerfile",
        "-t",
        tag,
        "--build-arg",
        f"BUILD_REF=sim-{sha}",
        context,
    ]
    run_checked(on_host(build), project_root, "SIM_IMAGE_BUILD_FAILED")
    run_checked(on_host(["rm", "-rf", context]), project_root, "SIM_IMAGE_BUILD_FAILED")
    return True


def smoke(project_root: Path) -> None:
    """Wait for the sim to serve room 1's terrain on the host's loopback.

    Args:
        project_root: Where to run ssh from.

    Raises:
        FleetHostError: SIM_SMOKE_FAILED naming the last answer, when
            :data:`SMOKE_ATTEMPTS` asks never read the whole field.
    """
    expected = f"200:{TERRAIN_BYTES}"
    ask = on_host(["curl", "-s", "-o", "/dev/null", "-w", SMOKE_FORMAT, f"{SIM_URL}{SMOKE_PATH}"])
    last = ""
    for attempt in range(SMOKE_ATTEMPTS):
        if attempt > 0:
            script_hooks.sleep_seconds(SMOKE_PAUSE_SECONDS)
        last = script_hooks.run_command(ask, project_root)["stdout"].strip()
        if last == expected:
            return
    raise FleetHostError(
        f"SIM_SMOKE_FAILED: {SIM_URL}{SMOKE_PATH} on {GAME_HOST_SSH} after {SMOKE_ATTEMPTS} "
        f"attempts answered {last!r}, not {expected!r}"
    )


def up(project_root: Path) -> list[str]:
    """Run HEAD's sim on the host and see it serve.

    Args:
        project_root: This package's directory, inside the git checkout.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: As :func:`head_sha`, :func:`stage`,
            :func:`ensure_image` and :func:`smoke`; SIM_PULL_FAILED when
            the database image cannot be pulled; SIM_COMPOSE_FAILED when
            compose refuses.
    """
    sha = head_sha(project_root)
    with tempfile.TemporaryDirectory() as scratch:
        stage(project_root, sha, Path(scratch))
        run_checked(_compose("pull", "sim-db"), project_root, "SIM_PULL_FAILED")
        built = ensure_image(project_root, sha, Path(scratch))
    run_checked(_compose("up", "-d", "--no-build"), project_root, "SIM_COMPOSE_FAILED")
    smoke(project_root)
    return [
        f"{COMPOSE_FILE} at {sha} staged into {GAME_HOST_SSH}:{SIM_DIR}",
        f"image {image_tag(sha)} {'built on the host' if built else 'already on the host'}",
        f"sim up; {SIM_URL}{SMOKE_PATH} on the host serves the whole field",
    ]


def down(project_root: Path) -> list[str]:
    """Stop the sim and its database on the host.

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
    return [f"sim and sim-db stopped on {GAME_HOST_SSH}; the accounts stay in the sim-db volume"]


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
    "COMPOSE_FILE",
    "SECRETS_FILE",
    "SIM_DIR",
    "SIM_URL",
    "SMOKE_ATTEMPTS",
    "SMOKE_FORMAT",
    "SMOKE_PATH",
    "SMOKE_PAUSE_SECONDS",
    "TERRAIN_BYTES",
    "down",
    "ensure_image",
    "head_sha",
    "image_tag",
    "main",
    "run",
    "smoke",
    "stage",
    "up",
]
