"""Freeze HEAD into a release on the game host's ladder.

The fleet runs a frozen snapshot, never the hot tree, so surgery on a
checkout can never crash a live session (2026-08-28: deleting a
runtime-read file killed one). A release is ``git archive`` of HEAD for the
bot and its two path-dependency libraries, so uncommitted work is excluded
by construction, unpacked into :data:`scripts.fleet_host.RELEASES_DIR` on
the host as ``v<version>-<short sha>``. ``make up`` then builds its image
there from that directory.

It carries no secret. The host's ``.env``, ``accounts.json`` and tunnel
token live in the host's own directory (:mod:`scripts.fleet_host` says
why), so a release is the committed tree and nothing else. Until board task
c481c259 this was a PowerShell recipe writing to a Windows path on the hub
and copying the hub's gitignored files in.

The archive is unpacked into ``<name>.partial`` and renamed into place only
once whole, so ``up`` never runs a half-written release.

Usage::

    python -m scripts.fleet_release
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Final

from scripts import _test_hooks as script_hooks
from scripts.fleet_host import RELEASES_DIR
from scripts.fleet_remote import GAME_HOST_SSH, FleetHostError, on_host, run_checked

#: What a release holds, relative to the repository root.
ARCHIVED_PATHS: Final[tuple[str, ...]] = (
    "clients/TankpitBot",
    "libs/platform_core",
    "libs/monorepo_guards",
)

#: The package's version line in ``pyproject.toml``.
_VERSION_LINE: Final[re.Pattern[str]] = re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.MULTILINE)

#: Unpacks the archive beside the ladder and renames it into place:
#: ``$1`` the archive, ``$2`` the partial directory, ``$3`` the release.
_UNPACK: Final[str] = 'set -e; rm -rf "$2"; mkdir "$2"; tar -xf "$1" -C "$2"; rm "$1"; mv "$2" "$3"'


def release_name(project_root: Path) -> str:
    """The release HEAD makes: the package version and HEAD's short sha.

    Args:
        project_root: This package's directory, inside the git checkout.

    Returns:
        ``v<version>-<short sha>``.

    Raises:
        FleetHostError: FLEET_NO_VERSION when ``pyproject.toml`` states no
            version line; FLEET_NO_HEAD when git cannot name HEAD.
    """
    found = _VERSION_LINE.search((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    if found is None:
        raise FleetHostError(f"FLEET_NO_VERSION: {project_root / 'pyproject.toml'} states none")
    sha = run_checked(["git", "rev-parse", "--short", "HEAD"], project_root, "FLEET_NO_HEAD")
    return f"v{found.group(1)}-{sha.strip()}"


def send_head(project_root: Path, scratch: Path, parent: str, name: str, code: str) -> None:
    """Unpack HEAD's :data:`ARCHIVED_PATHS` into ``<parent>/<name>`` on the host.

    Args:
        project_root: This package's directory, inside the git checkout.
        scratch: An empty local directory for the archive.
        parent: The host directory the tree lands in.
        name: The tree's directory name there.
        code: The error code a refusal by the host is raised with.

    Raises:
        FleetHostError: FLEET_ARCHIVE_FAILED when git cannot archive HEAD;
            ``code`` when the host refuses the directory, the copy or the
            unpack.
    """
    archive = scratch / f"{name}.tar"
    run_checked(
        ["git", "-C", "../..", "archive", "--format=tar", "-o", str(archive), "HEAD", "--",
         *ARCHIVED_PATHS],
        project_root,
        "FLEET_ARCHIVE_FAILED",
    )  # fmt: skip
    run_checked(on_host(["mkdir", "-p", parent]), project_root, code)
    run_checked(["scp", "-q", str(archive), f"{GAME_HOST_SSH}:{parent}/"], project_root, code)
    final = f"{parent}/{name}"
    run_checked(
        on_host(["sh", "-c", _UNPACK, "sh", f"{final}.tar", f"{final}.partial", final]),
        project_root,
        code,
    )


def release(project_root: Path, scratch: Path) -> list[str]:
    """Put HEAD's release on the host's ladder, unless it is there already.

    Args:
        project_root: This package's directory, inside the git checkout.
        scratch: An empty local directory for the archive.

    Returns:
        The lines reporting what was done.

    Raises:
        FleetHostError: As :func:`release_name`, and as :func:`send_head`
            with FLEET_RELEASE_FAILED.
    """
    name = release_name(project_root)
    final = f"{RELEASES_DIR}/{name}"
    held = script_hooks.run_command(on_host(["test", "-d", final]), project_root)
    if held["returncode"] == 0:
        return [f"release {name} is already on {GAME_HOST_SSH}:{RELEASES_DIR}; nothing to do"]
    send_head(project_root, scratch, RELEASES_DIR, name, "FLEET_RELEASE_FAILED")
    return [
        f"release {name} written to {GAME_HOST_SSH}:{final}",
        "activate: make down, then make up, which builds its image on the host",
    ]


def main() -> None:
    """Entry point, run from this package's directory.

    Raises:
        FleetHostError: As :func:`release`.
    """
    with tempfile.TemporaryDirectory() as scratch:
        for line in release(Path.cwd(), Path(scratch)):
            sys.stdout.write(f"{line}\n")


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHIVED_PATHS",
    "main",
    "release",
    "release_name",
    "send_head",
]
