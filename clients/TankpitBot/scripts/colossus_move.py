"""Measure TankPit's move from sedona to colossus on each host, by the fleet.

Board task c481c259 moved the game host from sedona to colossus on
2026-10-18. Its closure first rested on listings this session captured and
committed, and audit 85ae66f7 held that a committed capture proves only
what was written into it. So the claims are measured here, by code a fleet
runner executes on the host itself:

* ``sedona``: every file under sedona's ``runs`` and ``data`` still hashes
  to the committed listing, so the copy Codex asked to keep is as it was
  when the move read it;
* ``colossus``: every file of the copy on colossus hashes to the same
  listing; the release the move made is on the host's ladder; its image,
  ``tankpit-fleet:<release>``, is on the host's daemon and was built after
  the release was unpacked; and no container is named ``tankpit`` or was
  started from that image, because the operator asked that no bot run
  until he resumes.

THE LISTING. :data:`LISTING` is ``<sha256>  <path>`` per file, sorted by
path, as :func:`hash_tree` prints it; sedona's runs and data hashed to it at
the move and again at 15:28Z, and so did colossus's copy.

THESE ARE A MOVE'S MEASUREMENTS, NOT STANDING CHECKS. Once the operator
resumes and ``make up`` starts the fleet, a container from that image
exists and the ``colossus`` run refuses by design; the projects that run it,
``clients/TankpitBot-sedona-execution`` and
``clients/TankpitBot-colossus-execution``, are pinned to their hosts and
measure a commit, so their rows stay what they measured.

Usage::

    python -m scripts.colossus_move sedona
    python -m scripts.colossus_move colossus
"""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypedDict

from scripts.fleet_host import RELEASES_DIR, image_tag
from scripts.fleet_remote import GAME_HOST_HOME, FleetHostError, run_checked
from tankpit_bot import _test_hooks as core_hooks

#: The release the move made on colossus and built its image from.
RELEASE: Final[str] = "v0.1.0-8eae60c48"

#: The listing both hosts' trees must hash to, relative to this package.
LISTING: Final[Path] = Path("docs/sources/colossus-move-2026-10-18/sedona-runs-data.sha256.txt")

#: The directories of a TankPit host directory the move copied.
COPIED_DIRS: Final[tuple[str, ...]] = ("runs", "data")

#: One listing line: a lowercase sha256, two spaces, a path under a copied directory.
LISTING_LINE: Final[re.Pattern[str]] = re.compile(r"([0-9a-f]{64})  ((?:runs|data)/\S.*)")

#: docker's ``Created``: to the second, then an optional fraction, then the zone.
CREATED: Final[re.Pattern[str]] = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:\d{2})"
)

#: How many differing paths a refusal names.
NAMED_PATHS: Final[int] = 5

_USAGE: Final[str] = "usage: python -m scripts.colossus_move {sedona|colossus}\n"


class MovePlaces(TypedDict):
    """Where each host keeps what the move is measured against.

    Attributes:
        sedona_tree: sedona's TankPit directory, holding ``runs`` and ``data``.
        archive_tree: The copy on colossus, holding the same two.
        releases: colossus's release ladder.
    """

    sedona_tree: Path
    archive_tree: Path
    releases: Path


#: The real places, on each host.
HOST_PLACES: Final[MovePlaces] = MovePlaces(
    sedona_tree=Path("C:/fleet/tankpit"),
    archive_tree=Path(f"{GAME_HOST_HOME}/tankpit-archive/sedona-2026-10-18"),
    releases=Path(RELEASES_DIR),
)


def read_listing(path: Path) -> dict[str, str]:
    """Read a listing into path to sha256.

    Args:
        path: The listing.

    Returns:
        Each listed path, ``/``-separated, to its sha256.

    Raises:
        FleetHostError: MOVE_LISTING_MALFORMED for a line that is not a
            listing line or a path listed twice.
    """
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LISTING_LINE.fullmatch(line)
        if match is None or match.group(2) in hashes:
            raise FleetHostError(f"MOVE_LISTING_MALFORMED: {path}: {line!r}")
        hashes[match.group(2)] = match.group(1)
    return hashes


def file_sha256(path: Path) -> str:
    """Hash one file a megabyte at a time.

    Args:
        path: The file.

    Returns:
        Its sha256, lowercase hex.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """Hash every file under a host directory's copied directories.

    Args:
        root: The host directory.

    Returns:
        Each file's path under ``root``, ``/``-separated, to its sha256.

    Raises:
        FleetHostError: MOVE_TREE_MISSING naming a copied directory
            ``root`` does not hold.
    """
    hashes: dict[str, str] = {}
    for name in COPIED_DIRS:
        directory = root / name
        if not directory.is_dir():
            raise FleetHostError(f"MOVE_TREE_MISSING: {directory} is not a directory")
        for path in sorted(entry for entry in directory.rglob("*") if entry.is_file()):
            hashes[path.relative_to(root).as_posix()] = file_sha256(path)
    return hashes


def _named(paths: set[str]) -> str:
    """Name at most :data:`NAMED_PATHS` of a set of paths.

    Args:
        paths: The paths.

    Returns:
        Their count and the first of them in order.
    """
    first = ", ".join(sorted(paths)[:NAMED_PATHS])
    return f"{len(paths)} ({first})" if paths else "0"


def compare_tree(root: Path, listing: Path) -> str:
    """Require a host directory to hash to the listing, file for file.

    Args:
        root: The host directory.
        listing: The listing.

    Returns:
        The line saying how many files matched.

    Raises:
        FleetHostError: MOVE_TREE_MISSING as :func:`hash_tree`;
            MOVE_LISTING_MALFORMED as :func:`read_listing`; MOVE_TREE_DIFFERS
            naming the files missing, unlisted and changed.
    """
    expected = read_listing(listing)
    found = hash_tree(root)
    missing = set(expected) - set(found)
    unlisted = set(found) - set(expected)
    changed = {path for path in set(expected) & set(found) if expected[path] != found[path]}
    if missing or unlisted or changed:
        raise FleetHostError(
            f"MOVE_TREE_DIFFERS: {root} against {listing.name}: missing {_named(missing)}, "
            f"unlisted {_named(unlisted)}, changed {_named(changed)}"
        )
    return (
        f"{len(found)} of {len(expected)} files under {root} ({', '.join(COPIED_DIRS)}) "
        f"hash to {listing.name}"
    )


def created_at(created: str) -> datetime:
    """Read docker's ``Created`` instant.

    Args:
        created: As ``docker image inspect`` prints it.

    Returns:
        The instant, to the second, in UTC.

    Raises:
        FleetHostError: MOVE_IMAGE_UNREADABLE for any other shape.
    """
    match = CREATED.fullmatch(created)
    if match is None:
        raise FleetHostError(f"MOVE_IMAGE_UNREADABLE: Created {created!r}")
    second: str = match.group(1)
    zone: str = match.group(2)
    return datetime.fromisoformat(second + zone).astimezone(UTC)


def colossus_host(releases: Path, project_root: Path) -> list[str]:
    """Require the move's release, its image, and no container from it.

    Args:
        releases: The host's release ladder.
        project_root: Where to run docker from.

    Returns:
        The lines saying what was found.

    Raises:
        FleetHostError: MOVE_RELEASE_MISSING when the ladder lacks
            :data:`RELEASE`; MOVE_IMAGE_MISSING when the daemon lacks its
            image; MOVE_IMAGE_UNREADABLE as :func:`created_at`;
            MOVE_IMAGE_PREDATES_RELEASE when the image is older than the
            unpacked release; MOVE_CONTAINER_STARTED naming any container
            by name or by image.
    """
    release = releases / RELEASE
    if not release.is_dir():
        raise FleetHostError(f"MOVE_RELEASE_MISSING: {release} is not a directory")
    unpacked = datetime.fromtimestamp(release.stat().st_mtime, UTC).replace(microsecond=0)
    tag = image_tag(RELEASE)
    inspect = ["docker", "image", "inspect", "--format", "{{.Id}} {{.Created}}", tag]
    answer = run_checked(inspect, project_root, "MOVE_IMAGE_MISSING").strip()
    image_id, _, created = answer.partition(" ")
    built = created_at(created)
    if built < unpacked:
        raise FleetHostError(
            f"MOVE_IMAGE_PREDATES_RELEASE: {tag} was created {built.isoformat()}, before "
            f"{release} was unpacked at {unpacked.isoformat()}"
        )
    started: list[str] = []
    for selector in ("name=tankpit", f"ancestor={tag}"):
        listing = ["docker", "ps", "-a", "--filter", selector, "--format", "{{.Names}}"]
        names = run_checked(listing, project_root, "MOVE_CONTAINERS_UNREADABLE").split()
        started.extend(f"{name} ({selector})" for name in names)
    if started:
        raise FleetHostError(f"MOVE_CONTAINER_STARTED: {', '.join(started)}")
    return [
        f"release {RELEASE} on the ladder at {releases}, unpacked {unpacked.isoformat()}",
        f"image {tag} {image_id} created {built.isoformat()}",
        f"no container named tankpit and none from {tag}",
    ]


def run(arguments: list[str], project_root: Path, places: MovePlaces) -> list[str]:
    """Measure the host the command line names.

    Args:
        arguments: The arguments after the program name.
        project_root: This package's directory.
        places: Where each host keeps what is measured.

    Returns:
        The lines saying what was measured.

    Raises:
        SystemExit: 2 with the usage for anything but ``sedona`` or
            ``colossus``.
        FleetHostError: As :func:`compare_tree` and :func:`colossus_host`.
    """
    listing = project_root / LISTING
    if arguments == ["sedona"]:
        return [compare_tree(places["sedona_tree"], listing)]
    if arguments == ["colossus"]:
        return [
            compare_tree(places["archive_tree"], listing),
            *colossus_host(places["releases"], project_root),
        ]
    sys.stdout.write(_USAGE)
    raise SystemExit(2)


def write_lines(lines: list[str]) -> None:
    """Print a measurement's lines.

    Args:
        lines: As :func:`run` returns them.
    """
    for line in lines:
        sys.stdout.write(f"{line}\n")


def main() -> None:
    """Entry point, run from this package's directory on the host it names.

    Raises:
        SystemExit: As :func:`run`.
        FleetHostError: As :func:`run`.
    """
    write_lines(run(list(core_hooks.get_argv())[1:], Path.cwd(), HOST_PLACES))


if __name__ == "__main__":
    main()


__all__ = [
    "COPIED_DIRS",
    "HOST_PLACES",
    "LISTING",
    "RELEASE",
    "MovePlaces",
    "colossus_host",
    "compare_tree",
    "created_at",
    "file_sha256",
    "hash_tree",
    "main",
    "read_listing",
    "run",
    "write_lines",
]
