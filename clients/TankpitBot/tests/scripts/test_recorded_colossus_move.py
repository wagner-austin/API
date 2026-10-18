"""The move of TankPit's state from sedona to colossus, read from its raw outputs.

Board task c481c259. ``docs/sources/colossus-move-2026-10-18/`` holds what
the two machines printed on 2026-10-18, unedited:

* ``sedona-runs-data.sha256.txt``: ``Get-FileHash -Algorithm SHA256`` over
  every file under sedona's ``C:/fleet/tankpit/runs`` and ``data``, as
  ``<hash>  <relative path>``, sorted by path;
* ``colossus-archive.sha256.txt``: ``sha256sum`` over every file of the
  copy, ``/home/user/tankpit-archive/sedona-2026-10-18`` on colossus, in
  the same form;
* ``colossus-image.txt``: colossus's ``tankpit-fleet`` images, its
  containers named ``tankpit``, and its release ladder, under a header
  naming the instant and the host.

These cases compute the move's claims from those files, so the closure
rests on their arithmetic and not on prose.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCES = Path(__file__).resolve().parents[2] / "docs" / "sources" / "colossus-move-2026-10-18"
HASH_LINE = re.compile(r"(?P<hash>[0-9a-f]{64})  (?P<path>(?:runs|data)/\S.*)")


def _manifest(name: str) -> dict[str, str]:
    """Read one hash listing.

    Args:
        name: The file under :data:`SOURCES`.

    Returns:
        Relative path to sha256.

    Raises:
        AssertionError: If a line is not a hash line, or a path repeats.
    """
    hashes: dict[str, str] = {}
    for line in (SOURCES / name).read_text(encoding="utf-8").splitlines():
        match = HASH_LINE.fullmatch(line)
        if match is None:
            raise AssertionError(f"not a hash line in {name}: {line!r}")
        path: str = match.group("path")
        if path in hashes:
            raise AssertionError(f"{path} listed twice in {name}")
        hashes[path] = match.group("hash")
    return hashes


def _sections(name: str) -> dict[str, list[str]]:
    """Split a listing into its ``# <command>`` sections.

    Args:
        name: The file under :data:`SOURCES`.

    Returns:
        Each header line, without its ``# ``, to the lines under it.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] = []
    for line in (SOURCES / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            current = sections.setdefault(line.removeprefix("# "), [])
        else:
            current.append(line)
    return sections


def test_every_file_of_sedonas_runs_and_data_reached_colossus_whole() -> None:
    """The two listings name the same 614 files with the same hashes."""
    sedona = _manifest("sedona-runs-data.sha256.txt")
    colossus = _manifest("colossus-archive.sha256.txt")

    assert len(sedona) == 614
    assert sorted(path for path in sedona if path.startswith("data/")) == [
        "data/tank_registry.corrupt-20261017T0811Z.json",
        "data/tank_registry.json",
    ]
    assert colossus == sedona


def test_the_release_image_is_on_colossus_and_no_tankpit_container_is() -> None:
    """One release on the ladder, its image built, and no container started from it."""
    sections = _sections("colossus-image.txt")
    header = next(name for name in sections if name.endswith(": docker image ls tankpit-fleet"))

    assert header.startswith("2026-10-18T")
    assert " colossus: " in header
    assert sections["ls /home/user/tankpit-releases"] == ["v0.1.0-8eae60c48"]
    assert [line.split(" ")[0] for line in sections[header]] == ["tankpit-fleet:v0.1.0-8eae60c48"]
    assert sections["docker ps -a --filter name=tankpit --format {{.Names}}"] == []
    assert sections["end"] == []
