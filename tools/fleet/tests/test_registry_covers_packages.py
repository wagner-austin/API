"""Every API package with a ``make check`` is a fleet project (MCPs board task b1089284).

THE GAP. A runner takes a check only of a project the registry declares, so
a package missing from it can be verified on the machine that changed it and
nowhere else, and nothing says so until a session's check sits queued: on
2026-10-11 tools/ci-wake's waited until 7c96acfbd declared it (board task
24547f50), and 37 packages under tools/, libs/, services/ and clients/ were
found in the same state, tools/lock-wake among them with 24 commits in the
month before. This test makes the next one fail here instead.

WHAT IT READS. Each directory directly under those four whose Makefile has a
``check:`` target, against every path a project sourced from this repository
declares. A package that cannot be checked on a fleet node is named in
:data:`NOT_ON_FLEET` with the reason, and that list is held to the tree too:
an excused package that disappears, loses its check or gets declared fails
the test, so an excuse cannot outlive the thing it excused.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from fleet.contracts.workspace import decode_fleet_workspace
from tests._registry import shipped_document

#: The repository root: tools/fleet/tests is three levels below it.
ROOT: Final = Path(__file__).resolve().parents[3]

#: This repository as the registry names it.
API_REMOTE: Final = "https://github.com/wagner-austin/API.git"

#: The directories the repository keeps its packages in, one level deep.
PACKAGE_PARENTS: Final[tuple[str, ...]] = ("tools", "libs", "services", "clients")

#: A Makefile's check target.
CHECK_TARGET: Final = re.compile(r"^check:", re.MULTILINE)

#: Packages with a check that no fleet node can run, each with why.
NOT_ON_FLEET: Final[dict[str, str]] = {
    "tools/hpc3-execution": (
        "its check is a real sbatch --test-only preflight over ssh to HPC3, which this "
        "workstation reaches through sedona or loki with its own key (ProxyJump "
        "authenticates end to end), so no fleet node holds the route; the hpc3 "
        "capability tag its Makefile names was never built"
    ),
}


def _checkable() -> list[str]:
    """Every package directory whose Makefile has a check target.

    Returns:
        Repository-relative paths, sorted within each parent.
    """
    found: list[str] = []
    for parent in PACKAGE_PARENTS:
        for child in sorted((ROOT / parent).iterdir()):
            makefile = child / "Makefile"
            if makefile.is_file() and CHECK_TARGET.search(makefile.read_text(encoding="utf-8")):
                found.append(f"{parent}/{child.name}")
    return found


def _declared() -> set[str]:
    """Every path a project sourced from this repository runs in.

    Returns:
        The declared paths.
    """
    workspace = decode_fleet_workspace(shipped_document())
    return {
        project["source"]["path"]
        for project in workspace["projects"].values()
        if project["source"] is not None and project["source"]["remote"] == API_REMOTE
    }


def test_every_package_with_a_check_is_declared_or_excused() -> None:
    checkable = _checkable()
    declared = _declared()

    undeclared = [name for name in checkable if name not in declared and name not in NOT_ON_FLEET]

    assert undeclared == []
    # Not a pass over nothing: the tree carried 57 such packages on
    # 2026-10-11, and a walk that found none would read as clean.
    assert len(checkable) >= 57


def test_every_excuse_names_a_package_that_still_needs_one() -> None:
    checkable = set(_checkable())
    declared = _declared()

    stale = sorted(name for name in NOT_ON_FLEET if name not in checkable or name in declared)

    assert stale == []
    assert all(len(reason) >= 40 for reason in NOT_ON_FLEET.values())
