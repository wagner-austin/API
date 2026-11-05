"""A node runner's registry as a rolled extraction, and its reads of the ref.

Support module, not a test module. A serve runs from the registry the
launcher extracted, ``<extractions>/<commit>/tools/fleet/registry``
(:func:`fleet.core.rolled.extraction_commit`), and reads
``refs/fleet/rolled`` around every claim (:mod:`fleet.cli.node_roll`, MCPs
board task 1fd01f09). The shared ``config_path`` is written where an
extraction lands, under the scratch root the tests pin, and its records
directory carries a fetch stamp written at :data:`tests.conftest.DEMO_NOW`,
so each read of the ref inside the stamp's window is the two git calls
:func:`roll_read` answers: the git directory, then the ref.
"""

from __future__ import annotations

import os
import pathlib
from typing import Final

from fleet.core import _test_hooks, rolled_fetch, scratch
from fleet.core.rolled import REGISTRY_DIRECTORY

#: The commit the tests' registry was extracted from.
ROLLED_COMMIT: Final = "84bc5ff0b6a90c1d2e3f405162738495a6b7c8d9"

#: A commit a roll moved the ref to after the serve's extraction.
NEXT_COMMIT: Final = "5aa8ed06c1d2e3f405162738495a6b7c8d90e1f2"


def extracted_registry(scratch_root: pathlib.Path, *, now: int) -> pathlib.Path:
    """Where the tests' registry goes: a rolled extraction under the scratch root.

    Args:
        scratch_root: The directory ``_test_hooks.temp_root`` answers.
        now: The pinned clock, which the fetch stamp is written at.

    Returns:
        ``<scratch>/fleet-rolled/<ROLLED_COMMIT>/tools/fleet/registry``, not
        yet written; its records directory exists, holding a fresh stamp.
    """
    registry = (
        scratch_root
        / scratch.EXTRACTIONS_DIR
        / ROLLED_COMMIT
        / pathlib.PurePosixPath(REGISTRY_DIRECTORY)
    )
    registry.parent.mkdir(parents=True)
    stamp = registry.parent / rolled_fetch.STAMP_NAME
    stamp.write_text("", encoding="utf-8")
    os.utime(stamp, (now, now))
    return registry


def roll_read(
    config_path: pathlib.Path, commit: str = ROLLED_COMMIT, *, fetch: bool = False
) -> tuple[_test_hooks.CommandResult, ...]:
    """The git answers one read of the ref takes.

    Args:
        config_path: The registry the runner loaded; its parent is the
            records directory, which the read names as its git directory.
        commit: What the ref names.
        fetch: Whether the stamp is older than its window by the read, as
            in a later tick of a serve whose clock has moved, so the read
            fetches (and stamps again) before it reads the ref.

    Returns:
        ``git rev-parse --git-common-dir``'s answer, the fetch's when one
        runs, then the ref's.
    """
    fetched: tuple[str, ...] = ("",) if fetch else ()
    answers: tuple[str, ...] = (f"{config_path.parent}\n", *fetched, f"{commit}\n")
    return tuple(
        _test_hooks.CommandResult(returncode=0, stdout=answer, stderr="", timed_out=False)
        for answer in answers
    )


__all__ = ["NEXT_COMMIT", "ROLLED_COMMIT", "extracted_registry", "roll_read"]
