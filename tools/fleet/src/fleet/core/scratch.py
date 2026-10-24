"""The fleet's scratch root: what it holds and when a leftover is removed.

WHAT LIVES THERE. Two directories under
:func:`fleet.core._test_hooks.temp_root`: :data:`EXTRACTIONS_DIR`, the
rolled fleet code a tick runs from, one tree per commit
(:mod:`fleet.core.rolled`), and :data:`ARCHIVES_DIR`, the staging archives
and companion bundles a runner sends to a node
(:attr:`fleet.cli._config.LoadedWorkspace.archives`). On Linux the root is
memory (:mod:`fleet.core.scratch_root`), so whatever is
left there holds RAM until something removes it.

WHAT IS LEFT BEHIND (MCPs board task fb505198). Each writer removes its own
file when it has finished with it, and a process killed first removes
nothing. Measured on diphtheria on 2026-10-24, with the root still on its
SSD: 873 rolled extractions of about 9.3 MB, each a tick that died before
its launcher could remove it, and 3.6 GB of staging archives, mostly 150 MB
MCPs execution exports whose stage failed before the archive was unlinked.

WHAT COUNTS AS LEFT. An entry whose modification time is older than
:data:`STALE_SECONDS`, which is longer than any process that writes or
reads one can live: a tick is ended at
:data:`fleet.cli.tick.TICK_WALL_SECONDS` (39.5 minutes), and its agent at
:data:`fleet.cli.rolled.AGENT_WALL_SECONDS` inside it. A rolled tree is
touched by every launch that reuses it, so only a tree no launch has
started from for that long is old enough. Age, not a process probe,
because the entries name no owner: an archive is named for its run, and a
tree for its commit, which many launches share.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from typing import Final

from fleet.core import _test_hooks

#: The scratch-root subdirectory the rolled extractions live in.
EXTRACTIONS_DIR: Final = "fleet-rolled"

#: The scratch-root subdirectory the staging archives are built in.
ARCHIVES_DIR: Final = "fleet-archives"

#: How old an entry is when nothing can still be using it: 45 minutes,
#: five and a half past the tick's own wall.
STALE_SECONDS: Final = 45 * 60


def directory(name: str) -> pathlib.Path:
    """One of the scratch root's subdirectories.

    Args:
        name: :data:`EXTRACTIONS_DIR` or :data:`ARCHIVES_DIR`.

    Returns:
        Its path under the scratch root, which may not exist yet.
    """
    return _test_hooks.temp_root() / name


def reap(parent: pathlib.Path, *, keep: frozenset[str]) -> tuple[str, ...]:
    """Remove every entry of a scratch directory older than :data:`STALE_SECONDS`.

    Called by the rolled launcher inside its extraction turn
    (:func:`fleet.core.rolled.extract_rolled_tree`), so no two reapers run
    at once; a runner writing a fresh file meanwhile is untouched, because
    nothing fresh is old enough.

    Args:
        parent: The directory whose entries are read.
        keep: Names never removed, whatever their age.

    Returns:
        The names removed, in name order; empty when the directory does not
        exist yet.
    """
    if not _test_hooks.directory_exists(parent):
        return ()
    now = _test_hooks.now()
    removed: list[str] = []
    for entry in sorted(parent.iterdir()):
        if entry.name in keep or not stale(entry, now=now):
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed.append(entry.name)
    return tuple(removed)


def stale(entry: pathlib.Path, *, now: int) -> bool:
    """Say whether an entry was last modified more than :data:`STALE_SECONDS` ago.

    Args:
        entry: A path a directory listing returned.
        now: The current time in whole seconds.

    Returns:
        True when it is that old; False when it is younger, or gone since it
        was listed, which only a fresh entry can be: a runner unlinks its own
        archive the moment its stage ends.
    """
    try:
        modified = int(os.lstat(entry).st_mtime)
    except FileNotFoundError:
        return False
    return now - modified > STALE_SECONDS


__all__ = ["ARCHIVES_DIR", "EXTRACTIONS_DIR", "STALE_SECONDS", "directory", "reap", "stale"]
