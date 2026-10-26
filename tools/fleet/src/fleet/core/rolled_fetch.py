"""Bring this checkout's ``refs/fleet/rolled`` to origin's before it is read.

THE GAP (MCPs board task 8db0f536). ``make fleet-roll`` moves the ref in the
repository it runs in and pushes it to origin. The hub's ticks resolve the
ref in the hub's own API checkout, so a roll made on another machine, as the
sessions on ziggurat make them, reached the hub only when MCPs maketools
release-sync fetched it, every 900 s. Measured: the roll of 2026-10-24T20:34Z
left serendipity's runner on bd3ae7d48 at 20:37Z (fleet job 8c2a7982), and the
roll of about 2026-10-26T00:13:40Z left lavender's on 3b17480c6 until its
handover at 00:20:50Z, so wiki-check job 47b60226, claimed at 00:14:32Z, ran
without the creator-wiki companion that roll added.

WHAT IS FETCHED. ``git fetch origin +refs/fleet/rolled:refs/fleet/rolled``,
forced as the roll's push is, since a roll may move the ref back. Origin's
ref is the roll: one made anywhere is what the next read here sees, and a
local ref no push carried is put back to origin's.

ONCE PER FIRE BOUNDARY, IN A TURN. Every node runner reads the ref
:data:`fleet.cli.node_serve.HANDOVER_SECONDS` before each boundary and every
launch reads it again a few seconds after, ten node runners on the hub at
once on 2026-10-24. Each read takes :data:`FETCH_TURN` and finds the stamp
:data:`STAMP_NAME` the last fetch left: one younger than
:data:`FRESH_SECONDS` answers for it, so one fetch serves a boundary's reads,
and no two fetches race to move the ref. The stamp and the lock live in the
repository's git directory, so a read is answered only by a fetch of its own
repository.

A FETCH THAT FAILS REFUSES BY NAME. Its refusal is the stamp's text, so every
read inside the window answers with the same :data:`FETCH_FAILED_CODE`
refusal without asking origin again, and the read after it fetches again.
Nothing is read from a ref the fetch could not bring up to date: a tick on
code origin has moved past is the defect this removes.
"""

from __future__ import annotations

import os
import pathlib
from typing import Final

from fleet.core import _test_hooks, commit_tree, file_turn

#: The ref ``make fleet-roll`` sets and pushes; :data:`fleet.core.rolled.ROLLED_REF`.
REF: Final = "refs/fleet/rolled"

#: The remote the roll is pushed to.
REMOTE: Final = "origin"

#: How long one fetch answers for: longer than a boundary's reads span
#: (the handover's 10 s before it and a launch's few after), shorter than the
#: 180 s between boundaries, so each boundary fetches once.
FRESH_SECONDS: Final[int] = 60

#: The fetch's deadline, in seconds: one ref of a commit the checkout mostly
#: holds already.
FETCH_TIMEOUT_SECONDS: Final[int] = 60

#: The deadline of the local ``git rev-parse`` that finds the git directory.
LOCATE_TIMEOUT_SECONDS: Final[int] = 30

#: The stamp the last fetch leaves in the git directory: empty when it moved
#: the ref, its refusal when it did not.
STAMP_NAME: Final = "fleet-rolled-fetch"

#: The lock :data:`FETCH_TURN` is held by, beside the stamp.
LOCK_NAME: Final = "fleet-rolled-fetch.lock"

#: The turn a read fetches or reads the stamp in. Its longest turn is one
#: fetch, so a lock a minute older than the fetch's deadline was left by a
#: read that died holding it.
FETCH_TURN: Final = file_turn.TurnBounds(
    what="rolled-ref fetch turn",
    waited_for="another tick fetching refs/fleet/rolled",
    stale_seconds=FETCH_TIMEOUT_SECONDS + 60,
    poll_seconds=1,
)

#: The refusal when the git directory cannot be found.
UNLOCATED_CODE: Final = "FLEET_ROLL_GIT_DIR_UNREAD"

#: The refusal when the fetch fails.
FETCH_FAILED_CODE: Final = "FLEET_ROLL_FETCH_FAILED"


def fetch_argv(checkout: pathlib.Path) -> tuple[str, ...]:
    """The fetch that brings the checkout's ref to origin's.

    Args:
        checkout: A directory inside the checkout.

    Returns:
        The argv.
    """
    return ("git", "-C", str(checkout), "fetch", "--quiet", REMOTE, f"+{REF}:{REF}")


def git_directory(checkout: pathlib.Path) -> pathlib.Path | str:
    """The git directory every worktree of the checkout's repository shares.

    Args:
        checkout: A directory inside the checkout.

    Returns:
        Its absolute path, or a :data:`UNLOCATED_CODE` refusal.
    """
    located = _test_hooks.run(
        (
            "git",
            "-C",
            str(checkout),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ),
        timeout_seconds=LOCATE_TIMEOUT_SECONDS,
    )
    if located["returncode"] != 0:
        return commit_tree.step_refusal(
            UNLOCATED_CODE, f"git rev-parse --git-common-dir in {checkout}", located
        )
    return pathlib.Path(located["stdout"].strip())


def fetch_rolled(checkout: pathlib.Path) -> str | None:
    """Fetch origin's rolled ref into the checkout, once per fresh window.

    Args:
        checkout: A directory inside the checkout whose ref is about to be
            read.

    Returns:
        None when the ref is origin's as of a fetch younger than
        :data:`FRESH_SECONDS`, or the refusal that fetch, or this one, met.
    """
    directory = git_directory(checkout)
    if isinstance(directory, str):
        return directory
    stamp = directory / STAMP_NAME
    with file_turn.held(
        directory / LOCK_NAME, holder=f"a read of {REF} in {checkout}", bounds=FETCH_TURN
    ):
        now = _test_hooks.now()
        if _fresh(stamp, now=now):
            answered = _test_hooks.read_text(stamp)
            return answered or None
        fetched = _test_hooks.run(fetch_argv(checkout), timeout_seconds=FETCH_TIMEOUT_SECONDS)
        refusal = (
            None
            if fetched["returncode"] == 0
            else commit_tree.step_refusal(
                FETCH_FAILED_CODE, f"git fetch {REMOTE} +{REF}:{REF} in {checkout}", fetched
            )
        )
        _test_hooks.write_text(stamp, refusal or "")
        os.utime(stamp, (now, now))
        return refusal


def _fresh(stamp: pathlib.Path, *, now: int) -> bool:
    """Say whether the last fetch's stamp still answers for one.

    Args:
        stamp: The stamp file.
        now: The current time in whole seconds.

    Returns:
        True when it exists and is younger than :data:`FRESH_SECONDS`.
    """
    if not _test_hooks.file_exists(stamp):
        return False
    return now - int(os.stat(stamp).st_mtime) < FRESH_SECONDS


__all__ = [
    "FETCH_FAILED_CODE",
    "FETCH_TIMEOUT_SECONDS",
    "FETCH_TURN",
    "FRESH_SECONDS",
    "LOCATE_TIMEOUT_SECONDS",
    "LOCK_NAME",
    "REF",
    "REMOTE",
    "STAMP_NAME",
    "UNLOCATED_CODE",
    "fetch_argv",
    "fetch_rolled",
    "git_directory",
]
