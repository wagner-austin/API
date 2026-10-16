"""Which checkout's copy of a directory every worktree of a repository shares.

WHAT THIS FIXES, measured on 2026-10-05 (board task 747c391e). The workspace
document names its ledger relative to its own directory, ``runs/``, and
``runs/*`` is gitignored, so every ``git worktree`` of the repository kept a
ledger of its own: the main checkout held 16030 rows and none for the v56
cartridge-QA arrays, ``API-research-w3`` held the 20 rows for arrays
57783112/57783131/57783373, and ``API-research-w2`` held none. A worker
resuming that campaign in w2 was told ``0 in flight`` while 12 members were
running or pending, because the in-flight check reaches a live job's artifact
only through a ledger row, and it submitted a duplicate array (57783755).

THE RULE. A worktree's ledger is the MAIN checkout's ledger: the same relative
directory, under the working tree whose ``.git`` is the repository's common
directory. Every linked worktree resolves to it, so one file records every
submission made from any of them, and no worktree can see less than another.

How a checkout is recognised, which is git's own layout and not a convention
of this package:

* A directory whose ``.git`` is a DIRECTORY is a main checkout, so the
  directory itself is the answer.
* A directory whose ``.git`` is a FILE carries ``gitdir: <path>``. When that
  git directory holds a ``commondir`` file, it is a linked worktree and the
  common directory is named there; the main checkout is the common
  directory's parent. When it holds none, it is a submodule's checkout,
  which is its own main checkout.
* A directory with no ``.git`` in it or above it belongs to no repository,
  and nothing else can be sharing its ledger, so it is its own answer.

WHAT THIS DOES NOT REACH: a second MACHINE. A checkout on another host keeps
its own main checkout and so its own ledger. That gap is closed at
submission, not here: :func:`hpc3.core.inflight.check_no_unrecorded_live_job`
refuses to submit beside a live job that carries the submission's name and
that this ledger has no row for, wherever it came from.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core import _test_hooks

_GITDIR_PREFIX = "gitdir: "
_COMMON_DIR_NAME = ".git"


def _read_line(path: pathlib.Path) -> str:
    """Read the one line a git pointer file carries.

    Args:
        path: A ``.git`` file or a ``commondir`` file.

    Returns:
        Its text without the trailing newline.
    """
    return _test_hooks.read_bytes(path).decode("utf-8").strip()


def _linked_main_root(dot_git: pathlib.Path) -> pathlib.Path | None:
    """Find the main checkout a ``.git`` FILE points back to.

    Args:
        dot_git: A ``.git`` file at the root of a checkout.

    Returns:
        The main checkout's root for a linked worktree, or None for a
        submodule's checkout, whose git directory names no common directory
        because it shares nothing.

    Raises:
        AppError: With ``LEDGER_CHECKOUT_UNREADABLE`` if the file does not
            start ``gitdir: ``, or the common directory it reaches is not
            named ``.git`` -- the main working tree of a repository whose
            common directory is a bare repository does not exist, so there
            is no single checkout whose ledger the worktrees could share.
    """
    pointer = _read_line(dot_git)
    if not pointer.startswith(_GITDIR_PREFIX):
        raise AppError(
            Hpc3ErrorCode.LEDGER_CHECKOUT_UNREADABLE,
            f"{dot_git} does not start {_GITDIR_PREFIX!r}, so which repository this "
            f"checkout belongs to cannot be read; it carries {pointer!r}.",
        )
    git_dir = dot_git.parent / pointer[len(_GITDIR_PREFIX) :]
    commondir = git_dir / "commondir"
    if not _test_hooks.file_exists(commondir):
        return None
    common = (git_dir / _read_line(commondir)).resolve()
    if common.name != _COMMON_DIR_NAME:
        raise AppError(
            Hpc3ErrorCode.LEDGER_CHECKOUT_UNREADABLE,
            f"{dot_git} is a worktree of {common}, which is not a checkout's "
            f"{_COMMON_DIR_NAME} directory, so no main checkout holds the ledger "
            "every worktree would share. Run hpc3 from a repository with a main "
            "working tree.",
        )
    return common.parent


def shared_directory(directory: pathlib.Path) -> pathlib.Path:
    """Map a directory to the same directory in its repository's main checkout.

    Args:
        directory: An absolute directory, normally the one the workspace
            document was read from.

    Returns:
        The directory at the same path relative to the main checkout's root
        when ``directory`` is inside a linked worktree, and ``directory``
        itself otherwise.

    Raises:
        AppError: With ``LEDGER_CHECKOUT_UNREADABLE`` if a ``.git`` file on
            the way up cannot be read as a git pointer.
    """
    for root in (directory, *directory.parents):
        dot_git = root / _COMMON_DIR_NAME
        if dot_git.is_dir():
            return directory
        if _test_hooks.file_exists(dot_git):
            main_root = _linked_main_root(dot_git)
            if main_root is None:
                return directory
            return main_root / directory.relative_to(root)
    return directory


__all__ = ["shared_directory"]
