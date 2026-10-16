"""Which commit a checkout holds, and whether its tracked files still match it.

The hub's pump records, for each wake bridge it runs, the repository and
commit the bridge's code came from (MCPs board task 9f1367d5): a checkout's
HEAD alone does not say what ran when the tracked files were edited after it,
so the record carries that too. Untracked files are left out, because every
bridge keeps its own state (``runs/``, cursors) untracked beside its code,
and a checkout that always read dirty would say nothing.
"""

from __future__ import annotations

import pathlib
from typing import Final, TypedDict

from platform_core.errors import AppError, ErrorCode

from fleet.core import _test_hooks
from fleet.core.commit_tree import TREE_STEP_TIMEOUT_SECONDS, resolve_commit, step_refusal

#: The remote whose last path segment names a checkout's repository.
ORIGIN_REMOTE: Final = "origin"

#: The refusal prefix for a checkout this module cannot read.
CHECKOUT_UNREADABLE: Final = "CHECKOUT_UNREADABLE"


class Checkout(TypedDict):
    """A checkout as it stood when read.

    Attributes:
        repo: The last segment of its ``origin`` remote, ``.git`` removed.
        head_sha: The commit its HEAD names.
        dirty: Whether any tracked file differs from that commit.
    """

    repo: str
    head_sha: str
    dirty: bool


def repository_name(remote_url: str) -> str:
    """Name a repository by its remote's last path segment.

    Args:
        remote_url: The remote as git prints it: an https URL, an scp-like
            ``host:owner/name`` or a path.

    Returns:
        The segment after the last ``/`` or ``:``, without ``.git``; an
        empty string when the url ends in a separator.
    """
    segment = remote_url.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return segment.rsplit(":", 1)[-1].removesuffix(".git")


def _git_output(directory: pathlib.Path, *arguments: str) -> str:
    """Run one read-only git command in the checkout.

    Args:
        directory: Any directory inside the checkout.
        arguments: The git arguments.

    Returns:
        Its standard output.

    Raises:
        AppError: ``CONFIG_ERROR`` when git exits non-zero, naming the
            command, its exit and its stderr.
    """
    argv = ("git", "-C", str(directory), *arguments)
    result = _test_hooks.run(argv, timeout_seconds=TREE_STEP_TIMEOUT_SECONDS)
    if result["returncode"] != 0:
        raise AppError(
            code=ErrorCode.CONFIG_ERROR,
            message=step_refusal(CHECKOUT_UNREADABLE, " ".join(argv), result),
        )
    return result["stdout"]


def read_checkout(directory: pathlib.Path) -> Checkout:
    """Read the checkout a directory belongs to.

    Args:
        directory: Any directory inside the checkout.

    Returns:
        Its repository, HEAD commit and whether a tracked file differs.

    Raises:
        AppError: ``CONFIG_ERROR`` when the directory is not inside a
            checkout git can read, its HEAD names no commit, it has no
            ``origin`` remote, or that remote names no repository: each is a
            directory the caller was configured with that is not a checkout
            of a known repository.
    """
    resolved = resolve_commit(directory, "HEAD", CHECKOUT_UNREADABLE)
    if isinstance(resolved, str):
        raise AppError(code=ErrorCode.CONFIG_ERROR, message=resolved)
    remote = _git_output(directory, "remote", "get-url", ORIGIN_REMOTE)
    repo = repository_name(remote)
    if repo == "":
        raise AppError(
            code=ErrorCode.CONFIG_ERROR,
            message=(
                f"{CHECKOUT_UNREADABLE}: the {ORIGIN_REMOTE} remote of {directory} "
                f"is {remote.strip()!r}, which names no repository"
            ),
        )
    changes = _git_output(directory, "status", "--porcelain", "--untracked-files=no")
    return Checkout(repo=repo, head_sha=resolved["commit"], dirty=changes.strip() != "")


__all__ = ["CHECKOUT_UNREADABLE", "ORIGIN_REMOTE", "Checkout", "read_checkout", "repository_name"]
