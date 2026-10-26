"""The two commands a read of the rolled ref runs first, for scripted suites.

Every read of ``refs/fleet/rolled`` fetches origin's ref first
(:mod:`fleet.core.rolled_fetch`, MCPs board task 8db0f536): it finds the git
directory, then, with no fresh stamp there, fetches. The suites that script
the extraction's commands (``test_rolled``, ``test_cli_rolled``) script
these two ahead of it, against a git directory the test makes.
"""

from __future__ import annotations

import pathlib
from typing import Final

from fleet.core import _test_hooks, rolled_fetch
from tests.conftest import ok


def fetch_calls(checkout: pathlib.Path) -> list[tuple[str, ...]]:
    """The argv of each command a read with no fresh stamp runs first.

    Args:
        checkout: The directory the read was given.

    Returns:
        ``git rev-parse --git-common-dir`` and ``git fetch``, in order.
    """
    return [
        (
            "git",
            "-C",
            str(checkout),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ),
        rolled_fetch.fetch_argv(checkout),
    ]


def fetch_replies(git_directory: pathlib.Path) -> list[_test_hooks.CommandResult]:
    """The answers of a read whose fetch succeeds, its git directory made.

    Args:
        git_directory: Where the scripted ``rev-parse`` says the git
            directory is; created, since the fetch turn's lock and the stamp
            are written there.

    Returns:
        The two replies, in order.
    """
    git_directory.mkdir(parents=True, exist_ok=True)
    return [ok(f"{git_directory}\n"), ok("")]


#: The deadlines of the two commands, in order.
FETCH_TIMEOUTS: Final[tuple[int, int]] = (
    rolled_fetch.LOCATE_TIMEOUT_SECONDS,
    rolled_fetch.FETCH_TIMEOUT_SECONDS,
)

__all__ = ["FETCH_TIMEOUTS", "fetch_calls", "fetch_replies"]
