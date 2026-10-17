"""The commands every hub tick runs after its claim, for the suites that script a tick.

Two of the hub agent's passes read ``refs/fleet/rolled`` in the tick's
``--repo-root`` after the claim pass: the refuse pass (MCPs board task
1915b042, :mod:`fleet.core.declared`) first, and the reap pass last (MCPs
board task 715b9ec5, :mod:`fleet.core.runner_reap_dropped`), with the observe
pass between them when a tick is given a registry. The test repo root has no
such ref, so each read resolves nothing, and the queue is sent no registry
and no Worker is read. A suite that scripts every command a tick runs
scripts :data:`UNROLLED` for each read, :data:`UNROLLED_TICK` for a tick that
observes nothing, and one that asserts them names :func:`unrolled_calls`.
"""

from __future__ import annotations

import pathlib
from typing import Final

from fleet.core import _test_hooks, rolled

#: What ``git rev-parse`` answers in a tree with no rolled ref.
UNROLLED: Final[_test_hooks.CommandResult] = _test_hooks.CommandResult(
    returncode=128, stdout="", stderr="fatal: Needed a single revision", timed_out=False
)

#: The answers of a tick that observes nothing: the publish pass's read of
#: the ref, then the reap pass's.
UNROLLED_TICK: Final[tuple[_test_hooks.CommandResult, ...]] = (UNROLLED, UNROLLED)


def unrolled_call(repo: pathlib.Path) -> tuple[str, ...]:
    """One hub pass's read of the rolled ref in its repo root.

    Args:
        repo: The ``--repo-root`` the tick was given.

    Returns:
        The argv, with the root resolved as the agent resolves it.
    """
    return (
        "git",
        "-C",
        str(repo.resolve()),
        "rev-parse",
        "--verify",
        f"{rolled.ROLLED_REF}^{{commit}}",
    )


def unrolled_calls(repo: pathlib.Path) -> tuple[tuple[str, ...], ...]:
    """The reads of a tick that observes nothing, in the order it makes them.

    Args:
        repo: The ``--repo-root`` the tick was given.

    Returns:
        The publish pass's read, then the reap pass's.
    """
    return unrolled_call(repo), unrolled_call(repo)


__all__ = ["UNROLLED", "UNROLLED_TICK", "unrolled_call", "unrolled_calls"]
