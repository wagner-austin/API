"""The command every hub tick runs after its claim, for the suites that script a tick.

The hub agent's refuse pass (MCPs board task 1915b042, :mod:`fleet.core.declared`)
reads ``refs/fleet/rolled`` in the tick's ``--repo-root`` after the claim pass.
The test repo root has no such ref, so the read resolves nothing and the
queue is sent no registry; a suite that scripts every command a tick runs
scripts :data:`UNROLLED` for it, and one that asserts them names
:func:`unrolled_call` last.
"""

from __future__ import annotations

import pathlib
from typing import Final

from fleet.core import _test_hooks, rolled

#: What ``git rev-parse`` answers in a tree with no rolled ref.
UNROLLED: Final[_test_hooks.CommandResult] = _test_hooks.CommandResult(
    returncode=128, stdout="", stderr="fatal: Needed a single revision", timed_out=False
)


def unrolled_call(repo: pathlib.Path) -> tuple[str, ...]:
    """The hub tick's read of the rolled ref in its repo root.

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


__all__ = ["UNROLLED", "unrolled_call"]
