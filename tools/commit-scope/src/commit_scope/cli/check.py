"""The command a pre-commit hook runs.

Wires the three seams to the pure decision and turns the result into an exit
code. Everything decidable lives in :mod:`commit_scope.scope` and
:mod:`commit_scope.report`; this module owns only the order of operations and
the mapping from decision to status.

EXIT CODES ARE THE INTERFACE. ``0`` lets the commit proceed, ``1`` stops it.
An :class:`AppError` is not caught here: it propagates to
:func:`entrypoint`, which reports its code and message and exits ``2``, so a
broken environment is distinguishable from a refused commit. A hook that
returned ``1`` for both would teach the author that this check is noisy, and a
check people route around protects nothing.
"""

from __future__ import annotations

from platform_core.errors import AppError

from commit_scope import _test_hooks
from commit_scope.config import (
    REPO_ROOT_ARGUMENTS,
    SCOPE_VARIABLE,
    STAGED_PATHS_ARGUMENTS,
)
from commit_scope.contracts import (
    ScopeQuestion,
    decode_scope_declaration,
    decode_staged_paths,
)
from commit_scope.report import render
from commit_scope.scope import decide, refuses

#: Status when the commit may proceed.
EXIT_OK = 0
#: Status when the commit carries undeclared paths.
EXIT_REFUSED = 1
#: Status when the question could not be asked at all.
EXIT_ERROR = 2


def gather() -> ScopeQuestion:
    """Ask git and the environment what this commit is.

    The repository-root call runs first and its output is discarded: it exists
    to fail with a code naming the real problem when the hook is run outside a
    work tree, instead of letting the index query fail and send the reader
    looking at staging.

    Returns:
        The staged set and the declared scope, both decoded and validated.

    Raises:
        AppError: When git cannot answer, or when the declaration contains an
            entry that could never match a staged path.
    """
    _test_hooks.run_git(REPO_ROOT_ARGUMENTS)
    return {
        "staged": decode_staged_paths(_test_hooks.run_git(STAGED_PATHS_ARGUMENTS)),
        "scope": decode_scope_declaration(_test_hooks.env(SCOPE_VARIABLE)),
    }


def main() -> int:
    """Check the index against the declaration and report.

    Returns:
        :data:`EXIT_OK` or :data:`EXIT_REFUSED`.

    Raises:
        AppError: Propagated from :func:`gather`; :func:`entrypoint` renders
            it. Not handled here, because a check that softened a broken git
            into a pass would wave through exactly the commits it exists to
            stop.
    """
    decision = decide(gather())
    for line in render(decision):
        _test_hooks.emit(line)
    return EXIT_REFUSED if refuses(decision) else EXIT_OK


def entrypoint() -> int:
    """Console-script wrapper that renders a failure instead of a traceback.

    The single ``except`` in this package, and it is at the process boundary
    rather than in the logic: it does not recover, retry, or soften anything.
    It turns an already-final failure into the two lines an author can act on
    and a distinct status, then stops.

    Returns:
        :data:`EXIT_OK`, :data:`EXIT_REFUSED`, or :data:`EXIT_ERROR` when the
        question could not be asked.
    """
    try:
        return main()
    except AppError as error:
        # ErrorCodeBase subclasses str, so a member IS its own string value.
        # Annotating rather than reading ``.value`` keeps the expression
        # typed: ``Enum.value`` is Any, and this package admits no Any.
        code: str = error.code
        # CONCATENATED, not interpolated. An f-string calls
        # ``Enum.__format__``, which renders ``CommitScopeErrorCode.GIT_...``
        # -- a qualified name nobody can grep the codebase or the board for.
        # ``str.__add__`` uses the value, which is the half that is stable.
        _test_hooks.emit("=== commit-scope: " + code + " ===")
        _test_hooks.emit(error.message)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(entrypoint())
