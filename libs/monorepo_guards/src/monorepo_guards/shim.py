"""The guard entry point, written once instead of forty-one times.

Every package carries a ``scripts/guard.py`` so that ``make check`` can run
``python -m scripts.guard`` the same way everywhere. That file cannot simply
import this package -- ``monorepo_guards`` is a dependency of only four of
the forty-one -- so it has to put this package on the path before it can
hand over. That bootstrap is the ONLY thing that legitimately lives in each
package. Everything after it lives here.

It did not start that way. Forty-one hand-maintained copies of the same
sixty lines drifted into twenty-one distinct variants: some routing their
plumbing through ``scripts/_test_hooks.py``, some through a module-level
seam, some through neither. Their tests drifted further, into four separate
shapes of assertion that could not fail (``rc in (0, 2)``, ``rc == 0 or
rc > 0``, ``rc >= 0``, and an ``isinstance`` ternary that laundered a string
exit code into success). Seventy such assertions, all green, none of them
checking that the guard passed.

So the argument parsing, the root override and the verbose line are here,
covered once, and the per-package file is reduced to the bootstrap it always
should have been. The ``guard-shim-not-canonical`` rule keeps all forty-one
byte-identical, because that is the property that stops the drift from
starting again.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from monorepo_guards.orchestrator import run_for_project

ROOT_FLAG = "--root"
"""Flag selecting a different tree to check than the calling package."""

VERBOSE_FLAGS = ("-v", "--verbose")
"""Flags that make the exit code visible on stdout."""

EXIT_CODE_PREFIX = "guard_exit_code code="
"""Prefix of the verbose line, asserted by callers that parse it."""


class GuardArguments:
    """What the guard was asked to do.

    Attributes:
        root_override: Tree to check instead of the calling package, or None.
        verbose: Whether to write the exit code to stdout.
    """

    __slots__ = ("root_override", "verbose")

    def __init__(self, *, root_override: Path | None, verbose: bool) -> None:
        """Record the parsed arguments.

        Args:
            root_override: Tree to check instead of the calling package.
            verbose: Whether to write the exit code to stdout.
        """
        self.root_override = root_override
        self.verbose = verbose


def parse_arguments(args: Sequence[str]) -> GuardArguments:
    """Read the guard's two flags, ignoring anything else.

    Unknown tokens are skipped rather than rejected. That is deliberate and
    long-standing: this runs from a Makefile line shared by forty-one
    packages, and a guard that refused an unexpected argument would fail the
    build for a reason having nothing to do with the code it checks.

    Args:
        args: Arguments excluding the program name.

    Returns:
        The parsed arguments.
    """
    root_override: Path | None = None
    verbose = False
    index = 0
    while index < len(args):
        token = args[index]
        if token == ROOT_FLAG and index + 1 < len(args):
            root_override = Path(args[index + 1]).resolve()
            index += 2
        else:
            verbose = verbose or token in VERBOSE_FLAGS
            index += 1
    return GuardArguments(root_override=root_override, verbose=verbose)


def monorepo_root_of(project_root: Path) -> Path:
    """Locate the monorepo root from a package inside it.

    Every package sits at ``<repo>/<category>/<package>``, so the root is two
    levels up -- measured across all forty-one, not assumed. The predecessor
    walked up looking for a ``libs`` directory, which is the same answer
    reached by a loop and an error branch that each package then had to cover.

    Args:
        project_root: Directory of the package being checked.

    Returns:
        The monorepo root.

    Raises:
        RuntimeError: If the result holds no ``libs`` directory, meaning the
            package is not where the layout says it is. Raising beats
            searching: a guard that silently checks the wrong tree reports
            success for code it never read.
    """
    candidate = project_root.parents[1]
    if not (candidate / "libs").is_dir():
        raise RuntimeError(
            f"{project_root} is not <repo>/<category>/<package>: "
            f"expected a libs directory in {candidate}"
        )
    return candidate


def run_guard(argv: Sequence[str] | None, *, project_root: Path) -> int:
    """Run every guard rule against a package.

    Args:
        argv: Arguments excluding the program name, or None to read the
            process arguments.
        project_root: Directory of the package whose ``scripts/guard.py``
            called this.

    Returns:
        0 when no violations were found, 2 when there were.
    """
    arguments = parse_arguments(list(argv) if argv is not None else list(sys.argv[1:]))
    target = arguments.root_override if arguments.root_override is not None else project_root
    code = run_for_project(
        monorepo_root=monorepo_root_of(project_root),
        project_root=target,
    )
    if arguments.verbose:
        sys.stdout.write(f"{EXIT_CODE_PREFIX}{code}\n")
    return code


__all__ = [
    "EXIT_CODE_PREFIX",
    "ROOT_FLAG",
    "VERBOSE_FLAGS",
    "GuardArguments",
    "monorepo_root_of",
    "parse_arguments",
    "run_guard",
]
