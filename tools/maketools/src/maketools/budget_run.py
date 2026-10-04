"""``check-budget``: one package's whole ``make check`` timed against five minutes.

The operator, 2026-10-04: "ideally make check should be like 5 min for
every repo" (MCPs board task 1b152218). Every package Makefile here spells
``check:`` as this command then the pass banner, with the check's old
prerequisites moved to ``_check-unbudgeted``, which this runs and times
whole: no lock serialises an API package's check, so there is no queue to
leave out of the count. The limit, the exit code and the words a reader
sees are :mod:`maketools.check_budget`, which is MCPs' rule lifted byte for
byte (:mod:`maketools.lift`), never a second copy of it.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from maketools import _test_hooks
from maketools.check_budget import (
    UNBUDGETED_TARGET,
    WHOLE_CHECK_SPLIT,
    budgeted_exit_code,
    verdict_lines,
)
from maketools.workspace import FANOUT_WALL_SECONDS


def run_check_budget(package_directory: Path, package: str) -> int:
    """Run and time the package's ``_check-unbudgeted``, then judge it.

    Args:
        package_directory: The package whose check runs, where make is.
        package: Its path from the repository root, as the verdict names it.

    Returns:
        The child make's exit code, or
        :data:`maketools.check_budget.OVER_BUDGET_EXIT_CODE` when it passed
        past the budget.
    """
    started = _test_hooks.now()
    exit_code = _test_hooks.run_inheriting(
        ["make", UNBUDGETED_TARGET],
        cwd=package_directory,
        env=_test_hooks.environ(),
        new_session=False,
        timeout_seconds=FANOUT_WALL_SECONDS,
    )
    charged = timedelta(seconds=_test_hooks.now() - started)
    for line in verdict_lines(package, charged, WHOLE_CHECK_SPLIT):
        _test_hooks.write_line(line)
    return budgeted_exit_code(exit_code, charged)


__all__ = ["run_check_budget"]
