"""``check-budget``: one package's whole ``make check`` timed against five minutes.

The operator, 2026-10-04: "ideally make check should be like 5 min for
every repo" (MCPs board task 1b152218). Every package Makefile here spells
``check:`` as this command then the pass banner, with the check's old
prerequisites moved to ``_check-unbudgeted``, which this runs and times
whole: no lock serialises an API package's check, so there is no queue to
leave out of the count. The limit, the exit code and the words a reader
sees are :mod:`maketools.check_budget`, which is MCPs' rule lifted byte for
byte (:mod:`maketools.lift`), never a second copy of it.

A SPAN BOUNDED ON ITS OWN is not charged (:mod:`maketools.bounded_spans`):
each check names a fresh span file under the package's ``runs/`` in
``CORVIS_CHECK_SPANS``, the mutation step records its time there, and the
check subtracts it from the charge and holds it to its own bound.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Final

from maketools import _test_hooks
from maketools.bounded_spans import (
    SPANS_ENVIRONMENT_KEY,
    bounded_exit_code,
    covered,
    read_spans,
    span_lines,
    spans_split,
)
from maketools.check_budget import UNBUDGETED_TARGET, WHOLE_CHECK_SPLIT, verdict_lines
from maketools.workspace import FANOUT_WALL_SECONDS

#: The span file's name under the package's ``runs/``, before the check's token.
SPANS_PREFIX: Final[str] = ".check-spans-"


def run_check_budget(package_directory: Path, package: str) -> int:
    """Run and time the package's ``_check-unbudgeted``, then judge it.

    Args:
        package_directory: The package whose check runs, where make is.
        package: Its path from the repository root, as the verdict names it.

    Returns:
        The child make's exit code, or
        :data:`maketools.check_budget.OVER_BUDGET_EXIT_CODE` when it passed
        past the budget or a span passed its own bound.
    """
    runs = package_directory.resolve() / "runs"
    runs.mkdir(exist_ok=True)
    spans_file = runs / (SPANS_PREFIX + _test_hooks.token())
    environment = _test_hooks.environ()
    environment[SPANS_ENVIRONMENT_KEY] = str(spans_file)
    started = _test_hooks.now()
    try:
        exit_code = _test_hooks.run_inheriting(
            ["make", UNBUDGETED_TARGET],
            cwd=package_directory,
            env=environment,
            new_session=False,
            timeout_seconds=FANOUT_WALL_SECONDS,
        )
        whole = timedelta(seconds=_test_hooks.now() - started)
        spans = read_spans(spans_file)
    finally:
        if spans_file.is_file():
            _test_hooks.remove_file(spans_file)
    charged = whole - covered(spans)
    for line in verdict_lines(package, charged, spans_split(WHOLE_CHECK_SPLIT, spans)):
        _test_hooks.write_line(line)
    for line in span_lines(package, spans):
        _test_hooks.write_line(line)
    return bounded_exit_code(exit_code, charged, spans)


__all__ = ["run_check_budget"]
