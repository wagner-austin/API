"""A span of ``make check`` held to a bound of its own, beside the five-minute budget.

MCPs' check lock carries the rule (``packages/maketools/src/maketools/
check_spans.py``, board task 8bee7131): a span named in its bounds is not
charged to the 300 s budget; the run measures its own wall time and records
it, the check subtracts it from the charge, and the span is held to its own
bound, past which the check fails exactly as a check past the budget does.
This repository has one such span, the mutation step's (MCPs board task
e52e5148), so this module carries that one span with MCPs' name, bound,
file format and words; ``check-budget`` (:mod:`maketools.budget_run`) reads
it, and the test launcher's mutation step records it.

WHY THE MUTATION STEP IS ONE. Its time scales with the change, not the
package: MCPs measured 170 to 326 s for 241 to 250 mutants of four files,
and 299 s for mcp-shared-py's 380. A suite near 300 s could not take a
commit of any size under the budget, while the bound, 2.8 times the slowest
run MCPs measured, still fails a run that has hung.

A span can only be recorded by the run that measured it: ``check-budget``
names a fresh file for each check in :data:`SPANS_ENVIRONMENT_KEY`, so a
caller cannot set or clear anything to make a check pass. Its lines read
``<whole seconds> <epoch seconds it ended> <name>``, MCPs' format.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Final, NamedTuple

from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError

from maketools.check_budget import OVER_BUDGET_EXIT_CODE, budgeted_exit_code

#: The variable naming the check's span file: MCPs' name, so one variable
#: means one thing in both repositories.
SPANS_ENVIRONMENT_KEY: Final[str] = "CORVIS_CHECK_SPANS"

#: The mutation step's span, which :func:`maketools.mutation.launch.span_sink`
#: writes when the lifted step reports its time.
MUTATION_SPAN: Final[str] = "mutation testing"

#: Its bound, in seconds: MCPs' ``check_spans.MUTATION_BOUND_SECONDS``.
MUTATION_BOUND_SECONDS: Final[int] = 900

#: Every span bounded on its own, with its bound in seconds.
SEPARATE_BOUNDS: Final[Mapping[str, int]] = {MUTATION_SPAN: MUTATION_BOUND_SECONDS}

#: One line of the span file.
_SPAN_LINE: Final[re.Pattern[str]] = re.compile(
    r"(?P<seconds>\d+) (?P<ended>\d+(?:\.\d+)?) (?P<name>.+)"
)


class BoundedSpan(NamedTuple):
    """One recorded span and the bound it is held to.

    Attributes:
        name: A key of :data:`SEPARATE_BOUNDS`.
        seconds: The span's measured wall time, in whole seconds.
        bound: Its bound, in seconds.
        ended: When it ended, in seconds since the epoch.
    """

    name: str
    seconds: int
    bound: int
    ended: float


def span_line(name: str, seconds: int, ended: float) -> str:
    """The line one measured span is recorded as.

    Args:
        name: A key of :data:`SEPARATE_BOUNDS`.
        seconds: Its wall time, in whole seconds.
        ended: When it ended, in seconds since the epoch.

    Returns:
        The line, with its newline.
    """
    return f"{seconds} {ended:.3f} {name}\n"


def record_span(path: Path, name: str, seconds: int, ended: float) -> None:
    """Append one measured span to the check's span file.

    Args:
        path: The file :data:`SPANS_ENVIRONMENT_KEY` names.
        name: A key of :data:`SEPARATE_BOUNDS`.
        seconds: Its wall time, in whole seconds.
        ended: When it ended, in seconds since the epoch.
    """
    with path.open("a", encoding="utf-8") as sink:
        sink.write(span_line(name, seconds, ended))


def parse_span(line: str) -> BoundedSpan:
    """Read one line of the span file.

    Args:
        line: ``<whole seconds> <epoch seconds it ended> <name>``.

    Returns:
        The span with its bound.

    Raises:
        AppError: ``MUTATION_SPAN_UNREADABLE`` when the line is not that
            shape or the name has no bound.
    """
    matched = _SPAN_LINE.fullmatch(line.strip())
    if matched is None or matched["name"] not in SEPARATE_BOUNDS:
        raise AppError(
            MutationErrorCode.SPAN_UNREADABLE,
            f"span line {line!r} is not '<whole seconds> <epoch seconds it ended> "
            "<bounded span name>'",
        )
    return BoundedSpan(
        name=matched["name"],
        seconds=int(matched["seconds"]),
        bound=SEPARATE_BOUNDS[matched["name"]],
        ended=float(matched["ended"]),
    )


def read_spans(path: Path) -> tuple[BoundedSpan, ...]:
    """Every span the check's runs recorded.

    Args:
        path: The span file. A check none of whose runs is separately
            bounded never creates it, which reads as no spans.

    Returns:
        The spans, in the order they were recorded.
    """
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(parse_span(line) for line in lines if line.strip())


def covered(spans: Sequence[BoundedSpan]) -> timedelta:
    """How much time the spans cover together, an overlap counted once.

    Args:
        spans: The recorded spans.

    Returns:
        The length of their union.
    """
    total = 0.0
    reach = float("-inf")
    for began, ended in sorted((span.ended - span.seconds, span.ended) for span in spans):
        if ended > reach:
            total += ended - max(began, reach)
            reach = ended
    return timedelta(seconds=total)


def spans_split(split: str, spans: Sequence[BoundedSpan]) -> str:
    """The verdict's split with each span and its bound added.

    Args:
        split: What the check's charge covered.
        spans: The recorded spans.

    Returns:
        The split, then one clause per span.
    """
    clauses = [
        f"{span.name} {span.seconds}s of its own {span.bound}s bound (not counted)"
        for span in spans
    ]
    return "; ".join([split, *clauses])


def span_lines(package: str, spans: Sequence[BoundedSpan]) -> list[str]:
    """The loud block for each span past its bound.

    Args:
        package: The package.
        spans: The recorded spans.

    Returns:
        Nothing when every span is within its bound.
    """
    lines: list[str] = []
    for span in spans:
        if span.seconds <= span.bound:
            continue
        lines.extend(
            [
                "",
                "=" * 72,
                f"CHECK OVER BOUND: {package}'s {span.name} took {span.seconds}s, over its "
                f"{span.bound}s bound by {span.seconds - span.bound}s.",
                "  why   : a span outside the five-minute budget is held to its own bound "
                "(tools/maketools bounded_spans.py, MCPs board task e52e5148).",
                "  fix   : make the run lighter or the change smaller; nothing raises the bound.",
                "=" * 72,
            ]
        )
    return lines


def bounded_exit_code(exit_code: int, charged: timedelta, spans: Sequence[BoundedSpan]) -> int:
    """The code a check with spans exits with.

    Args:
        exit_code: The check's own exit code.
        charged: Its charge, the spans taken out.
        spans: The recorded spans.

    Returns:
        The check's own code when it failed;
        :data:`maketools.check_budget.OVER_BUDGET_EXIT_CODE` when it passed
        and a span is past its bound; otherwise
        :func:`maketools.check_budget.budgeted_exit_code`.
    """
    if exit_code == 0 and any(span.seconds > span.bound for span in spans):
        return OVER_BUDGET_EXIT_CODE
    return budgeted_exit_code(exit_code, charged)


__all__ = [
    "MUTATION_BOUND_SECONDS",
    "MUTATION_SPAN",
    "SEPARATE_BOUNDS",
    "SPANS_ENVIRONMENT_KEY",
    "BoundedSpan",
    "bounded_exit_code",
    "covered",
    "parse_span",
    "read_spans",
    "record_span",
    "span_line",
    "span_lines",
    "spans_split",
]
