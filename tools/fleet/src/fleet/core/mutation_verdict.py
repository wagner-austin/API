"""The mutation step's outcome, read off a check's transcript (MCPs board task 637f648f).

MCPs' StrykerJS step (``mcp-shared/src/mutation``), MCPs' Python step
(``mcp_shared_py.mutation``) and API's (``maketools.mutation``) each end a
package's suite by mutation testing the source lines its commit added, and
each prints one summary line::

    mutation: FAILED: 8 of 453 mutant(s) made in 18 src file(s) changed against ...
    mutation: passed: 0 of 275 mutant(s) made in 40 src file(s) changed against ...
    mutation: not applicable: 0 lines added to src files against origin/main ...

or, when it refused or its tool failed, the named error it raised, at the
start of a line as node and Python print an uncaught one::

    MutationStepError: MUTATION_STRYKER_FAILED: stryker run exited 1; ...
    mcp_shared_py.mutation.context.MutationStepError: MUTATION_BASELINE_FAILED: ...

and maketools' check budget line carries the step's span against its own
bound (``mutation testing 801s of its own 900s bound (not counted)``).

osrs-lab's build (``build-runner/check-summary.gradle``) prints the same
summary for its PIT steps and ends it with the steps' own seconds, which no
bound governs, and the verdicts their kept histories supplied (MCPs board
task 0546e41a)::

    mutation: passed: 0 of 2041 mutant(s) made in 1 mutation step(s): :mutationTest; 289s ...
    ... mutation testing, 0 verdict(s) reused from kept history

(one line, folded here at the dots).

A history lets a check skip every mutant whose class and covering tests did
not change, and until this reading what it saved was in the transcript
alone: the audit of that task's closure could decide neither that a warm
check reused verdicts nor how long its step ran.

WHY THE VERDICT LINE CARRIES IT. A check whose suite passed at 100 percent
coverage and whose mutation step then failed exits 2 like any other red
check, and until this field the queue row said nothing more: of the 275
fleet checks that failed in the 48 hours to 2026-11-08T23:08Z, 47 failed in
the mutation step (38 on surviving mutants, 9 on a refusal or a deadline),
and telling them apart meant reading each transcript on its node. Two
reopens of MCPs task e6c9604f rest on the same gap: a criterion that a
package completes the step could be decided by no query.

Each pattern is anchored at a line's start, so a test that prints a step's
line inside an assertion or a diff (``+     "mutation: FAILED: ...``, as the
step's own suites do) is not read as the step's outcome.
"""

from __future__ import annotations

import re
from typing import Final

#: The step's summary of a run that made mutants: how many survived of how
#: many were made.
SUMMARY: Final[re.Pattern[str]] = re.compile(
    r"^mutation: (passed|FAILED): (\d+) of (\d+) mutant\(s\) made in ", re.MULTILINE
)

#: The step's summary of a change that gave it nothing to mutate.
NOT_APPLICABLE: Final[re.Pattern[str]] = re.compile(r"^mutation: not applicable: ", re.MULTILINE)

#: The named error a step that refused or failed raised, as its runtime
#: printed it uncaught.
REFUSAL: Final[re.Pattern[str]] = re.compile(
    r"^(?:[\w.]+\.)?MutationStepError: (MUTATION_[A-Z_]+): ", re.MULTILINE
)

#: The step's span on maketools' check budget line, against its own bound.
SPAN: Final[re.Pattern[str]] = re.compile(r"mutation testing (\d+)s of its own (\d+)s bound")

#: A summary that ends with its steps' seconds and the verdicts their kept
#: histories supplied, as osrs-lab's build prints it.
HISTORY: Final[re.Pattern[str]] = re.compile(
    r"^mutation: (?:passed|FAILED): \d+ of \d+ mutant\(s\) made in .*; (\d+)s mutation testing, "
    r"(\d+) verdict\(s\) reused from kept history$",
    re.MULTILINE,
)

#: What the field, or either of its two parts, says when the tail does not hold it.
UNREAD: Final = "unread"


def _last(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """The last match of a pattern in a text.

    Args:
        pattern: The pattern.
        text: The text.

    Returns:
        The match that starts last, or None when nothing matches.
    """
    last: re.Match[str] | None = None
    for match in pattern.finditer(text):
        last = match
    return last


def _outcome(tail: str) -> str:
    """How the mutation step ended, or the empty string when the tail does not say.

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        ``passed:<survived>/<made>``, ``failed:<survived>/<made>``,
        ``not-applicable`` or ``error:<code>``, else ``""``.
    """
    ending = _ending(tail)
    if ending is None:
        return ""
    return ending[1]


def _ending(tail: str) -> tuple[int, str] | None:
    """Where the line the mutation step ended on starts, and what it says.

    The outcome is the LAST of the step's summary, not-applicable and error
    lines in the tail, so a check that ran the step twice is judged by the
    run that ended it. The summary's word is lower-cased rather than
    compared, since ``FAILED`` and ``passed`` are its only two values.

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        The line's offset in the tail and its outcome as :func:`_outcome`
        words it, or None when the tail holds none of the step's lines.
    """
    found: list[tuple[int, str]] = []
    summary = _last(SUMMARY, tail)
    if summary is not None:
        word = f"{summary.group(1)}".lower()
        found.append((summary.start(), f"{word}:{summary.group(2)}/{summary.group(3)}"))
    empty = _last(NOT_APPLICABLE, tail)
    if empty is not None:
        found.append((empty.start(), "not-applicable"))
    refusal = _last(REFUSAL, tail)
    if refusal is not None:
        found.append((refusal.start(), f"error:{refusal.group(1)}"))
    if not found:
        return None
    return max(found)


def _span(tail: str) -> str:
    """The step's span, or the empty string when no line names it.

    The span is the LAST of a check budget line's and a summary's own
    seconds, as the outcome is the last of the step's lines.

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        ``<seconds>s/<bound>s`` from a check budget line, ``<seconds>s``
        from a summary stating its steps' seconds, else ``""``.
    """
    found: list[tuple[int, str]] = []
    budget = _last(SPAN, tail)
    if budget is not None:
        found.append((budget.start(), f"{budget.group(1)}s/{budget.group(2)}s"))
    history = _last(HISTORY, tail)
    if history is not None:
        found.append((history.start(), f"{history.group(1)}s"))
    if not found:
        return ""
    return max(found)[1]


def _reused(tail: str) -> str:
    """The verdicts the step's kept history supplied, or the empty string.

    Read only off the line the step ended on, so a check whose last outcome
    came from a step keeping no history reports no reuse from an earlier one.

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        ``reused:<count>``, else ``""``.
    """
    ending = _ending(tail)
    history = _last(HISTORY, tail)
    if ending is None or history is None or history.start() != ending[0]:
        return ""
    return f"reused:{history.group(2)}"


def read_outcome(tail: str) -> str:
    """Read how the mutation step ended off a transcript tail.

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        ``passed:<survived>/<made>`` or ``failed:<survived>/<made>`` from
        the summary, ``not-applicable`` for a change that made no mutant,
        ``error:<code>`` for a step that raised its named error, else
        :data:`UNREAD`.
    """
    return _outcome(tail) or UNREAD


def read_span(tail: str) -> str:
    """Read the mutation step's span off a transcript tail.

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        ``<seconds>s/<bound>s`` from a check budget line or ``<seconds>s``
        from a summary stating its steps' seconds, whichever is last, else
        :data:`UNREAD`.
    """
    return _span(tail) or UNREAD


def read_mutation(tail: str) -> str:
    """Read the mutation step's outcome and span off a transcript tail.

    An absent part is the empty string until it is written, so the test of
    whether the tail held either is a truth test and not a comparison with
    :data:`UNREAD` (fleet job 5f5e38ba: ``==`` there could be mutated to
    ``>=`` or ``is`` with no reading changed, ``unread`` sorting after every
    outcome and span and being one interned string).

    Args:
        tail: The transcript's last lines, colours stripped.

    Returns:
        ``<outcome>,<span>`` (:func:`read_outcome`, :func:`read_span`), for
        example ``failed:8/453,801s/900s``, either part :data:`UNREAD` when
        the tail lacks it, or :data:`UNREAD` alone for a check whose tail
        holds neither, which is every check of a package with no mutation
        step. A step that keeps a history adds a third part,
        ``reused:<count>``, as in ``passed:0/2041,289s,reused:0``; a step
        that keeps none has no reuse to report, so the part is absent rather
        than unread.
    """
    outcome = _outcome(tail)
    span = _span(tail)
    if not (outcome or span):
        return UNREAD
    reused = _reused(tail)
    reading = f"{outcome or UNREAD},{span or UNREAD}"
    if not reused:
        return reading
    return f"{reading},{reused}"


__all__ = [
    "HISTORY",
    "NOT_APPLICABLE",
    "REFUSAL",
    "SPAN",
    "SUMMARY",
    "UNREAD",
    "read_mutation",
    "read_outcome",
    "read_span",
]
