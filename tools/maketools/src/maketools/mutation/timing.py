"""Bounded cost reports for mutation baselines and trials, independent of verdicts.

The shared runner and API's lifted core use only the standard library.
Timings include process startup and preparation, since both cost the caller
time. They never select tests, change deadlines, or decide a mutant's result.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from types import TracebackType
from typing import NamedTuple


class Phase(StrEnum):
    """The two kinds of test invocation in a mutation run."""

    BASELINE = "baseline"
    MUTANT = "mutant"


class Scope(StrEnum):
    """Whether the invocation selected named tests or the whole suite."""

    SELECTED = "selected"
    WHOLE_SUITE = "whole-suite"


class Trial(NamedTuple):
    """One completed or refused invocation, measured on a monotonic clock."""

    phase: Phase
    scope: Scope
    seconds: float
    identity: str


def _descending_seconds(trial: Trial) -> float:
    """Return the sort key that puts the longest invocation first.

    Args:
        trial: An invocation's recorded cost.

    Returns:
        Its negated duration; equal durations retain invocation order.
    """
    return -trial.seconds


class Measurement:
    """One invocation being measured: its start read when made, its cost recorded on exit.

    A class rather than a ``@contextmanager`` generator because API, which
    lifts this module byte for byte, forbids importing ``Iterator`` (its
    ``import-collections-iterator`` guard, board task 9976dca1), and a
    generator context manager cannot be annotated without it. Its exit
    returns nothing, so an invocation's error propagates unchanged.
    """

    def __init__(
        self,
        now: Callable[[], float],
        record: Callable[[Trial], None],
        phase: Phase,
        scope: Scope,
        identity: str,
    ) -> None:
        """Read the clock at the invocation's start.

        Args:
            now: Current monotonic seconds.
            record: Where the finished invocation's cost goes.
            phase: Baseline or mutant.
            scope: Named tests or whole suite.
            identity: Test count and, for mutants, their exact source site.
        """
        self._now = now
        self._record = record
        self._phase = phase
        self._scope = scope
        self._identity = identity
        self._started = now()

    def __enter__(self) -> None:
        """Enter the invocation; the clock was read when this was made."""

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Record the invocation's cost, whether it returned or raised.

        Args:
            kind: The error's type, or None when it returned.
            error: The error, or None.
            trace: Its traceback, or None.
        """
        self._record(Trial(self._phase, self._scope, self._now() - self._started, self._identity))


class TrialTimings:
    """Accumulate costs without intercepting an invocation's result or error."""

    def __init__(self, now: Callable[[], float]) -> None:
        """Bind the monotonic clock and start an empty run.

        Args:
            now: Current monotonic seconds, from the runner's hooks.
        """
        self._now = now
        self._trials: list[Trial] = []

    def measure(self, phase: Phase, scope: Scope, identity: str) -> Measurement:
        """Measure the enclosed invocation, including one that raises.

        Args:
            phase: Baseline or mutant.
            scope: Named tests or whole suite.
            identity: Test count and, for mutants, their exact source site.

        Returns:
            The context the invocation runs in, its clock read now; its
            failures propagate.
        """
        return Measurement(self._now, self._trials.append, phase, scope, identity)

    def lines(self) -> list[str]:
        """Render four totals and at most five slow invocations.

        Returns:
            Lines with a prefix distinct from the existing verdict grammar.
            Ties retain invocation order; seconds have millisecond precision.
        """
        groups: dict[tuple[Phase, Scope], list[Trial]] = {
            (phase, scope): [] for phase in Phase for scope in Scope
        }
        for trial in self._trials:
            groups[(trial.phase, trial.scope)].append(trial)
        lines: list[str] = []
        for phase in Phase:
            for scope in Scope:
                trials = groups[(phase, scope)]
                seconds = sum(t.seconds for t in trials)
                lines.append(
                    f"mutation timing: {phase} {scope} runs={len(trials)} seconds={seconds:.3f}"
                )
        slowest = sorted(self._trials, key=_descending_seconds)[:5]
        lines.extend(
            f"mutation timing: slow {trial.phase} {trial.scope} seconds={trial.seconds:.3f} "
            f"{trial.identity}"
            for trial in slowest
        )
        return lines


__all__ = ["Measurement", "Phase", "Scope", "Trial", "TrialTimings"]
