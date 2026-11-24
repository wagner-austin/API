"""Bounded cost reports for mutation baselines and trials, independent of verdicts.

The shared runner and API's lifted core use only the standard library.
Timings include process startup and preparation, since both cost the caller
time. They never select tests, change deadlines, or decide a mutant's result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
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


class TrialTimings:
    """Accumulate costs without intercepting an invocation's result or error."""

    def __init__(self, now: Callable[[], float]) -> None:
        """Bind the monotonic clock and start an empty run.

        Args:
            now: Current monotonic seconds, from the runner's hooks.
        """
        self._now = now
        self._trials: list[Trial] = []

    @contextmanager
    def measure(self, phase: Phase, scope: Scope, identity: str) -> Iterator[None]:
        """Measure the enclosed invocation, including one that raises.

        Args:
            phase: Baseline or mutant.
            scope: Named tests or whole suite.
            identity: Test count and, for mutants, their exact source site.

        Yields:
            Control to the original invocation, whose failures propagate.
        """
        started = self._now()
        try:
            yield
        finally:
            self._trials.append(Trial(phase, scope, self._now() - started, identity))

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


__all__ = ["Phase", "Scope", "Trial", "TrialTimings"]
