"""The determinism trial: the layer that composes the instrument.

Everything below this module is a part. :mod:`navprobe.rollout` produces one run
record, :mod:`navprobe.comparison` compares two. Neither states the experiment,
and an experiment left unstated is an experiment performed differently every
time it is run.

A trial is that statement: pin one seed, build ``repetitions`` independent
simulators from a factory, roll each out, and compare every repetition against
the first. One seed throughout is the whole design — repetitions under different
seeds would diverge as intended and prove nothing.

The simulator arrives as a factory rather than an instance because each
repetition must start from a simulator that was freshly constructed. Reusing one
instance would measure whether ``reset`` restores state, which is a weaker
question wearing the same name.

:class:`ProbeService` is constructed with its collaborator and holds no other
state, so a trial against MJX and a trial against an in-repo simulator differ
only by the factory passed in.
"""

from __future__ import annotations

from typing import Protocol

from navprobe import NavProbeError
from navprobe.comparison import compare_runs
from navprobe.records import ComparisonRecord, RunRecord, TrialRecord, TrialSpec
from navprobe.rollout import SimulatorProtocol, roll_out

#: Fewest repetitions that can establish anything. One rollout has nothing to
#: disagree with, so a trial of one would report perfect determinism over no
#: evidence.
MINIMUM_REPETITIONS = 2


class TrialError(NavProbeError):
    """A determinism trial could not be run or summarised.

    Args:
        code: Stable identifier in the ``NP-TRIAL-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


class SimulatorFactoryProtocol(Protocol):
    """Builds a freshly constructed simulator on each call."""

    def __call__(self) -> SimulatorProtocol:
        """Construct a simulator.

        Returns:
            A simulator in its initial state, not yet reset to any seed.
        """
        ...


def repetition_label(index: int) -> str:
    """Name a repetition for the record it produces.

    Labels are derived rather than supplied so that two trials of the same
    design produce records that compare by content. A caller-chosen label would
    make byte comparison of two trials fail on the name alone.

    Args:
        index: Zero-based repetition number.

    Returns:
        The label carried by that repetition's run record.
    """
    return f"repetition-{index}"


class ProbeService:
    """Runs determinism trials against simulators from an injected factory.

    Args:
        simulator_factory: Builds a freshly constructed simulator per
            repetition.
    """

    def __init__(self, simulator_factory: SimulatorFactoryProtocol) -> None:
        self._simulator_factory = simulator_factory

    def roll_out_repetitions(self, spec: TrialSpec) -> tuple[RunRecord, ...]:
        """Drive one freshly built simulator per repetition.

        Args:
            spec: The trial design.

        Returns:
            One run record per repetition, in repetition order.

        Raises:
            TrialError: When ``repetitions`` is below
                :data:`MINIMUM_REPETITIONS`.
            RolloutError: When a simulator reports an unusable world count, or
                the step count is negative.
            CanonicalEncodingError: When an observation cannot be canonically
                encoded, which includes any observation containing NaN.
        """
        repetitions = spec["repetitions"]
        if repetitions < MINIMUM_REPETITIONS:
            raise TrialError(
                "NP-TRIAL-001",
                f"a trial needs at least {MINIMUM_REPETITIONS} repetitions to compare, "
                f"got {repetitions}; a single rollout has nothing to disagree with",
            )
        return tuple(
            roll_out(
                self._simulator_factory(),
                repetition_label(index),
                spec["seed"],
                spec["step_count"],
            )
            for index in range(repetitions)
        )

    def compare_against_reference(
        self, runs: tuple[RunRecord, ...]
    ) -> tuple[ComparisonRecord, ...]:
        """Compare every repetition after the first against the first.

        Comparing each against a single reference rather than pairwise is what
        makes ``first_divergent_step`` mean "where this repetition left the
        reference". Pairwise comparison would produce a divergence point per
        pair with no shared origin to interpret it against.

        Args:
            runs: The repetitions, in repetition order.

        Returns:
            One comparison per repetition after the first.

        Raises:
            TrialError: When fewer than :data:`MINIMUM_REPETITIONS` runs are
                given.
            ComparisonError: When two runs were produced under different seeds,
                or a run's digest contradicts its own steps.
        """
        if len(runs) < MINIMUM_REPETITIONS:
            raise TrialError(
                "NP-TRIAL-002",
                f"comparison needs at least {MINIMUM_REPETITIONS} runs, got {len(runs)}",
            )
        reference = runs[0]
        return tuple(compare_runs(reference, run) for run in runs[1:])

    def summarise(self, spec: TrialSpec, runs: tuple[RunRecord, ...]) -> TrialRecord:
        """Fold repetitions into the trial's verdict.

        Args:
            spec: The trial design the runs were produced under.
            runs: The repetitions, in repetition order.

        Returns:
            The trial record: whether every repetition matched the reference,
            and the earliest step at which any of them did not.

        Raises:
            TrialError: When fewer than :data:`MINIMUM_REPETITIONS` runs are
                given, or when a run was not produced under ``spec``.
            ComparisonError: When two runs were produced under different seeds,
                or a run's digest contradicts its own steps.
        """
        comparisons = self.compare_against_reference(runs)
        self._require_runs_match_spec(spec, runs)
        divergences = [
            comparison["first_divergent_step"]
            for comparison in comparisons
            if comparison["first_divergent_step"] is not None
        ]
        reference = runs[0]
        return TrialRecord(
            spec=spec,
            world_count=reference["spec"]["world_count"],
            reference_digest=reference["digest"],
            deterministic=all(comparison["digests_match"] for comparison in comparisons),
            first_divergent_step=min(divergences) if divergences else None,
        )

    def run_trial(self, spec: TrialSpec) -> TrialRecord:
        """Run a complete trial and return its verdict.

        Args:
            spec: The trial design.

        Returns:
            The trial record.

        Raises:
            TrialError: When ``repetitions`` is below
                :data:`MINIMUM_REPETITIONS`.
            RolloutError: When a simulator reports an unusable world count, or
                the step count is negative.
            ComparisonError: When two runs were produced under different seeds,
                or a run's digest contradicts its own steps.
            CanonicalEncodingError: When an observation cannot be canonically
                encoded, which includes any observation containing NaN.
        """
        return self.summarise(spec, self.roll_out_repetitions(spec))

    @staticmethod
    def _require_runs_match_spec(spec: TrialSpec, runs: tuple[RunRecord, ...]) -> None:
        """Refuse runs that were not produced under this trial's design.

        A summary that describes runs it did not come from is worse than no
        summary: it reports a determinism verdict for a seed or a step count
        that was never exercised.

        Args:
            spec: The trial design.
            runs: The repetitions to check.

        Raises:
            TrialError: When any run's seed or step count differs from
                ``spec``.
        """
        for run in runs:
            run_spec = run["spec"]
            if run_spec["seed"] != spec["seed"] or run_spec["step_count"] != spec["step_count"]:
                raise TrialError(
                    "NP-TRIAL-003",
                    f"run {run_spec['label']!r} was produced at seed {run_spec['seed']} for "
                    f"{run_spec['step_count']} steps, but the trial declares seed "
                    f"{spec['seed']} for {spec['step_count']} steps",
                )


__all__ = [
    "MINIMUM_REPETITIONS",
    "ProbeService",
    "SimulatorFactoryProtocol",
    "TrialError",
    "repetition_label",
]
