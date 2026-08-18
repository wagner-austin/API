"""Measure how far apart two different configurations end up.

:mod:`navprobe.dispersion` measures one configuration against itself across
repetitions. This measures one configuration against another: a different
device, a different backend, a different library version — anything that can be
expressed as a second simulator factory.

The distinction matters because the two failures are unrelated. A configuration
can be perfectly self-reproducible and still disagree with its own CPU
counterpart, and both of those have been observed on the same simulator in the
same session.

Both sides are driven in *this* process, which is what makes a magnitude
available at all: a run record stores step digests rather than observations, so
nothing that travels as a run record can answer "by how much".

**Not every pair of configurations can share a process.** MuJoCo-Warp's device
is global process state rather than a property of a model or a simulator, so a
factory built under ``cuda:0`` and driven while the current device is ``cpu``
allocates across a device boundary and segfaults. Two Warp devices therefore
cannot be compared this way at all — that comparison needs one process each, and
:func:`compare_observations` taking values rather than factories is what makes
it available there: each process saves its final observation through
:mod:`navprobe.storage`, and a third loads both and compares.

So the split is: this module for configurations that can coexist — different
scenes, different parameters, different vendors — and saved observations for
those that cannot.
"""

from __future__ import annotations

from collections.abc import Sequence

from navprobe import NavProbeError
from navprobe.dispersion import final_observation
from navprobe.experiment import SimulatorFactoryProtocol
from navprobe.records import DivergenceRecord


class DivergenceError(NavProbeError):
    """Two configurations could not be compared.

    Args:
        code: Stable identifier in the ``NP-DIVERGE-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def compare_observations(left: Sequence[float], right: Sequence[float]) -> DivergenceRecord:
    """Compare two observations element-wise.

    Args:
        left: The first observation.
        right: The second observation.

    Returns:
        The divergence record. When the two agree exactly, the differing count
        is zero and both magnitudes are zero.

    Raises:
        DivergenceError: When the observations are of different lengths, or
            when either is empty. Comparing different shapes element-wise would
            line up one configuration's element three against another's, and
            report the mismatch as a difference.
    """
    if len(left) != len(right):
        raise DivergenceError(
            "NP-DIVERGE-001",
            f"observations differ in length ({len(left)} and {len(right)}); "
            "configurations of different shapes cannot be compared element-wise",
        )
    if not left:
        raise DivergenceError(
            "NP-DIVERGE-002",
            "cannot compare empty observations; a comparison over no elements "
            "would report perfect agreement over no evidence",
        )
    differences = [abs(a - b) for a, b in zip(left, right, strict=True)]
    differing = [value for a, b, value in zip(left, right, differences, strict=True) if a != b]
    return DivergenceRecord(
        observation_length=len(left),
        differing_elements=len(differing),
        max_absolute_difference=max(differences),
        mean_absolute_difference=sum(differing) / len(differing) if differing else 0.0,
    )


def measure_divergence(
    left_factory: SimulatorFactoryProtocol,
    right_factory: SimulatorFactoryProtocol,
    seed: int,
    step_count: int,
) -> DivergenceRecord:
    """Drive two configurations to the same point and compare where they landed.

    One rollout each rather than several, because this asks whether two
    configurations agree with *each other* — not whether either agrees with
    itself, which is what :func:`navprobe.dispersion.measure_dispersion`
    already answers and which must be established separately for this to mean
    anything.

    Args:
        left_factory: Builds the first configuration's simulator.
        right_factory: Builds the second configuration's simulator.
        seed: The seed both sides are pinned to. One seed is the whole design:
            two configurations at different seeds would diverge as intended.
        step_count: Steps each side takes.

    Returns:
        The divergence record.

    Raises:
        DivergenceError: When the two observations differ in length or are
            empty.
        DispersionError: When ``step_count`` is below one.
        CanonicalEncodingError: When either observation contains NaN.
        RolloutError: When a simulator reports an unusable world count.
    """
    return compare_observations(
        final_observation(left_factory(), seed, step_count),
        final_observation(right_factory(), seed, step_count),
    )


__all__ = ["DivergenceError", "compare_observations", "measure_divergence"]
