"""Run one trial design across a family of scenes.

The determinism findings in this package are all of the same shape: hold the
trial fixed, vary one property of the scene, and read off where the verdict
changes. Written as a loop in a throwaway script that is a measurement nobody
else can repeat; written here it is a function whose inputs are values.

The simulator vendor arrives as a builder rather than a factory, because a sweep
constructs a *different* factory per scene — each one compiles its own MJCF. That
is the injection point which keeps this layer vendor-agnostic: the same sweep
runs against MJX and against the MuJoCo-Warp renderer, and neither name appears
here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from navprobe import NavProbeError
from navprobe.experiment import ProbeService, SimulatorFactoryProtocol
from navprobe.records import SceneSpec, SweepEntry, TrialSpec
from navprobe.scenes import build_scene


class SweepError(NavProbeError):
    """A sweep could not be run.

    Args:
        code: Stable identifier in the ``NP-SWEEP-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


class SimulatorFactoryBuilderProtocol(Protocol):
    """Builds a simulator factory for one compiled scene."""

    def __call__(self, model_xml: str, world_count: int) -> SimulatorFactoryProtocol:
        """Build the factory for a scene.

        Args:
            model_xml: The compiled scene's MJCF document.
            world_count: Parallel worlds each simulator should carry.

        Returns:
            A factory producing freshly constructed simulators for that scene.
        """
        ...


def run_scene_sweep(
    build_factory: SimulatorFactoryBuilderProtocol,
    scenes: Sequence[SceneSpec],
    trial: TrialSpec,
    world_count: int,
) -> tuple[SweepEntry, ...]:
    """Run one trial design against every scene in a family.

    Scenes are run in the order given and each entry carries the scene it came
    from, so a result is readable without consulting the call site that produced
    it.

    Args:
        build_factory: Builds a simulator factory for a compiled scene.
        scenes: The scenes to sweep, in order.
        trial: The trial design applied to every scene, unchanged. Holding it
            fixed is what makes the scene the only variable.
        world_count: Parallel worlds each simulator carries.

    Returns:
        One entry per scene, in the order given.

    Raises:
        SweepError: When no scenes are given, or ``world_count`` is below one.
        SceneError: When a scene specification is not buildable.
        TrialError: When the trial design is unusable.
        RolloutError: When a simulator reports an unusable world count.
        ComparisonError: When two repetitions cannot be compared.
        CanonicalEncodingError: When an observation cannot be encoded.
    """
    if not scenes:
        raise SweepError("NP-SWEEP-001", "a sweep needs at least one scene, got none")
    if world_count < 1:
        raise SweepError("NP-SWEEP-002", f"world_count must be one or greater, got {world_count}")
    return tuple(
        SweepEntry(
            scene=scene,
            trial=ProbeService(build_factory(build_scene(scene), world_count)).run_trial(trial),
        )
        for scene in scenes
    )


def first_irreproducible(entries: Sequence[SweepEntry]) -> SweepEntry | None:
    """Locate the first scene in a sweep that failed to reproduce.

    A sweep's point is usually the boundary rather than the whole table, and
    reading it off by eye is how a threshold gets misreported.

    Args:
        entries: The sweep's entries, in sweep order.

    Returns:
        The first entry whose trial was not deterministic, or ``None`` when
        every scene reproduced.
    """
    for entry in entries:
        if not entry["trial"]["deterministic"]:
            return entry
    return None


__all__ = [
    "SimulatorFactoryBuilderProtocol",
    "SweepError",
    "first_irreproducible",
    "run_scene_sweep",
]
