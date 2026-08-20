"""World-count scaling sweep: determinism and throughput vs nworld, one scene.

Run as ``python -m scripts.world_scaling_sweep [--device DEV] <MODE>
<CACHE_DIR> <MAX_RECORDS> <CAPACITY> <WORLDS...>``.

  --device    Warp device to ladder on, default ``cuda:0``. A flag rather than
              a positional because WORLDS is variadic and would swallow it.
              The resolved device is recorded so a throughput figure is never
              separated from the card that produced it -- the figures are not
              comparable across devices, only the curve's shape is.
  MODE        NOT_GUARANTEED | RUN_TO_RUN | GPU_TO_GPU
  CACHE_DIR   Warp kernel-cache directory
  MAX_RECORDS ``wp.config.deterministic_max_records`` (0 = static bound)
  CAPACITY    per-world constraint capacity (njmax/nconmax). The canonical
              sweeps use 8192 at nworld 2; the default sparse-Jacobian
              allocation scales as njmax * nv * nworld, so scaling runs need a
              right-sized value (touching-8 peaks near 20 contacts/world). A
              smaller card needs this smaller again: the 4096-world rung fitted
              on 24 GiB at 256, and VRAM is the binding constraint.
  WORLDS...   world counts to ladder through, e.g. 2 64 512 4096

Scene is fixed: touching row of 8 spheres (spacing 0.055, radius 0.03,
timestep 0.005) -- solidly inside the coupled-body failure regime. Trial design
matches the canonical sweeps (seed 7, 150 steps, 12 repetitions). Digests are
not comparable across world counts (each world is seed-perturbed, so nworld
changes the state being digested); the per-count outputs are the determinism
verdict and the wall clock, from which world-steps/second is derived.

A rung that exhausts constraint capacity or VRAM raises, and the ladder stops
there. The rungs already completed have been streamed, so the last world count
that fitted is on the operator's screen along with the vendor's own error --
which is a better account of the ceiling than a row reading "failed" would be.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from navprobe.codecs.scaling_run import encode_scaling_run
from navprobe.experiment import SimulatorFactoryProtocol
from navprobe.records import ScalingRungRecord, ScalingRunRecord, SceneSpec, TrialSpec
from navprobe.scenes import row_scene
from navprobe.sweep import run_scene_sweep
from scripts import _test_hooks
from scripts.arguments import (
    ScriptArgumentError,
    require_count,
    require_positive_count,
    split_device,
)

#: Half-width of the seed-driven initial offset range.
PERTURBATION = 0.01

#: The trial design, held fixed so world count is the only variable.
TRIAL = TrialSpec(seed=7, step_count=150, repetitions=12)

#: The one scene, solidly inside the coupled-body failure regime.
SCENE: SceneSpec = row_scene(8, 0.055, 0.03, 0.005)

#: Positional arguments before the variadic world counts.
FIXED_POSITIONAL_COUNT = 4

USAGE = (
    "usage: world_scaling_sweep [--device DEV] <MODE> <CACHE_DIR> "
    "<MAX_RECORDS> <CAPACITY> <WORLDS...>"
)


class Invocation:
    """One parsed command line.

    Args:
        mode_name: Determinism mode to configure.
        cache_dir: Kernel-cache directory for this run.
        max_records: Deterministic record bound, zero for Warp's own.
        capacity: Per-world constraint capacity.
        world_counts: World counts to ladder through, in order.
        device: Warp device identifier to ladder on.
    """

    def __init__(
        self,
        mode_name: str,
        cache_dir: str,
        max_records: int,
        capacity: int,
        world_counts: tuple[int, ...],
        device: str,
    ) -> None:
        self.mode_name = mode_name
        self.cache_dir = cache_dir
        self.max_records = max_records
        self.capacity = capacity
        self.world_counts = world_counts
        self.device = device


def parse_invocation(args: Sequence[str]) -> Invocation:
    """Parse a command line.

    Args:
        args: Arguments excluding the program name.

    Returns:
        The parsed invocation.

    Raises:
        ScriptArgumentError: When the arity is short, no world count is given,
            or a numeric argument is not a number in range.
    """
    device, positional = split_device(args)
    if len(positional) <= FIXED_POSITIONAL_COUNT:
        raise ScriptArgumentError(
            "NP-ARGS-005",
            f"{USAGE} -- expected more than {FIXED_POSITIONAL_COUNT} positional "
            f"arguments, got {len(positional)}",
        )
    return Invocation(
        positional[0],
        positional[1],
        require_count(positional[2], "MAX_RECORDS"),
        require_positive_count(positional[3], "CAPACITY"),
        tuple(
            require_positive_count(token, f"WORLDS[{offset}]")
            for offset, token in enumerate(positional[FIXED_POSITIONAL_COUNT:])
        ),
        device,
    )


def rung_from(
    world_count: int, entry_digest: str, deterministic: bool, divergent: int | None, wall: float
) -> ScalingRungRecord:
    """Build one rung from a completed sweep.

    Args:
        world_count: Parallel worlds the rung ran.
        entry_digest: The rung's reference digest.
        deterministic: Whether every repetition agreed.
        divergent: First step at which two repetitions parted, if any.
        wall: Seconds the rung took.

    Returns:
        The rung, with throughput derived from the trial design.
    """
    world_steps = world_count * TRIAL["step_count"] * TRIAL["repetitions"]
    return ScalingRungRecord(
        world_count=world_count,
        reference_digest=entry_digest,
        deterministic=deterministic,
        first_divergent_step=divergent,
        wall_seconds=wall,
        world_steps_per_second=world_steps / wall,
    )


def progress_line(rung: ScalingRungRecord) -> str:
    """Render one rung's result as it completes.

    Args:
        rung: The rung just completed.

    Returns:
        One newline-terminated line.
    """
    return (
        f"nworld={rung['world_count']} deterministic={rung['deterministic']} "
        f"wall={rung['wall_seconds']:.1f}s "
        f"world_steps/s={rung['world_steps_per_second']:.0f}\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ladder and write its record.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        ``0``. Every failure raises instead.

    Raises:
        ScriptArgumentError: When the command line is unusable.
        ValueError: When Warp does not recognise the requested device.
    """
    invocation = parse_invocation(list(sys.argv[1:]) if argv is None else argv)
    _test_hooks.opt_out_of_power_throttling()
    warp = _test_hooks.init_warp(invocation.mode_name, invocation.cache_dir, invocation.max_records)
    device_label = str(warp.get_device(invocation.device))
    construct = _test_hooks.load_state_factory()

    def build_factory(model_xml: str, world_count: int) -> SimulatorFactoryProtocol:
        """Build the adapter's factory for the scene.

        Defined once rather than per rung: the capacity is fixed for the whole
        ladder, and only ``world_count`` varies, which the sweep passes in.

        Args:
            model_xml: The compiled scene's MJCF document.
            world_count: Parallel worlds each simulator carries.

        Returns:
            The factory for that scene.
        """
        return construct(model_xml, world_count, PERTURBATION, invocation.capacity)

    rungs: list[ScalingRungRecord] = []
    with warp.ScopedDevice(invocation.device):
        for world_count in invocation.world_counts:
            started = _test_hooks.monotonic()
            (entry,) = run_scene_sweep(build_factory, [SCENE], TRIAL, world_count)
            trial = entry["trial"]
            rung = rung_from(
                world_count,
                trial["reference_digest"],
                trial["deterministic"],
                trial["first_divergent_step"],
                _test_hooks.monotonic() - started,
            )
            _test_hooks.write_out(progress_line(rung))
            rungs.append(rung)

    _test_hooks.write_out(
        encode_scaling_run(
            ScalingRunRecord(
                mode=invocation.mode_name,
                device=device_label,
                device_request=invocation.device,
                max_records=invocation.max_records,
                capacity=invocation.capacity,
                scene=SCENE,
                spec=TRIAL,
                perturbation=PERTURBATION,
                rungs=tuple(rungs),
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
