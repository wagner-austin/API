"""GPU determinism sweep for the ten-scene family under a chosen Warp mode.

Run as ``python -m scripts.gpu_deterministic_sweep <MODE> <CACHE_DIR>
[MAX_RECORDS] [--device DEV]``.

  MODE        NOT_GUARANTEED | RUN_TO_RUN | GPU_TO_GPU
  CACHE_DIR   Warp kernel-cache directory for this run
  MAX_RECORDS optional ``wp.config.deterministic_max_records`` override. The
              default 0 uses Warp's code-generated lower bound, which the
              solver's data-dependent contact loops exceed at 32 bodies
              (RuntimeError: deterministic scatter buffer overflow in '_M').
  --device    Warp device to sweep, default ``cuda:0``. Present so a host
              holding two cards can address each one: the cross-architecture
              question needs the same sweep on a second device with OS, driver,
              CPU and RAM held constant, which is only possible when both cards
              sit in one machine. The resolved device is recorded in the report,
              so a result can never be attributed to the wrong card, and it is
              resolved *before* the sweep so a wrong name fails in a second
              rather than after a cold compile.

Mirrors the wiki-pinned CPU-control design exactly: scenes separated 2/8/16/32
at spacing 0.070 and touching 2/4/5/6/8/32 at 0.055, radius 0.03, timestep
0.005, ``TrialSpec(seed=7, step_count=150, repetitions=12)``, world_count 2,
perturbation 0.01, constraint_capacity 8192.

This module owns the *conditions* and nothing else. The sweep is
:func:`navprobe.sweep.run_scene_sweep`, the record is
:class:`navprobe.records.SweepRunRecord`, and the output is that record's codec
-- so a report can be decoded and compared against another card's run rather
than read. It emitted hand-built JSON until 2026-08-19, which both duplicated a
record layout the package already declares and contradicted
:mod:`navprobe.wireformat`'s stated reason for having no JSON anywhere.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from navprobe.codecs.sweep_run import encode_sweep_run
from navprobe.experiment import SimulatorFactoryProtocol
from navprobe.records import SceneSpec, SweepEntry, SweepRunRecord, TrialSpec
from navprobe.scenes import row_scene
from navprobe.sweep import run_scene_sweep
from scripts import _test_hooks
from scripts.arguments import (
    ScriptArgumentError,
    require_count,
    split_device,
    split_linesearch_block_dim,
)

#: Half-width of the seed-driven initial offset range.
PERTURBATION = 0.01

#: Upper bound on constraints, contacts and Jacobian non-zeros.
CONSTRAINT_CAPACITY = 8192

#: Parallel worlds each simulator carries.
WORLD_COUNT = 2

#: The trial design, held fixed so the scene is the only variable.
TRIAL = TrialSpec(seed=7, step_count=150, repetitions=12)

#: The scene family: separated rows, then the coupled rows that fail under the
#: default mode.
SCENES: tuple[SceneSpec, ...] = tuple(
    [row_scene(n, 0.070, 0.03, 0.005) for n in (2, 8, 16, 32)]
    + [row_scene(n, 0.055, 0.03, 0.005) for n in (2, 4, 5, 6, 8, 32)]
)

#: Positional arguments accepted, with and without the record bound.
POSITIONAL_ARITIES = (2, 3)

USAGE = (
    "usage: gpu_deterministic_sweep <MODE> <CACHE_DIR> [MAX_RECORDS] [--device DEV] "
    "[--linesearch-block-dim N]"
)


class Invocation:
    """One parsed command line.

    Args:
        mode_name: Determinism mode to configure.
        cache_dir: Kernel-cache directory for this run.
        max_records: Deterministic record bound, zero for Warp's own.
        device: Warp device identifier to sweep.
        linesearch_block_dim: Block size to pin the line-search kernel to, or
            ``None`` for the vendor default.
    """

    def __init__(
        self,
        mode_name: str,
        cache_dir: str,
        max_records: int,
        device: str,
        linesearch_block_dim: int | None,
    ) -> None:
        self.mode_name = mode_name
        self.cache_dir = cache_dir
        self.max_records = max_records
        self.device = device
        self.linesearch_block_dim = linesearch_block_dim


def parse_invocation(args: Sequence[str]) -> Invocation:
    """Parse a command line.

    Args:
        args: Arguments excluding the program name.

    Returns:
        The parsed invocation.

    Raises:
        ScriptArgumentError: When the arity is wrong or the record bound is not
            a non-negative integer. Raised rather than reported as an exit code
            so a mistyped sweep stops before it compiles anything, and says
            which argument was wrong.
    """
    device, without_device = split_device(args)
    linesearch_block_dim, positional = split_linesearch_block_dim(without_device)
    if len(positional) not in POSITIONAL_ARITIES:
        raise ScriptArgumentError(
            "NP-ARGS-001",
            f"{USAGE} -- expected {' or '.join(str(n) for n in POSITIONAL_ARITIES)} "
            f"positional arguments, got {len(positional)}",
        )
    max_records = 0
    if len(positional) == POSITIONAL_ARITIES[1]:
        max_records = require_count(positional[2], "MAX_RECORDS")
    return Invocation(positional[0], positional[1], max_records, device, linesearch_block_dim)


def progress_line(scene: SceneSpec, entry: SweepEntry, wall_seconds: float) -> str:
    """Render one scene's result as it completes.

    Streamed so a sweep that runs for an hour is legible while it runs. The
    document written at the end is the record; this is the progress.

    Args:
        scene: The scene just swept.
        entry: Its verdict.
        wall_seconds: Seconds the scene took, including any compilation.

    Returns:
        One newline-terminated line.
    """
    trial = entry["trial"]
    divergent = trial["first_divergent_step"]
    return (
        f"scene bodies={scene['body_count']} spacing={scene['spacing']} "
        f"deterministic={trial['deterministic']} "
        f"first_div={divergent} wall={wall_seconds:.1f}s\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep and write its record.

    Args:
        argv: Arguments excluding the program name. ``None`` reads the process
            arguments.

    Returns:
        ``0``. Every failure raises instead: an unusable command line raises
        :class:`ScriptArgumentError`, an unknown device raises ``ValueError``
        from Warp, and a sweep that cannot run raises from the package.

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
        """Build the adapter's factory for one compiled scene.

        Args:
            model_xml: The compiled scene's MJCF document.
            world_count: Parallel worlds each simulator carries.

        Returns:
            The factory for that scene.
        """
        return construct(
            model_xml,
            world_count,
            PERTURBATION,
            CONSTRAINT_CAPACITY,
            invocation.linesearch_block_dim,
        )

    entries: list[SweepEntry] = []
    with warp.ScopedDevice(invocation.device):
        for scene in SCENES:
            started = _test_hooks.monotonic()
            (entry,) = run_scene_sweep(build_factory, [scene], TRIAL, WORLD_COUNT)
            _test_hooks.write_out(progress_line(scene, entry, _test_hooks.monotonic() - started))
            entries.append(entry)

    _test_hooks.write_out(
        encode_sweep_run(
            SweepRunRecord(
                mode=invocation.mode_name,
                device=device_label,
                device_request=invocation.device,
                max_records=invocation.max_records,
                linesearch_block_dim=invocation.linesearch_block_dim,
                world_count=WORLD_COUNT,
                perturbation=PERTURBATION,
                constraint_capacity=CONSTRAINT_CAPACITY,
                entries=tuple(entries),
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
