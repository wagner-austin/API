"""Sweep the collision-pair family, recording a liveness witness beside each verdict.

Run as ``python -m scripts.collision_pair_probe <MODE> <CACHE_DIR>
[MAX_RECORDS] [--device DEV] [--linesearch-block-dim N]``.

  MODE        NOT_GUARANTEED | RUN_TO_RUN | GPU_TO_GPU
  CACHE_DIR   Warp kernel-cache directory for this run
  MAX_RECORDS optional ``wp.config.deterministic_max_records`` override.
  --device    Warp device to sweep, default ``cuda:0``.

This is the sweep the convex-narrowphase finding is stated from. It differs
from :mod:`scripts.gpu_deterministic_sweep` in what it varies: that one holds
the geometry fixed and grows the scene, this one holds the size fixed and
changes *what collides with what*, because MuJoCo-Warp's ``MJ_COLLISION_TABLE``
dispatches on the geom-type pair and sends it to one of two narrowphases.

**Every row carries contact counts as well as a verdict, and that is the
point.** A determinism verdict compares repetitions against each other and
never against the physics, so a mode that silently stops generating contacts
produces identical rollouts and scores ``deterministic: true``. Measured
2026-08-30: every pair routing to the convex narrowphase does exactly that
under ``RUN_TO_RUN`` while reproducing bit for bit. A sweep that reported only
the verdict would call that a clean result.

The witness comes from a rollout of its own, taken after the trial from a
freshly constructed simulator under the same seed and step count. It is a
separate rollout because :func:`navprobe.rollout.roll_out` returns an
observation digest and nothing else -- threading a vendor-specific counter
through it would put contacts into a layer that is vendor-agnostic on purpose.
Under a mode that reproduces, the witness rollout is the same trajectory the
trial measured; under one that does not, the trial has already said so.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from navprobe.codecs.contact_witness import encode_contact_witness_run
from navprobe.collision_pairs import COLLISION_PAIRS, build_pair
from navprobe.experiment import ProbeService
from navprobe.records import ContactWitnessEntry, ContactWitnessRunRecord, TrialSpec
from scripts import _test_hooks
from scripts.arguments import (
    DEVICE_FLAG,
    ScriptArgumentError,
    require_count,
    split_device,
    split_linesearch_block_dim,
)

#: Half-width of the seed-driven initial offset range.
PERTURBATION = 0.01

#: Upper bound on constraints, contacts and Jacobian non-zeros.
CONSTRAINT_CAPACITY = 4096

#: Parallel worlds each simulator carries.
WORLD_COUNT = 2

#: The trial every pair is measured under.
TRIAL = TrialSpec(seed=7, step_count=40, repetitions=4)

#: Accepted positional arities: with and without the record bound.
POSITIONAL_ARITIES = (2, 3)

USAGE = (
    "usage: collision_pair_probe <MODE> <CACHE_DIR> [MAX_RECORDS] [--device DEV] "
    f"[--linesearch-block-dim N]  ({DEVICE_FLAG} defaults to cuda:0)"
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
            a non-negative integer.
    """
    device, without_device = split_device(args)
    linesearch_block_dim, positional = split_linesearch_block_dim(without_device)
    if len(positional) not in POSITIONAL_ARITIES:
        raise ScriptArgumentError(
            "NP-ARGS-008",
            f"{USAGE} -- expected {' or '.join(str(n) for n in POSITIONAL_ARITIES)} "
            f"positional arguments, got {len(positional)}",
        )
    max_records = 0
    if len(positional) == POSITIONAL_ARITIES[1]:
        max_records = require_count(positional[2], "MAX_RECORDS")
    return Invocation(positional[0], positional[1], max_records, device, linesearch_block_dim)


def progress_line(entry: ContactWitnessEntry, wall_seconds: float) -> str:
    """Render one pair's result as it completes.

    The contact total sits beside the verdict here for the same reason it sits
    beside it in the record: ``deterministic=True contacts=0`` is the shape of
    the failure, and a progress line showing only the verdict would read as a
    pass while it scrolled past.

    Args:
        entry: The pair's verdict and witness.
        wall_seconds: Seconds the pair took, including any compilation.

    Returns:
        One newline-terminated line.
    """
    trial = entry["trial"]
    return (
        f"pair={entry['pair']} deterministic={trial['deterministic']} "
        f"contacts={entry['contact_total']} zero_steps={entry['zero_contact_steps']} "
        f"first_div={trial['first_divergent_step']} wall={wall_seconds:.1f}s\n"
    )


def measure_witness(
    factory: _test_hooks.WitnessFactoryProtocol, spec: TrialSpec
) -> tuple[int, int]:
    """Roll one repetition out, counting contacts at every step.

    Args:
        factory: Builds a freshly constructed simulator.
        spec: The trial design, whose seed and step count this rollout matches.

    Returns:
        The contact total and the number of steps that reported no contact.
    """
    simulator = factory()
    simulator.reset(spec["seed"])
    total = 0
    zero_steps = 0
    for _ in range(spec["step_count"]):
        simulator.advance()
        contacts = simulator.contact_count()
        total += contacts
        zero_steps += 1 if contacts == 0 else 0
    return total, zero_steps


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pair sweep and write its record.

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
    construct = _test_hooks.load_witness_factory()

    entries: list[ContactWitnessEntry] = []
    with warp.ScopedDevice(invocation.device):
        for pair in COLLISION_PAIRS:
            started = _test_hooks.monotonic()
            factory = construct(
                build_pair(pair),
                WORLD_COUNT,
                PERTURBATION,
                CONSTRAINT_CAPACITY,
                invocation.linesearch_block_dim,
            )
            trial = ProbeService(factory).run_trial(TRIAL)
            total, zero_steps = measure_witness(factory, TRIAL)
            entry = ContactWitnessEntry(
                pair=pair,
                trial=trial,
                contact_total=total,
                zero_contact_steps=zero_steps,
            )
            _test_hooks.write_out(progress_line(entry, _test_hooks.monotonic() - started))
            entries.append(entry)

    _test_hooks.write_out(
        encode_contact_witness_run(
            ContactWitnessRunRecord(
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
