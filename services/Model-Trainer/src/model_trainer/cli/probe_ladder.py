"""Run every probe rung on one device and record all of them together.

WHAT QUESTION THIS ANSWERS. The gate probe returns the same loss on sm_70,
sm_80 and sm_86 -- every digit -- while full gpt2 scoring on two of those same
cards produced not one bitwise-identical item score. Both had determinism
pinned. Somewhere between those two workloads, cross-card agreement stops, and
nothing measured where. Those two runs differ in model size AND in input, so
neither axis could be blamed without guessing.

This walks :data:`PROBE_SHAPES`, which varies one axis at a time from the gate
rung, and writes every rung's value into ONE record. Run it on several cards
and the rung where the values stop agreeing is the threshold, on the axis that
rung moved.

WHY ONE RECORD AND NOT ONE PER RUNG. A :class:`RunFingerprint` describes the
image, card, driver and determinism a measurement ran under. Every rung here
runs in one process on one card under one pin, so they genuinely share one
fingerprint, and writing it eight times would invite the eight copies to drift
in a way nothing checks. The rungs are the record's observations, each named
by its own :func:`probe_label`, so a reshaped rung appears as a new
observation rather than as a changed value of the old one.

WHY IT CANNOT BE MISTAKEN FOR THE GATE. It declares its own experiment, and
:func:`~platform_core.known_answer_registry.entry_from_record` and
:func:`~platform_core.known_answer_registry.gate_record` both require exactly
one observation. A ladder record offered to either is refused by count rather
than quietly registering whichever rung happened to sort first.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from collections.abc import Mapping, Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)

from model_trainer.cli import _measurement_hooks

# `probe_determinism` is imported from the gate CLI rather than re-derived.
# The device-conditional pin, the thread count it applies on cpu and the
# measurements justifying both live there; a second pin here would be free to
# drift from the one the registered answers were measured under, and then a
# ladder and a gate run on the same card would not be describing the same
# configuration.
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.known_answer_probe import probe_forward_loss
from model_trainer.core.services.model.probe_shapes import ProbeShape, probe_label

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: Distinct from the gate's experiment. Two records may only be differenced
#: when they answer the same question, and "does this environment still
#: compute" is not "where does agreement break".
LADDER_EXPERIMENT = "probe-shape-threshold"

#: How much of the digest goes in the label. Twelve hex characters is 48 bits;
#: the collision it guards against is two hand-edited rung tables, not an
#: adversary.
LADDER_DIGEST_CHARS = 12


def ladder_rung_labels(shapes: Mapping[str, ProbeShape]) -> tuple[str, ...]:
    """Name every rung's measurement, in table order.

    Args:
        shapes: The ladder to name.

    Returns:
        One label per rung.
    """
    return tuple(probe_label(shape) for shape in shapes.values())


def ladder_label(rung_labels: tuple[str, ...]) -> str:
    """Build the label identifying a ladder by the rungs it contains.

    Derived rather than a version constant someone must remember to bump. A
    rung added, removed or reshaped produces a different label, so two records
    that walked different ladders can never be mistaken for two runs of one
    ladder -- which matters because the interesting reading is "these agree",
    and a missing rung would make a shorter ladder agree trivially.

    The digest is not the authority on what ran: the observations carry the
    full rung labels. It exists so the difference is visible in a filename or
    a log line, where eight labels would not fit.

    Args:
        rung_labels: The rung labels, in the order they were run.

    Returns:
        The label, e.g. ``probe-ladder-8x1a2b3c4d5e6f``.
    """
    digest = hashlib.sha256("\n".join(rung_labels).encode("utf-8")).hexdigest()
    return f"probe-ladder-{len(rung_labels)}x{digest[:LADDER_DIGEST_CHARS]}"


def ladder_run_record(device: str, shapes: Mapping[str, ProbeShape]) -> RunRecord:
    """Pin determinism, run every rung, and record what they ran on.

    Determinism is pinned FIRST, before any rung builds a model, because
    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created and
    constructing a model on cuda is enough to create it. Pinning once for the
    whole ladder is also what makes the rungs comparable to each other: a
    per-rung pin would be a no-op after the first and would misreport the
    later rungs as having pinned something.

    Args:
        device: Device to run every rung on.
        shapes: The ladder to walk, in the order to walk it. Taken as an
            argument rather than read from the module so the suite can
            exercise this on two cheap rungs: the real ladder ends in a
            1.5-billion-parameter model, which is a GPU measurement and not
            something to build on a test runner's CPU.

    Returns:
        The record: one observation per rung, named by that rung's label, and
        the fingerprint of the configuration all of them ran under. The
        payload digest is :const:`NO_PAYLOAD` -- the values ARE the output, so
        a digest over them would restate the numbers rather than add the
        independent check a digest is for.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )

    observations: list[Observation] = []
    for rung, shape in shapes.items():
        label = probe_label(shape)
        loss = probe_forward_loss(device, shape)
        # Logged per rung rather than only at the end: the large rungs take
        # long enough that a job killed by a wall clock or preemption would
        # otherwise leave no record of which rungs had already succeeded.
        _log.info("rung %s (%s) = %.17g", rung, label, loss)
        observations.append(Observation(name=label, value=loss))

    return run_record(
        experiment=LADDER_EXPERIMENT,
        label=ladder_label(ladder_rung_labels(shapes)),
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ladder once and write its record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent. Nothing is computed on a command line
            that was not understood, because a ladder run on a device other
            than the one named would write a record claiming to be this one.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    record = ladder_run_record(
        cli_args.require_flag(parsed, DEVICE_FLAG), _measurement_hooks.ladder_shapes()
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "ladder %s over %d rungs %s -> %s",
        record["label"],
        len(record["observations"]),
        describe_run_fingerprint(record["fingerprint"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Logging is configured here rather than left to whatever a caller did,
    because the per-rung lines are this command's only partial output: without
    a handler at INFO they are dropped, and a ladder killed by preemption
    part-way through would leave nothing at all behind.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="probe-ladder",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "LADDER_DIGEST_CHARS",
    "LADDER_EXPERIMENT",
    "entrypoint",
    "ladder_label",
    "ladder_run_record",
    "ladder_rung_labels",
    "main",
]


# Without this, `python -m model_trainer.cli.probe_ladder` IMPORTS the module,
# runs nothing, and exits 0. Measured on the gate probe: HPC3 jobs 55595084 and
# 55595086 each "succeeded" in six seconds having written no record and no
# stderr, and only the absent output file said so.
if __name__ == "__main__":
    entrypoint()
