"""Run every isolated GEMM on one device and record what each produced.

The attribution step. The ladder says which RUNGS disagree across cards; the
cuBLASLt trace says which SHAPES draw different kernels; neither says which
matmul carries a rung's difference, because a forward pass reduces everything
to one loss. This runs the GEMMs one at a time and records each output's
identity, so a difference can be attributed to a call rather than to a rung.

Two observations per shape -- a folded digest of the output bytes and its
float64 sum. The digest answers "are these the same tensor" without the
cancellation blind spot a sum has; the sum says how large a difference is
once the digest reports one.

WHY IT SHARES THE PROBE'S DETERMINISM PIN. It imports ``probe_determinism``
from the gate CLI for the reason the ladder does: a GEMM measured under a
different posture than the ladder that motivated it would not be describing
the same configuration, and the two results could not be read together. Which
posture is a required flag rather than a constant, because as of 2026-08-29
the measurement it has to line up with is the four-card forward trace, which
runs under ``--controls both``.

WHAT ``--kernel`` IS FOR. Everything above answers "which vendor kernel", and
the four-card trace has now exhausted that question: with every control on,
three architectures agree on every tensor of a 355M-parameter model and the
V100 still breaks at layer zero's QKV projection. What is left is not a
setting, it is who chooses the summation order, so
:mod:`~model_trainer.core.services.model.deterministic_gemm` supplies three
arms -- ``cublas``, ``fp64``, ``rank1`` -- and this command measures the same
table under each. See that module for what each arm claims and why only one
of them claims anything.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

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

from model_trainer.cli import _test_hooks
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.control_arms import CONTROLS_FLAG, require_control_arm
from model_trainer.core.services.model.deterministic_gemm import require_kernel_arm
from model_trainer.core.services.model.gemm_probe import gemm_identity
from model_trainer.core.services.model.gemm_shapes import (
    DIGEST_SUFFIX,
    GEMM_EXPERIMENT,
    SUM_SUFFIX,
    gemm_label,
)

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"
KERNEL_FLAG = "--kernel"

_FLAGS = (DEVICE_FLAG, OUT_FLAG, CONTROLS_FLAG, KERNEL_FLAG)

#: Label for the record. Fixed rather than digest-derived like the ladder's,
#: because this table is read as a whole: a record missing shapes is visible
#: as missing observations, which `agree_across_runs` reports as unmatched.
#:
#: The kernel arm and the control arm are IN the label rather than recorded
#: beside it, because both change what was computed. ``agree_across_runs``
#: compares records sharing a label, and a rank-one sum compared against a
#: cuBLAS one would report a disagreement that is the experiment working
#: rather than a card misbehaving. Two cards under one arm share a label and
#: are compared; two arms do not and are not.
GEMM_LABEL_PREFIX = "gemm-attribution-v2"


def gemm_label_for(controls: str, kernel: str) -> str:
    """Name the record one arm pair produces.

    Args:
        controls: The ``--controls`` value, unresolved, as the operator typed
            it -- the label says which arm was ASKED for.
        kernel: The ``--kernel`` value.

    Returns:
        e.g. ``gemm-attribution-v2-both-rank1``.
    """
    return f"{GEMM_LABEL_PREFIX}-{controls}-{kernel}"


def gemm_run_record(device: str, *, controls: str, kernel: str) -> RunRecord:
    """Pin determinism, run every declared GEMM, and record the results.

    Determinism is pinned FIRST, before any operand reaches the device,
    because ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is
    created and the first transfer is enough to create it.

    Args:
        device: Device to run every GEMM on.
        controls: Which cross-card controls to apply, by
            :data:`~control_arms.CONTROL_ARMS` name. Required rather than
            fixed at ``none`` as this command was until 2026-08-29: the trace
            that motivated the QKV shapes ran under ``both``, and a baseline
            measured under a different posture than the trace it explains
            cannot be read against it.
        kernel: Which arithmetic, by
            :data:`~deterministic_gemm.KERNEL_ARMS` name.

    Returns:
        The record: two observations per shape and the fingerprint of the
        configuration they ran under.

    Raises:
        ValueError: Propagated from
            :func:`~control_arms.require_control_arm` or
            :func:`~deterministic_gemm.require_kernel_arm` for an unknown arm.
    """
    remove_split_k, math_attention = require_control_arm(controls)
    named_kernel = require_kernel_arm(kernel)

    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=remove_split_k, math_attention=math_attention),
    )

    observations: list[Observation] = []
    # Through the hook rather than the table directly, so the suite can walk
    # every line of this loop without digesting ninety-three shapes on a CPU
    # -- the reason `benchmark_shapes` is a hook. Production's default IS the
    # full table.
    for name, shape in _test_hooks.probed_shapes_hook():
        digest, total = gemm_identity(shape, device, kernel=named_kernel)
        _log.info(
            "gemm %s M%d K%d N%d digest=%.0f sum=%.17g",
            name,
            shape["rows"],
            shape["inner"],
            shape["cols"],
            digest,
            total,
        )
        observations.append(Observation(name=gemm_label(name, shape, DIGEST_SUFFIX), value=digest))
        observations.append(Observation(name=gemm_label(name, shape, SUM_SUFFIX), value=total))

    return run_record(
        experiment=GEMM_EXPERIMENT,
        label=gemm_label_for(controls, named_kernel),
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run every GEMM once and write the record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    # Every flag resolved BEFORE anything computes. Until 2026-08-29 the
    # destination was read last, so a command line missing `--out` ran all
    # fifty-nine GEMMs and then discovered it had nowhere to put them.
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    controls = cli_args.require_flag(parsed, CONTROLS_FLAG)
    kernel = cli_args.require_flag(parsed, KERNEL_FLAG)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = gemm_run_record(device, controls=controls, kernel=kernel)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "%d GEMMs %s %s -> %s",
        len(_test_hooks.probed_shapes_hook()),
        record["label"],
        describe_run_fingerprint(record["fingerprint"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="gemm-probe",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "GEMM_LABEL_PREFIX",
    "entrypoint",
    "gemm_label_for",
    "gemm_run_record",
    "main",
]


# Without this, `python -m model_trainer.cli.gemm_probe` imports the module,
# runs nothing and exits 0 -- measured on the gate probe, where two Slurm jobs
# "succeeded" in six seconds having written no record and no stderr.
if __name__ == "__main__":
    entrypoint()
