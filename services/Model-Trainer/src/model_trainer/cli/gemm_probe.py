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
the same configuration, and the two results could not be read together.
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

from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.gemm_probe import gemm_identity
from model_trainer.core.services.model.gemm_shapes import (
    DIGEST_SUFFIX,
    GEMM_EXPERIMENT,
    SUM_SUFFIX,
    gemm_label,
    probed_shapes,
)

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: Label for the record. Fixed rather than digest-derived like the ladder's,
#: because this table is read as a whole: a record missing shapes is visible
#: as missing observations, which `agree_across_runs` reports as unmatched.
GEMM_LABEL = "gemm-attribution-v1"


def gemm_run_record(device: str) -> RunRecord:
    """Pin determinism, run every declared GEMM, and record the results.

    Determinism is pinned FIRST, before any operand reaches the device,
    because ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is
    created and the first transfer is enough to create it.

    Args:
        device: Device to run every GEMM on.

    Returns:
        The record: two observations per shape and the fingerprint of the
        configuration they ran under.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(device, probe_determinism(device))

    observations: list[Observation] = []
    for name, shape in probed_shapes():
        digest, total = gemm_identity(shape, device)
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
        label=GEMM_LABEL,
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

    record = gemm_run_record(cli_args.require_flag(parsed, DEVICE_FLAG))

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "%d GEMMs %s -> %s",
        len(probed_shapes()),
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


__all__ = ["GEMM_LABEL", "entrypoint", "gemm_run_record", "main"]


# Without this, `python -m model_trainer.cli.gemm_probe` imports the module,
# runs nothing and exits 0 -- measured on the gate probe, where two Slurm jobs
# "succeeded" in six seconds having written no record and no stderr.
if __name__ == "__main__":
    entrypoint()
