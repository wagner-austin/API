"""The isolated-GEMM digest probe under the ordered kernel.

The record-compatible twin of Model-Trainer's ``gemm_probe``: the same shape
tables, the same operands, the same digest fold, the same record schema, so
``agree_across_runs`` and every analysis script read it unchanged. The label
arm is ``ordered`` and the controls are HARDWIRED to ``both`` -- this
package exists inside one experiment's posture, and a flag whose only valid
value is a constant would be a place to hold the record's name wrong.

The load-bearing expectation, asserted by the suite per shape and by the
cluster per card: an ``ordered`` record equals a ``rank1`` record digest for
digest. Same order, same roundings; only the speed differs.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from model_trainer.cli import _test_hooks
from model_trainer.cli.gemm_probe import gemm_label_for
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.gemm_probe import gemm_operands
from model_trainer.core.services.model.gemm_shapes import (
    DIGEST_SUFFIX,
    GEMM_EXPERIMENT,
    SUM_SUFFIX,
    GemmShape,
    gemm_label,
)
from model_trainer.core.services.model.tensor_digest import (
    describe_tensor,
    require_reproduced,
)
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

from ordered_kernels.kernels import gemm

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: The arm this command writes into every label.
ORDERED_ARM = "ordered"


def ordered_identity(shape: GemmShape, device: str) -> tuple[float, float]:
    """Run one GEMM twice under the ordered kernel and describe the result.

    Args:
        shape: The call to measure.
        device: Device to run it on.

    Returns:
        ``(folded digest, float64 sum)`` of the output.

    Raises:
        RuntimeError: Propagated from
            :func:`~tensor_digest.require_reproduced` when the same call on
            the same device produced two different tensors.
        ValueError: Propagated from :func:`~ordered_kernels.kernels.gemm`.
    """
    bias, x, w = gemm_operands(shape, device)
    first = gemm(x, w, bias).cpu()
    second = gemm(x, w, bias).cpu()
    what = f"an ordered GEMM M{shape['rows']}xK{shape['inner']}xN{shape['cols']}"
    return describe_tensor(require_reproduced(first, second, what, device))


def ordered_run_record(device: str) -> RunRecord:
    """Pin the experiment's posture, run every declared GEMM, record it.

    Args:
        device: Device to run every GEMM on.

    Returns:
        The record, labelled ``gemm-attribution-v2-both-ordered``.

    Raises:
        RuntimeError: Propagated from :func:`ordered_identity`.
        ValueError: Propagated from :func:`~ordered_kernels.kernels.gemm`.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=True, math_attention=True),
    )
    observations: list[Observation] = []
    for name, shape in _test_hooks.probed_shapes_hook():
        digest, total = ordered_identity(shape, device)
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
        label=gemm_label_for("both", ORDERED_ARM),
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run every GEMM once and write the record.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            absent -- resolved before anything computes.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = ordered_run_record(device)

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
        service_name="ordered-gemm-probe",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["ORDERED_ARM", "entrypoint", "main", "ordered_identity", "ordered_run_record"]


# Without this, `python -m ordered_kernels.cli.gemm_probe` imports the
# module, runs nothing and exits 0 -- the gate-probe lesson.
if __name__ == "__main__":
    entrypoint()
