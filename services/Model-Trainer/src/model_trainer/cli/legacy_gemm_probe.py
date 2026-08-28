"""Record both library entry points for the one matmul the switch cannot reach.

Runs each output-projection shape twice on one card -- ``mm(x, w)`` on the
legacy ``cublasSgemm`` path that `lm_head` takes today, and
``addmm(zeros, x, w)`` on the cuBLASLt path a fused epilogue routes it to --
and records a digest of each. Differencing these records across cards answers
the question `a-loss-agrees-where-the-computation-does-not` left open: whether
the residual `lm_head` divergence between two same-architecture cards has a
remedy, or whether the legacy path simply differs everywhere.

THREE OUTCOMES, ALL INFORMATIVE, WHICH IS WHY BOTH ARMS ARE RECORDED RATHER
THAN JUST THE INTERESTING ONE:

* ``mm`` differs across cards and ``addmm`` agrees -- the divergence is the
  entry point, and giving `lm_head` a zero bias fixes it for one add per
  output element.
* Both differ -- routing does not help, and the residual is not addressable
  this way.
* Neither differs -- the shapes here are not the ones that carry it, and the
  next question is which are.

It also records whether the two arms agree ON ONE CARD, because that is the
control: they compute the same product in real arithmetic, so a within-card
difference is reduction order and a within-card agreement means the card gave
both arms the same kernel and the cross-card comparison says nothing about
routing.

Determinism is pinned through ``probe_determinism`` for the reason the gemm
probe gives: a measurement taken under a different posture than the trace that
motivated it is not describing the same configuration.
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
from model_trainer.core.services.model.gemm_shapes import DIGEST_SUFFIX, SUM_SUFFIX
from model_trainer.core.services.model.legacy_gemm_probe import (
    EPILOGUE_ARM,
    LEGACY_ARM,
    LM_HEAD_SHAPES,
    arm_outputs,
    arms_agree,
)
from model_trainer.core.services.model.tensor_digest import describe_tensor

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: What this experiment answers. Distinct from the gemm probe's, because a
#: record here carries two arms per shape and one differenced against a
#: single-arm record would report every observation unmatched.
LEGACY_GEMM_EXPERIMENT = "lm-head-entry-point"

#: Label for the record, fixed because the table is read whole.
LEGACY_GEMM_LABEL = "lm-head-entry-point-v1"

#: Suffix marking the within-card control.
SAME_SUFFIX = "arms_agree"

#: How the control reads when the two arms produced identical bytes.
ARMS_AGREE = 1.0

#: How it reads when they did not.
ARMS_DIFFER = 0.0


def arm_label(origin: str, arm: str, suffix: str) -> str:
    """Name one observation.

    Args:
        origin: Which shape, e.g. ``"xl-lm-head"``.
        arm: Which entry point, from
            :data:`~legacy_gemm_probe.ARMS`.
        suffix: What is being reported.

    Returns:
        e.g. ``"xl-lm-head|mm|digest"``.
    """
    return f"{origin}|{arm}|{suffix}"


def legacy_run_record(device: str) -> RunRecord:
    """Pin determinism, run both arms of every shape, and record them.

    Args:
        device: Device to run on.

    Returns:
        The record: five observations per shape -- a digest and a sum for
        each arm, plus the within-card control.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(device, probe_determinism(device))

    observations: list[Observation] = []
    for shape in LM_HEAD_SHAPES:
        origin = shape["origin"]
        outputs = arm_outputs(shape, device)
        same = arms_agree(outputs)

        for arm, tensor in ((LEGACY_ARM, outputs["legacy"]), (EPILOGUE_ARM, outputs["epilogue"])):
            digest, total = describe_tensor(tensor)
            observations.append(
                Observation(name=arm_label(origin, arm, DIGEST_SUFFIX), value=digest)
            )
            observations.append(Observation(name=arm_label(origin, arm, SUM_SUFFIX), value=total))

        observations.append(
            Observation(
                name=f"{origin}|{SAME_SUFFIX}",
                value=ARMS_AGREE if same else ARMS_DIFFER,
            )
        )
        _log.info(
            "lm-head %s M%d K%d N%d arms_agree=%s",
            origin,
            shape["rows"],
            shape["inner"],
            shape["cols"],
            same,
        )

    return run_record(
        experiment=LEGACY_GEMM_EXPERIMENT,
        label=LEGACY_GEMM_LABEL,
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run both arms of every output-projection shape and write the record.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value.
        OSError: If the output cannot be written.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = legacy_run_record(device)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    _log.info(
        "lm-head entry points over %d shapes %s -> %s",
        len(LM_HEAD_SHAPES),
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
        service_name="lm-head-entry-point-probe",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "ARMS_AGREE",
    "ARMS_DIFFER",
    "LEGACY_GEMM_EXPERIMENT",
    "LEGACY_GEMM_LABEL",
    "SAME_SUFFIX",
    "arm_label",
    "entrypoint",
    "legacy_run_record",
    "main",
]


# Without this, `python -m model_trainer.cli.legacy_gemm_probe` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
