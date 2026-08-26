"""Run the environment known-answer probe and emit a run record.

The probe itself is
:func:`model_trainer.core.services.model.known_answer_probe.probe_forward_loss`;
this is the command a job document can name. It exists as a separate entry
from ``modeltrainer-score-baseline`` because it answers a different question:
the scorer measures an experiment, this measures the environment the
experiment would run in, and it does so in seconds without staging anything.

WHY IT WRITES A FULL RunRecord RATHER THAN A NUMBER. The probe's output is
meaningless apart from the image, card, driver and determinism settings that
produced it -- that is the entire premise of
:mod:`platform_core.known_answer`. Writing the number alone would recreate
the defect the record exists to close: the 52.3030% floor was stored with no
device, no driver and no torch version, and could not afterwards be told
apart from a working image that had merely moved cards.

The fingerprint comes from :func:`capture_run_fingerprint`, the same function
the scorer uses, rather than from anything assembled here. That is deliberate
and was learned: the first three probe runs recorded their card from a batch
script's prologue, the prologue differed between two of them, and the result
was a measurement whose ``driver_version`` simply did not exist for one card.
A run that records part of its own fingerprint from its launcher can be
launched a way that forgets.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)
from platform_ml.determinism import with_torch_thread_count

from model_trainer.cli import _test_hooks
from model_trainer.core.run_fingerprint import (
    CUDA_DEVICE,
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.known_answer_probe import (
    PROBE_EXPERIMENT,
    PROBE_LABEL,
    PROBE_OBSERVATION,
    probe_forward_loss,
)

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)


PROBE_CPU_THREADS = 1
"""Thread count a cpu probe pins itself to.

STABLE BY CONSTRUCTION RATHER THAN BY LUCK, which is a weaker claim than the
one first written here and is the one the measurements support.

This probe is NOT currently sensitive to the count. Measured 2026-08-26 by
running ``probe_forward_loss("cpu")`` at 1, 2, 4 and 8 threads: all four
return 6.250983238220215, bit for bit. A 4096x4096 matmul at one thread and
at eight DOES differ, in 865,498 of 16,777,216 elements -- but that is a
different computation, and citing it as the reason this probe needed pinning
was an inference, not a result.

The pin stays because the count is a real degree of freedom that otherwise
varies with the node's core count, unpinned and unrecorded. Invariance at
L2/d128/h2/v512/len64 on one machine's BLAS is an empirical accident of size,
not a guarantee that survives a larger probe or a different library. One
thread costs nothing here and makes the number reproducible by construction.
"""


def probe_determinism(device: str) -> DeterminismRecord:
    """Pin whatever governs this device's arithmetic, and report it.

    Device-conditional for the same reason
    :func:`~model_trainer.core.run_fingerprint.capture_run_fingerprint` reads
    the card and driver only for cuda: a setting that did not govern a
    measurement does not belong in its record. No card computed a cpu number,
    so a cpu record carries none; CUDA kernels do a cuda run's arithmetic and
    the host thread count does not enter it, so a cuda record carries none.

    A CPU run's thread count participates in the arithmetic -- a reduction
    split across threads sums its partials in completion order -- and the
    probe pinned none at all, so it varied with the answering node's core
    count, unrecorded. See :data:`PROBE_CPU_THREADS` for what that is and is
    not worth: this probe turns out to be INSENSITIVE to the count, so the
    pin buys reproducibility by construction rather than fixing a live drift.

    MEASURED, on cuda, 2026-08-26 -- HPC3 jobs 55598648 and 55598652, one
    A100 each in image ``1112dbb1``, driver 580.82.07, identical but for
    ``OMP_NUM_THREADS``:

        resolved_threads 1 -> 6.250983715057373
        resolved_threads 8 -> 6.250983715057373

    Bit-identical, and the artifacts carry ``resolved_threads`` so the
    manipulation is evidenced rather than assumed. The host thread count does
    not reach a cuda result, which is why one is not recorded on that path.

    That also spares a real cost: all eight entries in
    ``known-answers.json`` were registered without a thread count, so adding
    one to the cuda axis would make every future probe report
    ``configuration_differs`` against every one of them -- for a setting now
    shown not to change the number.

    Args:
        device: Device the measurement runs on.

    Returns:
        What was pinned, ready to place in the fingerprint.
    """
    applied = _test_hooks.apply_determinism_hook()
    if device == CUDA_DEVICE:
        return applied
    return with_torch_thread_count(applied, _test_hooks.pin_torch_threads(PROBE_CPU_THREADS))


def probe_run_record(device: str) -> RunRecord:
    """Pin determinism, run the probe, and record what it ran on.

    Determinism is pinned FIRST, before the probe builds a model, because
    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created and
    constructing a model on cuda is enough to create it.

    Args:
        device: Device to run the probe on.

    Returns:
        The record: the probe loss as its single observation, and the
        fingerprint of the configuration that produced it. The payload digest
        is :const:`NO_PAYLOAD` -- the probe's entire output IS the
        observation, so a digest over it would restate the number rather than
        add the independent check a digest is for.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(device, probe_determinism(device))

    loss = probe_forward_loss(device)

    return run_record(
        experiment=PROBE_EXPERIMENT,
        label=PROBE_LABEL,
        fingerprint=fingerprint,
        observations=(Observation(name=PROBE_OBSERVATION, value=loss),),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe once and write its record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent. Nothing is computed on a command line
            that was not understood, because a probe run on a device other
            than the one named would write a record claiming to be this one.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    record = probe_run_record(cli_args.require_flag(parsed, DEVICE_FLAG))

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "probe %s = %.17g %s -> %s",
        record["label"],
        record["observations"][0]["value"],
        describe_run_fingerprint(record["fingerprint"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = ["entrypoint", "main", "probe_determinism", "probe_run_record"]


# Without this, `python -m model_trainer.cli.known_answer_probe` IMPORTS the
# module, runs nothing, and exits 0. Measured: HPC3 jobs 55595084 and 55595086
# each "succeeded" in six seconds having written no record and no stderr, and
# only the absent output file said so. A silent no-op that reports success is
# worse than a crash, and a console script alone does not cover it because a
# job document may name the module rather than the script.
if __name__ == "__main__":
    entrypoint()
