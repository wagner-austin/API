"""What cross-card bitwise agreement costs, measured on the card.

Disabling cuBLASLt's split-K makes three GPUs produce bit-identical tensors on
every probed shape. Split-K exists to make long reductions fast, and it is
selected on exactly the large shapes -- so the open question is not whether
the fix works but what it costs, and that decides whether it is a sane default
or a switch to flip only when two runs must be compared.

HOW THE TWO CONDITIONS ARE MEASURED. ``CUBLASLT_WORKSPACE_SIZE`` is read once,
when the cuBLASLt handle is created. Setting it part-way through a process does
nothing -- MEASURED, not assumed: two ``addmm`` calls with the variable set
between them both used split-K, 2 of 2. So one process can measure exactly one
condition, and this command runs the default in-process and re-executes itself
in a child for the other.

WHY A CHILD RATHER THAN TWO JOBS. Both conditions must run on the SAME GPU. Two
Slurm jobs can land on different nodes, and the run contract admits one artifact
per document anyway, so a two-job comparison would be two records with no
guarantee they are comparable. A child process shares the allocation, the card
and the thermal state.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.determinism_env import (
    CUBLASLT_NO_SPLIT_K,
    CUBLASLT_WORKSPACE_ENV_VAR,
)
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    decode_run_record,
    encode_run_record,
    run_record,
)

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.gemm_shapes import GemmShape, gemm_label
from model_trainer.core.services.model.gemm_timing import time_gemm

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"
CONDITION_FLAG = "--condition"

_FLAGS = (DEVICE_FLAG, OUT_FLAG, CONDITION_FLAG)

BENCHMARK_EXPERIMENT = "gemm-splitk-cost"
BENCHMARK_LABEL = "gemm-splitk-cost-v1"

#: The environment variable that removes split-K from cuBLASLt's options, and
#: the value that does it.
#:
#: Re-exported rather than spelled, since
#: :func:`platform_ml.determinism.apply_determinism` now writes the same pair
#: for training runs. Two spellings of the variable this whole experiment
#: manipulates would let the measured condition and the applied one drift
#: apart in silence, which is the one failure no result here could survive.
WORKSPACE_VAR = CUBLASLT_WORKSPACE_ENV_VAR
NO_SPLIT_K = CUBLASLT_NO_SPLIT_K

#: The two conditions, and the suffix each one's timings are recorded under.
DEFAULT_CONDITION = "default"
NOSPLITK_CONDITION = "nosplitk"

SECONDS_SUFFIX = "seconds"
SPREAD_SUFFIX = "spread"


def timing_observations(
    device: str, condition: str, shapes: tuple[tuple[str, GemmShape], ...]
) -> tuple[Observation, ...]:
    """Time each given shape once, under whatever condition the process has.

    Args:
        device: Device to time on.
        condition: Which condition this process is running, used only to name
            the observations. It does NOT set the condition -- the environment
            did that before the process started, and a flag that claimed to
            set it here would silently measure the default twice.
        shapes: The shapes to time. Taken as an argument rather than read from
            the table so the suite can exercise this on one shape: timing is
            deliberately repetitive -- warmup plus several batches of many
            calls each -- and walking all 43 shapes per test spent minutes
            measuring a laptop nobody will read the numbers from.

    Returns:
        Two observations per shape: seconds per call and the batch spread.
    """
    observations: list[Observation] = []
    for name, shape in shapes:
        seconds, spread = time_gemm(shape, device)
        _log.info(
            "%s %s M%d K%d %.6f s/call (spread %.6f)",
            condition,
            name,
            shape["rows"],
            shape["inner"],
            seconds,
            spread,
        )
        seconds_name = f"{gemm_label(name, shape, SECONDS_SUFFIX)}|{condition}"
        spread_name = f"{gemm_label(name, shape, SPREAD_SUFFIX)}|{condition}"
        observations.append(Observation(name=seconds_name, value=seconds))
        observations.append(Observation(name=spread_name, value=spread))
    return tuple(observations)


def run_child(device: str, out: pathlib.Path) -> tuple[Observation, ...]:
    """Measure the no-split-K condition in a fresh process, and read it back.

    Args:
        device: Device to time on.
        out: Where the child writes its record.

    Returns:
        The child's observations.

    Raises:
        RuntimeError: If the child failed, or produced a record whose
            observations are not the no-split-K ones. A child that silently
            ran the default condition would make the whole comparison read as
            "the fix is free", which is the most damaging way this could be
            wrong.
    """
    completed = _test_hooks.run_benchmark_child(
        [
            sys.executable,
            "-m",
            "model_trainer.cli.gemm_benchmark",
            DEVICE_FLAG,
            device,
            OUT_FLAG,
            str(out),
            CONDITION_FLAG,
            NOSPLITK_CONDITION,
        ],
        WORKSPACE_VAR,
        NO_SPLIT_K,
    )
    if completed != 0:
        raise RuntimeError(f"the {NOSPLITK_CONDITION} child exited {completed}")

    child = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
    wrong = [o["name"] for o in child["observations"] if not o["name"].endswith(NOSPLITK_CONDITION)]
    if wrong:
        raise RuntimeError(f"the child recorded observations under another condition: {wrong[:3]}")
    return child["observations"]


def benchmark_run_record(device: str, out: pathlib.Path) -> RunRecord:
    """Time both conditions on one device and record them together.

    Args:
        device: Device to time on.
        out: The parent's output path, used to site the child's record beside
            it.

    Returns:
        The record: both conditions' timings under one fingerprint, because
        both ran on one card in one allocation.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )

    default = timing_observations(device, DEFAULT_CONDITION, _measurement_hooks.benchmark_shapes())
    child = run_child(device, out.with_name(f"{out.stem}-child{out.suffix}"))

    return run_record(
        experiment=BENCHMARK_EXPERIMENT,
        label=BENCHMARK_LABEL,
        fingerprint=fingerprint,
        observations=default + child,
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Time the shapes, in one condition or both.

    With no ``--condition`` this is the parent: it times the default in-process
    and spawns the child for the other. With ``--condition`` it times exactly
    what its environment gives it and writes that alone -- which is how the
    child is invoked, and is also usable directly.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)

    condition = parsed.get(CONDITION_FLAG)
    if condition is None:
        record = benchmark_run_record(device, out)
    else:
        record = run_record(
            experiment=BENCHMARK_EXPERIMENT,
            label=BENCHMARK_LABEL,
            fingerprint=capture_run_fingerprint(
                device, probe_determinism(device, remove_split_k=False, math_attention=False)
            ),
            observations=timing_observations(
                device, condition, _measurement_hooks.benchmark_shapes()
            ),
            payload_digest=NO_PAYLOAD,
        )

    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    _log.info(
        "%d timings %s -> %s",
        len(record["observations"]),
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
        service_name="gemm-benchmark",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_EXPERIMENT",
    "BENCHMARK_LABEL",
    "DEFAULT_CONDITION",
    "NOSPLITK_CONDITION",
    "NO_SPLIT_K",
    "SECONDS_SUFFIX",
    "SPREAD_SUFFIX",
    "WORKSPACE_VAR",
    "benchmark_run_record",
    "entrypoint",
    "main",
    "run_child",
    "timing_observations",
]


if __name__ == "__main__":
    entrypoint()
