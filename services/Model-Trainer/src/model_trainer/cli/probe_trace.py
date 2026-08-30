"""Trace several rungs on one device and record every module boundary.

Where :mod:`probe_ladder` records one number per rung, this records several
thousand: a digest and a sum for every tensor entering or leaving a module,
in execution order. Run it on several cards and the FIRST observation whose
digests stop matching names the operation that carries the difference, which
a loss cannot.

WHY ONE RECORD FOR ALL THE RUNGS, as the ladder does: every rung runs in one
process on one card under one pin, so they genuinely share one fingerprint,
and writing it four times would invite the four copies to drift in a way
nothing checks. Each observation carries its rung in its own name.

WHY IT SHARES THE PROBE'S DETERMINISM PIN. It imports ``probe_determinism``
from the gate CLI for the reason the ladder and the gemm probe do: a trace
taken under a different posture than the ladder it explains would not be
describing the same configuration, and the two could not be read together.

WHAT IT COSTS. The largest rung digests about a hundred and seventy million
floats, which is a minute of CPU beside a forward pass measured in
milliseconds. That asymmetry is the point: the arithmetic under study is
cheap and the observation of it is not, and doing it the other way round --
sampling some boundaries -- would leave exactly the gaps a first-difference
reading depends on not having.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.determinism_env import CUBLASLT_WORKSPACE_ENV_VAR
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
from model_trainer.core.services.model.forward_trace import TracedTensor, traced_forward
from model_trainer.core.services.model.probe_shapes import require_probe_shape
from model_trainer.core.services.model.trace_plan import (
    DIGEST_SUFFIX,
    SUM_SUFFIX,
    TRACE_EXPERIMENT,
    WORKSPACE_NAME,
    WORKSPACE_UNSET,
    TraceName,
    describe_workspace,
    trace_label,
    trace_loss_name,
    trace_tensor_name,
)

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"
KERNEL_FLAG = "--kernel"

_FLAGS = (DEVICE_FLAG, OUT_FLAG, CONTROLS_FLAG, KERNEL_FLAG)


def tensor_observations(rung: str, traced: tuple[TracedTensor, ...]) -> tuple[Observation, ...]:
    """Name one rung's traced tensors.

    Args:
        rung: The rung they came from.
        traced: The tensors, in execution order.

    Returns:
        Two observations per tensor, its digest and its sum.
    """
    observations: list[Observation] = []
    for tensor in traced:
        for suffix, value in ((DIGEST_SUFFIX, tensor["digest"]), (SUM_SUFFIX, tensor["total"])):
            name = TraceName(
                rung=rung,
                step=tensor["step"],
                kind=tensor["kind"],
                index=tensor["index"],
                module_class=tensor["module_class"],
                path=tensor["path"],
                suffix=suffix,
            )
            observations.append(Observation(name=trace_tensor_name(name), value=value))
    return tuple(observations)


def workspace_observation() -> Observation:
    """Record which split-K condition this process is running under.

    Read here rather than left to the filename, because the whole experiment
    is a contrast between two values of one variable and a record that cannot
    name its own arm is a record whose arm someone has to remember. See
    :data:`~model_trainer.core.services.model.trace_plan.WORKSPACE_NAME` for
    why this is an observation and not a fingerprint axis.

    Returns:
        The observation: the size in bytes, or
        :data:`~model_trainer.core.services.model.trace_plan.WORKSPACE_UNSET`
        when the variable is not set.

    Raises:
        ValueError: If the variable is set to something that is not an
            integer. Recording it as "unset" would be a lie and recording
            nothing would leave the arm unnamed, so the run stops instead --
            before spending a GPU on a measurement whose condition it cannot
            report.
    """
    raw = _test_hooks.env_cublaslt_workspace()
    if raw is None:
        return Observation(name=WORKSPACE_NAME, value=WORKSPACE_UNSET)
    if not raw.lstrip("-").isdigit():
        raise ValueError(
            f"{CUBLASLT_WORKSPACE_ENV_VAR} is {raw!r}, which is not an integer; "
            "this trace could not say which condition it ran under"
        )
    return Observation(name=WORKSPACE_NAME, value=float(int(raw)))


def trace_run_record(
    device: str,
    rungs: tuple[str, ...],
    *,
    remove_split_k: bool,
    math_attention: bool,
    kernel: str,
) -> RunRecord:
    """Pin determinism, trace every rung, and record what they ran on.

    Determinism is pinned FIRST, before any rung builds a model, because
    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created and
    constructing a model on cuda is enough to create it. The controls are part
    of that pin, so they are applied there too -- and the record says which
    were, because :func:`~platform_ml.determinism.apply_determinism` writes
    ``cublaslt_split_k`` and ``sdpa_backends`` into the determinism block only
    when it applied them.

    Args:
        device: Device to trace every rung on.
        remove_split_k: Whether to take split-K out of cuBLASLt's options.
        math_attention: Whether to restrict attention to the math kernel.
        kernel: Which arithmetic the model's matmuls use, by
            :data:`~deterministic_gemm.KERNEL_ARMS` name. The controls
            above pick which VENDOR kernel runs; this picks whether a
            vendor kernel runs at all, so it is in the record's LABEL
            rather than beside it -- see :func:`trace_label`.
        rungs: The rungs to walk, in order. Taken as an argument rather than
            read from the module so the suite can exercise this on one cheap
            rung: the declared set ends in a 1.5-billion-parameter model,
            which is a GPU measurement.

    Returns:
        The record: two observations per traced tensor, one loss per rung,
        and the fingerprint they all ran under. The payload digest is
        :const:`NO_PAYLOAD` -- the digests ARE the output.

    Raises:
        KeyError: Propagated from
            :func:`~model_trainer.core.services.model.probe_shapes.require_probe_shape`
            when a rung is not one the ladder declares. Nothing is traced
            under a name that names no shape.
        ValueError: Propagated from :func:`trace_label` when a rung repeats,
            or from :func:`~kernel_arm_modules.use_kernel_arm` for an
            unknown arm.
    """
    label = trace_label(rungs, kernel)
    shapes = tuple((rung, require_probe_shape(rung)) for rung in rungs)

    # Read BEFORE anything computes. A trace that cannot name its own arm is
    # worth nothing, so finding that out must not cost a GPU-hour first.
    workspace = workspace_observation()

    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=remove_split_k, math_attention=math_attention),
    )

    observations: list[Observation] = [workspace]
    for rung, shape in shapes:
        traced, loss = traced_forward(device, shape, kernel=kernel)
        # Logged per rung rather than only at the end: the large rungs take
        # long enough that a job killed by a wall clock or preemption would
        # otherwise leave no record of which rungs had already succeeded.
        _log.info("rung %s traced %d tensors, loss %.17g", rung, len(traced), loss)
        observations.extend(tensor_observations(rung, traced))
        observations.append(Observation(name=trace_loss_name(rung), value=loss))

    return run_record(
        experiment=TRACE_EXPERIMENT,
        label=label,
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Trace every rung once and write the record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent. Nothing is computed on a command line
            that was not understood.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    # Every flag resolved BEFORE anything computes, so a command line that
    # cannot be honoured costs nothing rather than a GPU-hour.
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    remove_split_k, math_attention = require_control_arm(
        cli_args.require_flag(parsed, CONTROLS_FLAG)
    )
    kernel = require_kernel_arm(cli_args.require_flag(parsed, KERNEL_FLAG))
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = trace_run_record(
        device,
        _test_hooks.trace_rungs(),
        remove_split_k=remove_split_k,
        math_attention=math_attention,
        kernel=kernel,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    workspace = next(o for o in record["observations"] if o["name"] == WORKSPACE_NAME)
    _log.info(
        "trace %s over %d observations %s cublaslt_workspace=%s -> %s",
        record["label"],
        len(record["observations"]),
        describe_run_fingerprint(record["fingerprint"]),
        describe_workspace(workspace["value"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Logging is configured here rather than left to whatever a caller did,
    because the per-rung lines are this command's only partial output.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="probe-trace",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "entrypoint",
    "main",
    "tensor_observations",
    "trace_run_record",
    "workspace_observation",
]


# Without this, `python -m model_trainer.cli.probe_trace` IMPORTS the module,
# runs nothing, and exits 0 -- measured on the gate probe, where two Slurm
# jobs "succeeded" in six seconds having written no record and no stderr.
if __name__ == "__main__":
    entrypoint()
