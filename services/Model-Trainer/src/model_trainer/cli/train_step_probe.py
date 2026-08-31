"""Take one training step per rung on one device and record every gradient.

The training twin of :mod:`probe_trace`. That command walks forward passes
and records every module boundary; this one runs ``loss.backward()`` and one
SGD update, and records every parameter's gradient and post-update value.
Run it on several cards and the first parameter whose gradient digests stop
matching names where the backward pass leaves the cards' agreement -- which
no forward measurement can say, because a backward pass runs GEMM shapes and
scatter-adds no forward pass issues.

WHY IT SHARES THE PROBE'S DETERMINISM PIN. It imports ``probe_determinism``
from the gate CLI for the reason the ladder, the trace and the gemm probe
do: a step measured under a different posture than the forward measurements
it is read against would not be describing the same configuration. The pin
also carries ``torch.use_deterministic_algorithms(True)``, which is what
puts the embedding gradient's scatter-add on its deterministic path -- and
the record still proves self-reproduction rather than assuming it, by
running every step twice.
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
from model_trainer.cli.probe_trace import workspace_observation
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.control_arms import CONTROLS_FLAG, require_control_arm
from model_trainer.core.services.model.deterministic_gemm import require_kernel_arm
from model_trainer.core.services.model.probe_shapes import require_probe_shape
from model_trainer.core.services.model.trace_plan import DIGEST_SUFFIX, SUM_SUFFIX
from model_trainer.core.services.model.train_step_plan import (
    RUNGS_FLAG,
    TRAIN_STEP_EXPERIMENT,
    require_train_rungs,
    train_loss_name,
    train_step_label,
    train_tensor_name,
)
from model_trainer.core.services.model.train_step_probe import (
    TrainTensor,
    train_step_identity,
)

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"
KERNEL_FLAG = "--kernel"

_FLAGS = (DEVICE_FLAG, RUNGS_FLAG, OUT_FLAG, CONTROLS_FLAG, KERNEL_FLAG)


def step_observations(rung: str, tensors: tuple[TrainTensor, ...]) -> tuple[Observation, ...]:
    """Name one rung's digested tensors.

    Args:
        rung: The rung they came from.
        tensors: The tensors, in walk order.

    Returns:
        Two observations per tensor, its digest and its sum.
    """
    observations: list[Observation] = []
    for tensor in tensors:
        for suffix, value in ((DIGEST_SUFFIX, tensor["digest"]), (SUM_SUFFIX, tensor["total"])):
            observations.append(
                Observation(
                    name=train_tensor_name(rung, tensor["kind"], tensor["path"], suffix),
                    value=value,
                )
            )
    return tuple(observations)


def train_step_run_record(
    device: str,
    rungs: tuple[str, ...],
    *,
    controls: str,
    kernel: str,
) -> RunRecord:
    """Pin determinism, step every rung twice, and record what they ran on.

    Determinism is pinned FIRST, before any rung builds a model, because
    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created and
    constructing a model on cuda is enough to create it.

    Args:
        device: Device to step every rung on.
        rungs: The rungs to walk, in order, already known distinct.
        controls: Which cross-card controls to apply, by
            :data:`~control_arms.CONTROL_ARMS` name.
        kernel: Which arithmetic the model's matmuls use, by
            :data:`~deterministic_gemm.KERNEL_ARMS` name.

    Returns:
        The record: two observations per digested tensor, one loss per rung,
        the split-K condition, and the fingerprint they all ran under.

    Raises:
        KeyError: Propagated from
            :func:`~probe_shapes.require_probe_shape` when a rung is not one
            the ladder declares. Nothing is stepped under a name that names
            no shape.
        RuntimeError: Propagated from
            :func:`~train_step_probe.train_step_identity` when a step did not
            reproduce itself.
        ValueError: Propagated from :func:`~control_arms.require_control_arm`
            or :func:`~deterministic_gemm.require_kernel_arm` for an unknown
            arm, or from :func:`~train_step_probe.train_step_once`.
    """
    remove_split_k, math_attention = require_control_arm(controls)
    named_kernel = require_kernel_arm(kernel)
    label = train_step_label(rungs, controls, named_kernel)
    shapes = tuple((rung, require_probe_shape(rung)) for rung in rungs)

    # Read BEFORE anything computes, for the reason probe_trace reads it
    # there: a record that cannot name its own arm is worth nothing, and
    # finding that out must not cost a GPU-hour first.
    workspace = workspace_observation()

    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=remove_split_k, math_attention=math_attention),
    )

    observations: list[Observation] = [workspace]
    for rung, shape in shapes:
        tensors, loss = train_step_identity(device, shape, kernel=named_kernel)
        # Logged per rung rather than only at the end, so a job killed by a
        # wall clock or preemption leaves a trail of which rungs succeeded.
        _log.info("rung %s stepped, %d tensors digested, loss %.17g", rung, len(tensors), loss)
        observations.extend(step_observations(rung, tensors))
        observations.append(Observation(name=train_loss_name(rung), value=loss))

    return run_record(
        experiment=TRAIN_STEP_EXPERIMENT,
        label=label,
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Step every rung once and write the record.

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
    rungs = require_train_rungs(cli_args.require_flag(parsed, RUNGS_FLAG))
    controls = cli_args.require_flag(parsed, CONTROLS_FLAG)
    kernel = cli_args.require_flag(parsed, KERNEL_FLAG)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = train_step_run_record(device, rungs, controls=controls, kernel=kernel)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "train step %s over %d observations %s -> %s",
        record["label"],
        len(record["observations"]),
        describe_run_fingerprint(record["fingerprint"]),
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
        service_name="train-step-probe",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "entrypoint",
    "main",
    "step_observations",
    "train_step_run_record",
]


# Without this, `python -m model_trainer.cli.train_step_probe` IMPORTS the
# module, runs nothing, and exits 0 -- measured on the gate probe, where two
# Slurm jobs "succeeded" in six seconds having written no record and no
# stderr.
if __name__ == "__main__":
    entrypoint()
