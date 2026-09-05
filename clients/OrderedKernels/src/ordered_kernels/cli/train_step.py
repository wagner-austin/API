"""The train-step probe under the ordered kernels.

Record-compatible with Model-Trainer's ``train_step_probe`` -- same
experiment string, same observation names, same twice-run self-check -- with
the model's projections swapped onto :mod:`ordered_kernels.modules`.
Controls hardwired to ``both``, as everywhere in this package. Two arms,
chosen by ``--attention``, stated explicitly on every invocation:

* ``--attention vendor`` is the arm labelled ``ordered`` -- projections and
  ``lm_head`` owned, attention on the vendor's math pin. The load-bearing
  expectation: its record equals an ``owned`` record tensor for tensor,
  since the two arms are one arithmetic in different clothes; the cluster
  asserts it card by card. Its records are the pinned corpus, which is why
  the arm's label and arithmetic never change.
* ``--attention ordered`` is the arm labelled ``ordered-full`` -- the
  attention walk runs first, so all three attention reductions AND their
  backward reductions are program-ordered too. New label, new records: the
  any-length training claim lives here without touching what the corpus
  pins.

THE MODEL STAYS IN EVAL MODE, in both arms, for ``train_step_probe``'s
documented reason: train mode enables dropout, whose Philox masks are a
question about the RNG rather than the arithmetic under study, and
gradients flow identically in eval mode. ``OrderedSdpaAttention`` enforces
the same discipline by refusing training mode outright.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

import torch
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.cli.probe_trace import workspace_observation
from model_trainer.cli.train_step_probe import step_observations
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import ProbeShape, require_probe_shape
from model_trainer.core.services.model.train_step_plan import (
    RUNGS_FLAG,
    TRAIN_STEP_EXPERIMENT,
    require_train_rungs,
    train_loss_name,
    train_step_label,
)
from model_trainer.core.services.model.train_step_probe import (
    TrainTensor,
    digest_step_tensors,
    require_step_reproduced,
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

from ordered_kernels.cli.gemm_probe import ORDERED_ARM
from ordered_kernels.modules import use_ordered_attention, use_ordered_kernels

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"
ATTENTION_FLAG = "--attention"

#: The fully-owned training arm's label. Distinct from ``ORDERED_ARM`` so
#: the corpus's pinned ``ordered`` records keep meaning what they meant.
ORDERED_FULL_ARM = "ordered-full"

#: ``--attention`` value -> (run the attention walk, record label arm).
_ATTENTION_ARMS: dict[str, tuple[bool, str]] = {
    "vendor": (False, ORDERED_ARM),
    "ordered": (True, ORDERED_FULL_ARM),
}

_FLAGS = (DEVICE_FLAG, RUNGS_FLAG, OUT_FLAG, ATTENTION_FLAG)


def require_attention_arm(raw: str) -> tuple[bool, str]:
    """Resolve the ``--attention`` value, refusing anything undeclared.

    Args:
        raw: The flag's value.

    Returns:
        ``(run the attention walk, the record's arm label)``.

    Raises:
        ValueError: For a value outside the two declared arms -- a record
            whose attention posture was guessed names a condition it may not
            have run under.
    """
    if raw not in _ATTENTION_ARMS:
        raise ValueError(f"{ATTENTION_FLAG} must be one of {sorted(_ATTENTION_ARMS)}, got {raw!r}")
    return _ATTENTION_ARMS[raw]


def require_swapped(replaced: int) -> int:
    """Return the swap count, refusing zero.

    A separate function for the reason every guard here is one: the real
    builder always yields swappable modules, so the refusing arm cannot be
    reached through it, and an arm no test can drive is an arm nobody has
    confirmed says what it means.

    Args:
        replaced: What ``use_ordered_kernels`` reported.

    Returns:
        ``replaced``, once known non-zero.

    Raises:
        RuntimeError: On zero -- a record claiming an arm that did not run
            is the defect every arm in this experiment refuses.
    """
    if replaced == 0:
        raise RuntimeError("the ordered swap replaced nothing; this record would lie")
    return replaced


def ordered_step_once(
    device: str, shape: ProbeShape, ordered_attention: bool
) -> tuple[tuple[TrainTensor, ...], float]:
    """Build one rung's model fresh, swap it ordered, take one step.

    Args:
        device: Device to run on.
        shape: The rung to build.
        ordered_attention: Whether to run the attention walk first, so the
            step's attention reductions -- forward and backward -- are owned
            too. The projections walk runs either way, and finds the
            attention wrapper's held projections when it does.

    Returns:
        ``(digested tensors, the loss)``.

    Raises:
        RuntimeError: When a swap matched nothing -- a record claiming an
            arm that did not run is the defect every arm here refuses.
        ValueError: Propagated from the builder, the swaps, or the digests.
    """
    model, ids = probe_model_and_input(device, shape)
    if ordered_attention:
        require_swapped(use_ordered_attention(model))
    require_swapped(use_ordered_kernels(model))
    outputs = model.forward(input_ids=ids, labels=ids)
    loss = outputs.loss
    torch.autograd.backward([loss])
    return digest_step_tensors(model), float(loss.item())


def ordered_step_identity(
    device: str, shape: ProbeShape, ordered_attention: bool
) -> tuple[tuple[TrainTensor, ...], float]:
    """Take the same ordered step twice and refuse a card that cannot repeat.

    Args:
        device: Device to run on.
        shape: The rung to run.
        ordered_attention: Passed through to :func:`ordered_step_once`.

    Returns:
        ``(digested tensors, the loss)``.

    Raises:
        RuntimeError: Propagated from ``require_step_reproduced`` or from
            :func:`ordered_step_once`.
        ValueError: Propagated from :func:`ordered_step_once`.
    """
    first, first_loss = ordered_step_once(device, shape, ordered_attention)
    second, second_loss = ordered_step_once(device, shape, ordered_attention)
    return require_step_reproduced(first, second, first_loss, second_loss, device)


def ordered_train_record(device: str, rungs: tuple[str, ...], attention: str) -> RunRecord:
    """Pin the posture, step every rung twice, and record everything.

    Args:
        device: Device to step on.
        rungs: The rungs to walk, already known distinct.
        attention: The ``--attention`` value, resolved by
            :func:`require_attention_arm` before anything computes.

    Returns:
        The record, labelled ``train-step-...-both-ordered`` or
        ``...-both-ordered-full``.

    Raises:
        KeyError: Propagated from ``require_probe_shape`` for an undeclared
            rung, before anything computes.
        RuntimeError: Propagated from :func:`ordered_step_identity`.
        ValueError: For an undeclared attention arm, or propagated from
            :func:`ordered_step_once`.
    """
    ordered_attention, arm = require_attention_arm(attention)
    label = train_step_label(rungs, "both", arm)
    shapes = tuple((rung, require_probe_shape(rung)) for rung in rungs)
    workspace = workspace_observation()
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=True, math_attention=True),
    )
    observations: list[Observation] = [workspace]
    for rung, shape in shapes:
        tensors, loss = ordered_step_identity(device, shape, ordered_attention)
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
    rungs = require_train_rungs(cli_args.require_flag(parsed, RUNGS_FLAG))
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    attention = cli_args.require_flag(parsed, ATTENTION_FLAG)

    record = ordered_train_record(device, rungs, attention)

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

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="ordered-train-step",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "ORDERED_FULL_ARM",
    "entrypoint",
    "main",
    "ordered_step_identity",
    "ordered_step_once",
    "ordered_train_record",
    "require_attention_arm",
    "require_swapped",
]


if __name__ == "__main__":
    entrypoint()
