"""Time whole GPT-2 TRAINING STEPS with and without the attention pin.

The measurement every earlier page in this thread refused to make. The
forward benchmark found the pin costs 1.14-1.31x and no extra memory, and
said each time that this licensed nothing about training, because the
backward pass through the math backend is a different computation and the
forward ran under ``no_grad``.

BOTH ARMS TIME THE SAME WEIGHTS AND THE SAME OPTIMIZER STATE. Each row is
built once and stepped by both arms, so a fresh random init cannot enter the
comparison. The arms therefore run in sequence on one optimizer, which is
what a training run does to its own weights anyway -- and neither arm's
TIMING depends on what the weights contain.

WHY MEMORY IS THE THING TO WATCH HERE. A training forward keeps what backward
needs, and what the math path needs is the full score matrix -- the tensor the
fused kernel exists to avoid keeping. The forward benchmark's "no extra
memory" cannot survive that, and finding out by how much is most of the point.
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
from model_trainer.cli.sdpa_benchmark import PINNED_KEY, cost_observations
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.cost_labels import DEFAULT_KEY
from model_trainer.core.services.model.forward_cost import ForwardCostShape, release_row
from model_trainer.core.services.model.sdpa_probe import BACKENDS
from model_trainer.core.services.model.train_cost import time_train_step, train_step_setup

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: Distinct from the forward-pass experiment. A step is a forward, a
#: backward and an optimizer update; a forward is a forward. Two records
#: answering those must not be differenced against each other.
TRAIN_COST_EXPERIMENT = "training-step-cost"

#: Label for the record. Fixed rather than digest-derived: the sweep is read
#: as a whole, and a record missing a row shows up as missing observations.
TRAIN_COST_LABEL = "training-step-cost-v1"


def train_prefix(shape: ForwardCostShape) -> str:
    """Name one timed training step, without saying which measurement.

    The dimensions are in the name rather than only in the table, so a record
    read on its own still says what it timed -- including the VOCABULARY,
    which moves attention's share of the pass and therefore moves the answer.

    Args:
        shape: The row.

    Returns:
        e.g. ``train-small-b8-s512-v50257``.
    """
    return f"train-{shape['name']}-v{shape['vocab_size']}"


def measure_row(shape: ForwardCostShape, device: str) -> tuple[Observation, ...]:
    """Build one row, time both arms on it, and name what they produced.

    A function rather than a loop body so that the model it builds is
    unreachable the moment it returns. That is what lets
    :func:`~forward_cost.release_row` actually free anything.

    Args:
        shape: The row to measure.
        device: Device to time on.

    Returns:
        The observations for both arms.
    """
    prefix = train_prefix(shape)
    step = train_step_setup(shape, device)
    observations: list[Observation] = []
    for key, backend in ((DEFAULT_KEY, None), (PINNED_KEY, BACKENDS[PINNED_KEY])):
        cost = time_train_step(step, device, backend)
        # Logged per row rather than only at the end: the largest rows are
        # the ones a wall clock is most likely to cut short, and a job that
        # died there would otherwise leave no record of what it had already
        # measured.
        _log.info(
            "train %s %s: %s",
            shape["name"],
            key,
            "DID NOT FIT"
            if cost is None
            else f"{cost['seconds'] * 1e3:.3f} ms/step "
            f"(spread {cost['spread'] * 1e3:.3f}) "
            f"peak {cost['peak_bytes'] / 2**20:.0f} MiB",
        )
        observations.extend(cost_observations(prefix, key, cost))
    return tuple(observations)


def train_run_record(device: str) -> RunRecord:
    """Pin determinism, time every row under both arms, and record it.

    Args:
        device: Device to time on.

    Returns:
        The record.

    Raises:
        RuntimeError: Propagated when a pass failed for a reason that is not
            an out-of-memory.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )

    observations: list[Observation] = []
    for shape in _test_hooks.train_shapes():
        observations.extend(measure_row(shape, device))
        # The row's model goes out of scope when `measure_row` returns, and
        # only then can its blocks be handed back. Doing this between rows
        # rather than at the end is what keeps two models from being
        # resident at once -- see `release_row` for the 436-millisecond
        # measurement that mistake produced.
        release_row()

    return run_record(
        experiment=TRAIN_COST_EXPERIMENT,
        label=TRAIN_COST_LABEL,
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the forward sweep once and write the record.

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

    record = train_run_record(cli_args.require_flag(parsed, DEVICE_FLAG))

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "training-step cost over %d rows %s -> %s",
        len(_test_hooks.train_shapes()),
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
        service_name="train-benchmark",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "TRAIN_COST_EXPERIMENT",
    "TRAIN_COST_LABEL",
    "entrypoint",
    "main",
    "measure_row",
    "train_prefix",
    "train_run_record",
]


# Without this, `python -m model_trainer.cli.train_benchmark` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
