"""Price the attention backend that makes three cards agree.

:mod:`sdpa_probe` established the correctness result: all three cards select
`EFFICIENT_ATTENTION`, the V100 disagrees with the Ampere pair inside it, and
forcing `MATH` gives one identical value on every card. This measures what
forcing `MATH` costs.

WHY IT SWEEPS RATHER THAN QUOTING A NUMBER. The math path materialises the
full score matrix and the fused kernel does not, so the price is quadratic in
sequence length where the alternative is linear. A single figure at the
ladder's 64 tokens would be the same error as the split-K cost table that had
to be corrected: measured at one unrepresentative point and read as the cost
of reproducibility. The sweep runs batch 1 and 8 against lengths 64 to 4096,
so the wall -- if there is one -- is inside the measured range.

WHY ONE SPLIT-K CONDITION. Measured, not assumed: the selection probe found
0 of 72 attention digests changing between the two conditions, so the same
kernels run either way and there is nothing for a second arm to move.
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
from model_trainer.core.services.model.cost_labels import (
    DEFAULT_KEY,
    FALSE_VALUE,
    FITTED_SUFFIX,
    PEAK_SUFFIX,
    SECONDS_SUFFIX,
    SPREAD_SUFFIX,
    TRUE_VALUE,
    labelled,
)
from model_trainer.core.services.model.sdpa_probe import BACKENDS
from model_trainer.core.services.model.sdpa_shapes import SDPA_COST_EXPERIMENT, cost_prefix
from model_trainer.core.services.model.sdpa_timing import time_sdpa
from model_trainer.core.services.model.timing_harness import MeasuredCost

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: Label for the record. Fixed rather than digest-derived: the sweep is read
#: as a whole, and a record missing a point shows up as missing observations.
SDPA_COST_LABEL = "sdpa-backend-cost-v1"

#: The backend under test, and the only one worth pricing: it is the one the
#: selection probe showed to be bit-identical across all three cards.
PINNED_KEY = "math"


def cost_observations(
    prefix: str, backend: str, cost: MeasuredCost | None
) -> tuple[Observation, ...]:
    """Name what one timed measurement produced.

    A call that did not fit contributes its ``fitted`` row and no timings.
    The asymmetry is deliberate and matches the selection probe: an absent
    timing is reported as an unmatched observation, so a configuration that
    fits on one card and not another is visible as a structural difference
    rather than hidden behind a sentinel that compares equal to nothing.

    Args:
        prefix: What was measured, including its dimensions.
        backend: Which arm it ran under.
        cost: What it cost, or None when it did not fit.

    Returns:
        The observations.
    """
    fitted = Observation(
        name=labelled(prefix, backend, FITTED_SUFFIX),
        value=FALSE_VALUE if cost is None else TRUE_VALUE,
    )
    if cost is None:
        return (fitted,)
    return (
        fitted,
        Observation(name=labelled(prefix, backend, SECONDS_SUFFIX), value=cost["seconds"]),
        Observation(name=labelled(prefix, backend, SPREAD_SUFFIX), value=cost["spread"]),
        Observation(name=labelled(prefix, backend, PEAK_SUFFIX), value=cost["peak_bytes"]),
    )


def benchmark_run_record(device: str) -> RunRecord:
    """Pin determinism, time every call under both backends, and record it.

    Args:
        device: Device to time on.

    Returns:
        The record: for every shape, the dispatcher's own choice and the
        pinned math backend, each with its timing, spread, peak allocation
        and whether it fitted at all.

    Raises:
        RuntimeError: Propagated when a call failed for a reason that is
            neither an out-of-memory nor a no-kernel refusal.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(device, probe_determinism(device))

    observations: list[Observation] = []
    for shape in _test_hooks.cost_shapes_hook():
        for key, backend in ((DEFAULT_KEY, None), (PINNED_KEY, BACKENDS[PINNED_KEY])):
            cost = time_sdpa(shape, device, backend)
            # Logged per point rather than only at the end: the sweep's last
            # shapes are the ones most likely to be killed by a wall clock,
            # and a job that died there would otherwise leave no record of
            # what it had already measured.
            _log.info(
                "cost %s b%d h%d s%d %s: %s",
                shape["name"],
                shape["batch"],
                shape["heads"],
                shape["sequence_len"],
                key,
                "DID NOT FIT"
                if cost is None
                else f"{cost['seconds'] * 1e3:.4f} ms/call "
                f"(spread {cost['spread'] * 1e3:.4f}) "
                f"peak {cost['peak_bytes'] / 2**20:.1f} MiB",
            )
            observations.extend(cost_observations(cost_prefix(shape), key, cost))

    return run_record(
        experiment=SDPA_COST_EXPERIMENT,
        label=SDPA_COST_LABEL,
        fingerprint=fingerprint,
        observations=tuple(observations),
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep once and write the record.

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

    record = benchmark_run_record(cli_args.require_flag(parsed, DEVICE_FLAG))

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "sdpa cost over %d calls %s -> %s",
        len(_test_hooks.cost_shapes_hook()),
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
        service_name="sdpa-benchmark",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "PINNED_KEY",
    "SDPA_COST_LABEL",
    "benchmark_run_record",
    "cost_observations",
    "entrypoint",
    "main",
]


# Without this, `python -m model_trainer.cli.sdpa_benchmark` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
