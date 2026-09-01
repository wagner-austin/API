"""Price the ordered kernel against the vendor, shape by shape.

The number the whole project exists to produce: what does owning the
reduction order COST at tiled-kernel speed, against the same vendor baseline
the experiment's posture allows (TF32 off, so cuBLAS FP32 on CUDA cores)?
The rank-one instrument answered "175x at a real vocabulary"; this measures
the production-shaped answer over the same declared timing table.

CUDA events, warmup then repeated timed batches, median reported -- the
same discipline the abl benchmarks use, in miniature.
"""

from __future__ import annotations

import pathlib
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Protocol

import torch
from model_trainer.cli import _test_hooks
from model_trainer.core.services.model.gemm_probe import gemm_operands
from model_trainer.core.services.model.gemm_shapes import GemmShape, gemm_label
from platform_core import cli_args
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging

from ordered_kernels.kernels import gemm

_log = get_logger(__name__)

DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (DEVICE_FLAG, OUT_FLAG)

#: Untimed calls before the clock starts, amortising compile and cache.
WARMUP_CALLS = 3

#: Timed repetitions; the median is reported.
TIMED_CALLS = 7


class _CudaEventProto(Protocol):
    """The three members of a timing event this benchmark touches.

    ``torch.cuda.Event`` is untyped in the stubs; the dynamic-import dance
    types the surface at the boundary, as everywhere in this experiment.
    """

    def record(self) -> None: ...
    def synchronize(self) -> None: ...
    def elapsed_time(self, end: _CudaEventProto) -> float: ...


class _EventCtorProto(Protocol):
    def __call__(self, *, enable_timing: bool) -> _CudaEventProto: ...


def _timing_event() -> _CudaEventProto:
    """Build one timing-enabled CUDA event, typed."""
    module = __import__("torch.cuda", fromlist=["Event"])
    ctor: _EventCtorProto = module.Event
    return ctor(enable_timing=True)


def _time_call(fn: Callable[[], torch.Tensor]) -> float:
    """Median milliseconds over the timed repetitions.

    Args:
        fn: The call to time.

    Returns:
        Median wall milliseconds, CUDA-event timed.
    """
    for _ in range(WARMUP_CALLS):
        fn()
    times: list[float] = []
    for _ in range(TIMED_CALLS):
        start = _timing_event()
        stop = _timing_event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def bench_shape(shape: GemmShape, device: str) -> tuple[float, float]:
    """Time the vendor call and the ordered kernel on one shape.

    Args:
        shape: The call to price.
        device: Device to run on.

    Returns:
        ``(vendor milliseconds, ordered milliseconds)``, medians.
    """
    bias, x, w = gemm_operands(shape, device)
    vendor = _time_call(lambda: torch.addmm(bias, x, w))
    ordered = _time_call(lambda: gemm(x, w, bias))
    return vendor, ordered


def bench_table(device: str) -> dict[str, dict[str, float]]:
    """Price every declared timed shape.

    Args:
        device: Device to run on.

    Returns:
        Per shape label: vendor ms, ordered ms, and their ratio.
    """
    results: dict[str, dict[str, float]] = {}
    for name, shape in _test_hooks.benchmark_shapes():
        vendor, ordered = bench_shape(shape, device)
        # A zero vendor median is a broken clock, and dividing by it raising
        # loudly is the right report.
        ratio = ordered / vendor
        label = gemm_label(name, shape, "ms")
        results[label] = {"vendor_ms": vendor, "ordered_ms": ordered, "ratio": ratio}
        _log.info("%s vendor=%.3fms ordered=%.3fms ratio=%.2fx", label, vendor, ordered, ratio)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Price the whole table and write the results.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the results are written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            absent -- resolved before anything computes.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    results = bench_table(device)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(results), encoding="utf-8")
    ratios = sorted(row["ratio"] for row in results.values())
    _log.info(
        "%d shapes priced, ratio min/median/max %.2fx/%.2fx/%.2fx -> %s",
        len(ratios),
        ratios[0],
        ratios[len(ratios) // 2],
        ratios[-1],
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
        service_name="ordered-bench",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["TIMED_CALLS", "WARMUP_CALLS", "bench_shape", "bench_table", "entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
