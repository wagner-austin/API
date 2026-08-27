"""Time an attention call under one backend, and say what it cost in memory.

The price of the correctness result in :mod:`sdpa_probe`: pinning
``SDPBackend.MATH`` makes attention bit-identical across a V100, an A30 and
an A100, and this measures what that costs.

WHY MEMORY IS MEASURED AND NOT ONLY TIME. The math path materialises the full
``[batch, heads, seq, seq]`` score matrix; the fused kernel does not. That is
a QUALITATIVE difference, not a constant factor, and a table of seconds would
report it as "slower" right up to the point where it stops fitting on the
card at all. Peak allocation is recorded beside every timing, and an
out-of-memory is recorded as a result rather than raised -- it is the
strongest cost statement available.

WHY THE TIMING CONSTANTS COME FROM :mod:`gemm_timing`. Same cluster, same
question, same reasons: a CUDA launch is asynchronous so every measurement
brackets a synchronise; calls are issued in batches so launch overhead does
not swamp the small shapes; and the median of several batches is reported
because a shared node schedules other work. Restating them here would let two
benchmarks on one page drift into being measured differently.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch
from torch.nn.attention import SDPBackend
from typing_extensions import TypedDict

from model_trainer.core.services.model.gemm_timing import (
    BATCHES,
    INNER,
    WARMUP,
    synchroniser,
)
from model_trainer.core.services.model.sdpa_probe import forced_sdpa_output, sdpa_output
from model_trainer.core.services.model.sdpa_shapes import SDPA_SEED, SdpaCostShape

#: What a cpu run reports for peak device memory. Zero, because the allocator
#: being measured is CUDA's and a cpu run does not use it -- recorded rather
#: than omitted so a cpu record has the same shape as a cuda one.
NO_PEAK = 0.0


class SdpaCost(TypedDict):
    """What one attention call cost under one backend.

    Attributes:
        seconds: Median seconds per call.
        spread: Slowest batch minus fastest, in seconds per call.
        peak_bytes: Peak CUDA bytes allocated across the timed run.
    """

    seconds: float
    spread: float
    peak_bytes: float


def _no_peak_to_reset() -> None:
    """Reset the peak counter of a device that keeps none."""


def _no_peak() -> float:
    """Report the peak allocation of a device that keeps none.

    Returns:
        :data:`NO_PEAK`.
    """
    return NO_PEAK


def peak_resetter(device: str) -> Callable[[], None]:
    """Choose how to clear a device's peak-allocation counter.

    Returns the callable rather than calling it, for the reason
    :func:`~gemm_timing.synchroniser` does: both arms are then reachable
    without a GPU, because a test can assert the cuda arm by identity.

    Args:
        device: The device about to be timed.

    Returns:
        A callable that resets the peak counter.
    """
    if device == "cpu":
        return _no_peak_to_reset
    return torch.cuda.reset_peak_memory_stats


def peak_reader(device: str) -> Callable[[], float]:
    """Choose how to read a device's peak allocation.

    Args:
        device: The device that was timed.

    Returns:
        A callable returning peak bytes allocated since the last reset.
    """
    if device == "cpu":
        return _no_peak
    return torch.cuda.max_memory_allocated


def cost_operands(
    shape: SdpaCostShape, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one timed call's query, key and value.

    Laid out the way ``_split_heads`` lays them out -- a view then a permute,
    leaving them non-contiguous -- because that is what the model hands the
    dispatcher and strides reach kernel selection. Generated on the device
    rather than on the CPU and moved, unlike
    :func:`~sdpa_probe.sdpa_operands`: this measures TIME, the values do not
    enter the answer, and a 4096-token batch would otherwise be built in host
    memory and copied for nothing.

    Args:
        shape: The call to build for.
        device: Where to build.

    Returns:
        ``(query, key, value)``, each ``[batch, heads, sequence_len,
        head_dim]``.
    """
    torch.manual_seed(SDPA_SEED)
    flat = [
        torch.randn(
            shape["batch"],
            shape["sequence_len"],
            shape["heads"] * shape["head_dim"],
            dtype=torch.float32,
            device=device,
        )
        for _ in range(3)
    ]
    return (
        flat[0]
        .view(shape["batch"], shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
        flat[1]
        .view(shape["batch"], shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
        flat[2]
        .view(shape["batch"], shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
    )


def _run_once(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, backend: SDPBackend | None
) -> None:
    """Issue one attention call, forced or not, discarding its output.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        backend: The backend to force, or None to let the dispatcher choose.
    """
    if backend is None:
        sdpa_output(query, key, value)
    else:
        forced_sdpa_output(query, key, value, backend)


def timed_or_unfitted(run: Callable[[], SdpaCost]) -> SdpaCost | None:
    """Run a measurement, reporting an out-of-memory as a result.

    Separated from :func:`measure_sdpa` so both arms are reachable: a card
    large enough to run the suite would never take the second one, and an arm
    no test can reach is an arm nobody has checked says what it means. A test
    supplies a callable that raises the real exception; nothing is faked.

    Args:
        run: The measurement to attempt.

    Returns:
        Its result, or None when the device ran out of memory. "This
        configuration does not fit" is the strongest cost statement available
        and is a measurement, not a failure of one.
    """
    try:
        return run()
    except torch.cuda.OutOfMemoryError:
        # The sweep continues to the next shape, so the allocator is handed
        # its blocks back rather than left holding them for a configuration
        # that has already been abandoned.
        torch.cuda.empty_cache()
        return None


def measure_sdpa(shape: SdpaCostShape, device: str, backend: SDPBackend | None) -> SdpaCost:
    """Time one attention call and read its peak allocation.

    Args:
        shape: The call to time.
        device: Device to run it on.
        backend: The backend to force, or None for the dispatcher's choice.

    Returns:
        The cost.

    Raises:
        torch.cuda.OutOfMemoryError: When the device cannot hold the call.
            Caught by :func:`timed_or_unfitted`, which is the only caller.
        RuntimeError: Propagated from
            :func:`~sdpa_probe.forced_sdpa_output` for a failure that is
            neither an out-of-memory nor a no-kernel refusal.
    """
    wait = synchroniser(device)
    reset_peak = peak_resetter(device)
    read_peak = peak_reader(device)

    query, key, value = cost_operands(shape, device)
    reset_peak()
    for _ in range(WARMUP):
        _run_once(query, key, value, backend)
    wait()

    per_call: list[float] = []
    for _ in range(BATCHES):
        start = time.perf_counter()
        for _ in range(INNER):
            _run_once(query, key, value, backend)
        wait()
        per_call.append((time.perf_counter() - start) / INNER)

    return SdpaCost(
        seconds=statistics.median(per_call),
        spread=max(per_call) - min(per_call),
        peak_bytes=float(read_peak()),
    )


def time_sdpa(shape: SdpaCostShape, device: str, backend: SDPBackend | None) -> SdpaCost | None:
    """Measure seconds per call and peak memory for one attention call.

    Args:
        shape: The call to time.
        device: Device to run it on.
        backend: The backend to force, or None for the dispatcher's choice.

    Returns:
        The cost, or None when the call did not fit in device memory.

    Raises:
        RuntimeError: Propagated for a failure that is neither an
            out-of-memory nor a no-kernel refusal.
    """

    def run() -> SdpaCost:
        return measure_sdpa(shape, device, backend)

    return timed_or_unfitted(run)


__all__ = [
    "NO_PEAK",
    "SdpaCost",
    "cost_operands",
    "measure_sdpa",
    "peak_reader",
    "peak_resetter",
    "time_sdpa",
    "timed_or_unfitted",
]
