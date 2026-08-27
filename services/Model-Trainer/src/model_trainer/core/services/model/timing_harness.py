"""Timing and memory measurement shared by every benchmark here.

Extracted from :mod:`sdpa_timing` when the end-to-end forward benchmark
needed the same machinery. That extraction is not tidiness: the whole point
of the forward measurement is to be compared against the per-call one --
"attention costs 4-7x per call, so what does a forward pass cost" -- and two
harnesses that drifted apart would make the comparison meaningless. One
harness, two callers.

WHY A CALLABLE RATHER THAN A SHAPE. The per-call benchmark times one
attention call; the end-to-end one times a whole GPT-2 forward pass. What
they share is the protocol -- warm up, then several batches of calls between
synchronises, take the median -- and nothing else, so the protocol takes the
work as an argument.

WHY THE BATCHING CONSTANTS ARE PER CALLER. A launch is ~10 microseconds and
an attention call at 64 tokens is ~20, so that benchmark issues twenty calls
between synchronises to amortise it. A forward pass is milliseconds to
seconds and has already amortised its own launches; issuing twenty would just
multiply the wall clock. Each caller declares its own and says why.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing_extensions import TypedDict

#: What a cpu run reports for peak device memory. Zero, because the allocator
#: being measured is CUDA's and a cpu run does not use it -- recorded rather
#: than omitted so a cpu record has the same shape as a cuda one.
NO_PEAK = 0.0


class MeasuredCost(TypedDict):
    """What one piece of work cost.

    Attributes:
        seconds: Median seconds per call.
        spread: Slowest batch minus fastest, in seconds per call. Carried
            beside the median because a median with an enormous spread is a
            number that must not be compared with another one, and only the
            caller can see both.
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


class _Unrestricted(AbstractContextManager[None]):
    """A context that restricts nothing.

    Written out rather than using :func:`contextlib.nullcontext`, whose type
    this package's mypy settings resolve to Any. It exists so the UNFORCED
    arm of a benchmark is wrapped in a context too -- see
    :func:`backend_context` for the 20% bias that asymmetry caused.
    """

    def __enter__(self) -> None:
        """Enter, restricting nothing."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave, having restricted nothing and swallowing nothing."""


def backend_context(backend: SDPBackend | None) -> AbstractContextManager[None]:
    """Restrict the attention dispatcher for the duration of a whole run.

    ENTERED ONCE, NOT PER CALL, AND THAT IS A CORRECTNESS FIX RATHER THAN A
    TIDINESS ONE. An earlier revision forced the backend inside the timing
    loop, so the pinned arm paid a context enter and exit on every call and
    the unforced arm paid nothing. Measured on an RTX 3090 Ti, 2026-08-27:
    27.8 us per call, which is **20% of the whole measurement** at batch 1
    and 64 tokens, -1.5% at batch 8 and 512, and 0.1% at batch 8 and 2048.
    That bias sat exactly where the ladder's rungs are, and it would have
    been published as the cost of the kernel.

    The unforced arm enters a null context, so both arms are wrapped
    identically and the only difference between them is which kernels the
    dispatcher may choose.

    Args:
        backend: The backend to force, or None to let the dispatcher choose.

    Returns:
        A context manager restricting the dispatcher, or one that does
        nothing.
    """
    if backend is None:
        return _Unrestricted()
    # Bound to a typed name before returning: `sdpa_kernel` gives back a
    # context manager whose type parameter is Any, which this package forbids.
    restricted: AbstractContextManager[None] = sdpa_kernel([backend])
    return restricted


def timed_or_unfitted(run: Callable[[], MeasuredCost]) -> MeasuredCost | None:
    """Run a measurement, reporting an out-of-memory as a result.

    Separated from the measurement itself so both arms are reachable: a card
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


def time_calls(
    run: Callable[[], None],
    wait: Callable[[], None],
    device: str,
    warmup: int,
    inner: int,
    batches: int,
) -> MeasuredCost:
    """Time repeated calls and read the peak allocation they reached.

    Args:
        run: The work to time. Called ``warmup + inner * batches`` times.
        wait: Blocks until the device is idle; see
            :func:`~gemm_timing.synchroniser`.
        device: Device being timed, for the peak counters.
        warmup: Calls issued and discarded first. The first call on a shape
            pays for kernel selection and lazy module loading, which is a
            one-time cost and not what a per-call time should carry.
        inner: Calls per timed batch.
        batches: Timed batches. The median of these is reported, because a
            shared node schedules other work and a GPU clocks up and down:
            the mean chases those, the median does not.

    Returns:
        The cost.

    Raises:
        torch.cuda.OutOfMemoryError: When the device cannot hold the work.
            Caught by :func:`timed_or_unfitted`.
    """
    reset_peak = peak_resetter(device)
    read_peak = peak_reader(device)

    reset_peak()
    for _ in range(warmup):
        run()
    wait()

    per_call: list[float] = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(inner):
            run()
        wait()
        per_call.append((time.perf_counter() - start) / inner)

    return MeasuredCost(
        seconds=statistics.median(per_call),
        spread=max(per_call) - min(per_call),
        peak_bytes=float(read_peak()),
    )


__all__ = [
    "NO_PEAK",
    "MeasuredCost",
    "backend_context",
    "peak_reader",
    "peak_resetter",
    "time_calls",
    "timed_or_unfitted",
]
