"""Time one GEMM, on the device, honestly.

WHY THIS IS NOT ``time.perf_counter()`` AROUND A CALL. A CUDA launch is
asynchronous: the call returns as soon as the work is queued, so timing it
that way measures the launch and not the arithmetic. Every measurement here
brackets a synchronise.

WHY A BATCH RATHER THAN ONE CALL PER MEASUREMENT. The shapes span M128/K128 to
M1600/K6400, and at the small end a single call is dominated by launch
overhead -- which is real cost, but it is the same under both conditions and
would swamp the kernel difference this exists to measure. Issuing
:data:`INNER` calls between two synchronises amortises it.

WHY THE MEDIAN OF SEVERAL BATCHES. A shared cluster node schedules other work,
and a GPU clocks up and down. The mean chases those; the median does not.
The spread is reported beside it so a reader can see when it mattered.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from model_trainer.core.services.model.gemm_probe import gemm_operands
from model_trainer.core.services.model.gemm_shapes import GemmShape

#: Calls issued before any measurement, discarded. The first call on a shape
#: pays for kernel selection and any lazy module loading, which is a one-time
#: cost and not what a per-call time should carry.
WARMUP = 3

#: Calls per timed batch, to amortise launch overhead at the small shapes.
INNER = 20

#: Timed batches. The median of these is reported.
BATCHES = 7


def _already_done() -> None:
    """Wait for a CPU, which has already finished."""


def synchroniser(device: str) -> Callable[[], None]:
    """Choose how to wait for a device to finish its queued work.

    Returns the waiter rather than doing the waiting, so both arms are
    reachable without a GPU: a test can assert that a cuda device selects
    ``torch.cuda.synchronize`` by identity, which is the same trick the
    monorepo uses for ``process.exit``. A branch inside the timing loop would
    have left the cuda arm uncovered on every machine that runs the suite --
    which is every machine except the cluster.

    Args:
        device: The device about to be timed.

    Returns:
        A callable that blocks until that device is idle. A CPU run has
        nothing to wait for: its work is done when the call returns.
    """
    if device == "cpu":
        return _already_done
    return torch.cuda.synchronize


def time_gemm(shape: GemmShape, device: str) -> tuple[float, float]:
    """Measure seconds per call for one GEMM.

    Args:
        shape: The call to time.
        device: Device to run it on.

    Returns:
        ``(median seconds per call, spread)`` where the spread is the
        difference between the slowest and fastest batch, in seconds per
        call. The spread is returned rather than discarded because a median
        with an enormous spread is a number that should not be compared with
        another one, and only the caller can see both.
    """
    bias, x, w = gemm_operands(shape, device)
    wait = synchroniser(device)

    for _ in range(WARMUP):
        torch.addmm(bias, x, w)
    wait()

    per_call: list[float] = []
    for _ in range(BATCHES):
        start = time.perf_counter()
        for _ in range(INNER):
            torch.addmm(bias, x, w)
        wait()
        per_call.append((time.perf_counter() - start) / INNER)

    return statistics.median(per_call), max(per_call) - min(per_call)


__all__ = ["BATCHES", "INNER", "WARMUP", "synchroniser", "time_gemm"]
