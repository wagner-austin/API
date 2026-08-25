"""Pinning determinism for a CPU numeric stack, and reporting what was pinned.

The counterpart to :mod:`platform_ml.determinism`, for the research that is
not torch. Most of this monorepo's is not: gradient boosting, transliteration
and metabolomics pull no torch at all, and a future job may have no GPU to
pin anything on.

WHAT THIS PINS, STATED AS MEASURED RATHER THAN AS ASSUMED. A multi-threaded
BLAS splits a reduction across threads, and the thread COUNT decides the
partitioning. Floating-point addition is not associative, so two runs that
split the same sum differently reach different bits.

Measured on 2026-08-25, numpy 2.3.5 on scipy-openblas 0.3.30, 24 cores, a
4096x4096 float32 matmul over identical bytes, fresh process each time:

* At a FIXED thread count the result is bit-identical, run after run. 1, 8
  and 24 threads each reproduced themselves exactly.
* ACROSS thread counts the results differ. 1, 8 and 24 threads produced three
  different answers -- 865,498 of 16,777,216 elements changed, max absolute
  difference 1.4e-4 on values averaging 51.

So the hazard is not that a threaded BLAS is unpredictable; on this stack it
is quite predictable. The hazard is that the thread count is an INPUT nobody
records. A run on a 24-core box and a run on an 8-core box, or one that set
the variable and one that inherited a default, produce different numbers from
the same code and the same data -- and nothing in either result says why.
Pinning it makes the count part of the configuration a fingerprint carries.

That is the same shape as the cuBLAS workspace problem, and seeding does not
touch either.

WHAT IT DOES NOT DO, deliberately:

* It does not seed anything. Seeding is the caller's and is orthogonal -- a
  seeded run with a racing reduction and an unseeded deterministic one are
  both irreproducible, for different reasons.
* It does not promise reproducibility across different CPUs. Different
  microarchitectures select different kernels, and a wheel built for AVX-512
  does not compute the same partial sums as one built for AVX2. What this
  buys is reproducibility WITHIN one configuration, which is what makes a
  rerun comparable to its own earlier self and a cross-configuration
  difference something that can be MEASURED rather than assumed to be zero.
* It does not choose for you. Serial reductions cost throughput, and the
  caller passes the thread count so the trade is made where someone knows
  the run's budget.
"""

from __future__ import annotations

from platform_core.determinism_env import (
    BLAS_THREAD_ENV_VARS,
    SetEnvProtocol,
)
from platform_core.determinism_record import DeterminismRecord, determinism_record

#: The stack this module pins. A record carries it so a numpy run and a torch
#: run never compare equal merely because neither mentioned the other's
#: settings.
CPU_STACK = "cpu"


def apply_cpu_determinism(set_env: SetEnvProtocol, threads: str) -> DeterminismRecord:
    """Pin CPU reduction order by fixing the thread count, and report it.

    Must be called before the native numeric libraries load, which happens on
    first import of numpy or anything built on it. Setting these afterwards
    is accepted without error and has no effect -- exactly the cuBLAS hazard,
    which is why this returns the values it wrote: a caller that records the
    return records what the run actually had.

    Every variable in :data:`BLAS_THREAD_ENV_VARS` is written, not only the
    one this wheel happens to read, because which BLAS a numpy build links
    against is not knowable here and a partially pinned stack reports a
    posture it does not have.

    Args:
        set_env: Writer for a process environment variable.
        threads: Thread count to pin, as a string because that is what an
            environment variable holds. :data:`SINGLE_THREAD` is the value
            that makes reductions order-stable; a larger one is a deliberate
            throughput choice and is recorded as such rather than silently
            treated as deterministic.

    Returns:
        A record naming :const:`CPU_STACK` and every variable that was set.
    """
    for name in BLAS_THREAD_ENV_VARS:
        set_env(name, threads)
    return determinism_record(CPU_STACK, dict.fromkeys(BLAS_THREAD_ENV_VARS, threads))


__all__ = [
    "CPU_STACK",
    "apply_cpu_determinism",
]
