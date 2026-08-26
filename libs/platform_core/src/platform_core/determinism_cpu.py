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

import sys
from collections.abc import Container

from platform_core.determinism_env import (
    BLAS_THREAD_ENV_VARS,
    SetEnvProtocol,
)
from platform_core.determinism_record import DeterminismRecord, determinism_record

#: The stack this module pins. A record carries it so a numpy run and a torch
#: run never compare equal merely because neither mentioned the other's
#: settings.
CPU_STACK = "cpu"

#: Modules whose import loads a native numeric library and fixes its thread
#: pool. Every one of these pulls a BLAS; once any is in ``sys.modules`` the
#: thread-count variables have already been read and writing them changes
#: nothing.
#:
#: ``scipy`` and ``sklearn`` are listed beside ``numpy`` because importing
#: either imports numpy transitively, and a caller reading this list should
#: not have to know that.
NUMERIC_MODULES: tuple[str, ...] = ("numpy", "scipy", "sklearn", "torch", "pandas")


class NativeLibrariesAlreadyLoadedError(RuntimeError):
    """A CPU determinism pin arrived after the libraries it must precede.

    Raised rather than returned, and raised rather than ignored, because the
    alternative is a record that lies. :func:`apply_cpu_determinism` reports
    every variable it sets as the run's posture; if the write cannot take
    effect, that report asserts a configuration the run does not have, and a
    manifest carrying it is worse than a manifest carrying nothing -- an
    absent posture stops a comparison, a false one licenses it.

    Measured 2026-08-26, numpy on scipy-openblas, a fixed 2048x2048 float32
    matmul, digests of the exact result bytes:

        pin to 1 BEFORE importing numpy   f364ecedb70f678b
        pin to 8 BEFORE importing numpy   628f2231d6fe0a62
        import numpy, THEN pin to 1       20d850081f69206f

    The late pin produced a third answer, reproducibly. It did not fail
    loudly and it did not achieve what the early pin achieves.
    """


def _loaded_numeric_modules(modules: Container[str]) -> tuple[str, ...]:
    """Report which native numeric libraries are already imported.

    Args:
        modules: The importer's module table, normally ``sys.modules``.
            Injected so a test can state the condition rather than have to
            manufacture it by importing.

    Returns:
        The names from :data:`NUMERIC_MODULES` already present, in declared
        order so the message is stable.
    """
    return tuple(name for name in NUMERIC_MODULES if name in modules)


def apply_cpu_determinism(
    set_env: SetEnvProtocol, threads: str, modules: Container[str] | None = None
) -> DeterminismRecord:
    """Pin CPU reduction order by fixing the thread count, and report it.

    Must be called before the native numeric libraries load, which happens on
    first import of numpy or anything built on it. That requirement USED TO
    LIVE ONLY IN THIS DOCSTRING, and on 2026-08-26 the sentence above sat
    directly over a caller that violated it: `benchmark_cleargbm_regression`
    imported numpy at module scope and pinned from `main`. Every gate passed
    -- mypy, ruff, the guards, 2,564 tests at 100% branches -- and the
    manifest it wrote asserted `OMP_NUM_THREADS=1` for a run that was
    multi-threaded. A docstring cannot fail a build, so the requirement is
    now executed instead of described.

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
        modules: The importer's module table, defaulting to ``sys.modules``.
            Injected so a test can state that the natives are loaded without
            importing them into the worker it shares with other tests.

    Returns:
        A record naming :const:`CPU_STACK` and every variable that was set.

    Raises:
        NativeLibrariesAlreadyLoadedError: When a native numeric library is
            already imported, so the write cannot take effect. Refused rather
            than performed, because performing it returns a record that
            claims a posture the run does not have -- and a manifest carrying
            a false posture is worse than one carrying none. The message
            names the modules found and what to do, since the fix is always
            the same: pin at the process entry, above the first numeric
            import.
    """
    loaded = _loaded_numeric_modules(sys.modules if modules is None else modules)
    if loaded:
        raise NativeLibrariesAlreadyLoadedError(
            f"cannot pin CPU determinism: {', '.join(loaded)} already imported, so "
            f"{', '.join(BLAS_THREAD_ENV_VARS)} have been read and writing them now "
            "changes nothing. Pin at the process entry point, above the first import "
            "that pulls a numeric library. Recording this pin as the run's posture "
            "would assert a configuration the run does not have."
        )
    for name in BLAS_THREAD_ENV_VARS:
        set_env(name, threads)
    return determinism_record(CPU_STACK, dict.fromkeys(BLAS_THREAD_ENV_VARS, threads))


__all__ = [
    "BLAS_THREAD_ENV_VARS",
    "CPU_STACK",
    "NUMERIC_MODULES",
    "NativeLibrariesAlreadyLoadedError",
    "apply_cpu_determinism",
]
