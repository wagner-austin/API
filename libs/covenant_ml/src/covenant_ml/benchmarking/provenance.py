"""The one way a ClearGBM benchmark says what it ran on.

WHAT THIS FIXES, MEASURED RATHER THAN SUSPECTED. On 2026-08-27 an inventory of
``libs/cleargbm/docs/BENCHMARK_MANIFEST_*.json`` found 41 manifests, of which
**four** carried an environment block -- ``platform``, ``python``, ``numpy``,
``sklearn``, ``lightgbm``, ``timestamp_utc`` -- all dated between 2026-07-20
and 2026-07-24. Every one of the 37 written since 2026-07-30 carried none of
it, and no code under ``scripts/`` or ``src/`` still emitted it. The practice
did not fail to start; it stopped.

Worse, and found in the same pass: of the six ``benchmark_cleargbm_*`` entry
points, exactly ONE pinned the BLAS thread count and built a fingerprint. The
other five did neither, so their numbers were not reproducible against
themselves, let alone comparable with each other.

WHY A SHARED BUILDER RATHER THAN A LINE IN EACH SCRIPT. Five of six scripts
already proved the counter-argument. A convention that has to be remembered in
each new entry point is a convention that decays; a function every script must
call is one the guard can check for.

WHAT GOES IN AND WHAT DOES NOT. :data:`BENCHMARK_DISTRIBUTIONS` names the
libraries whose arithmetic can move a boosting number. It is not every
installed distribution: a fingerprint over all of them differs whenever a
formatter is bumped, and a difference that cannot reach a split gain makes the
differences that can harder to see.

THE CPU MODEL IS STILL NOT RECORDED, and for a project whose headline is a
TIMING claim against LightGBM that is the sharpest remaining gap. See
:mod:`platform_core.environment_record` -- no stdlib call reports the model
portably, so the host axis separates operating system, architecture and core
count and nothing finer. A caller that knows its node type should inject a
sharper :class:`~platform_core.environment_record.HostProbe`.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.comparability import RunFingerprint, cpu_run_fingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.environment_record import (
    capture_host_record,
    capture_package_versions,
)

from covenant_ml.benchmarking import _test_hooks

#: The libraries whose arithmetic decides a gradient-boosting benchmark's
#: numbers, in the order the axis renders them.
#:
#: ``cleargbm`` and ``cleargbm_rs`` are the subject; ``lightgbm`` and
#: ``xgboost`` are the arms it is measured against, so a bump in either moves
#: the comparison without moving ClearGBM; ``numpy`` and ``scikit-learn``
#: back the data preparation and the metrics.
BENCHMARK_DISTRIBUTIONS: tuple[str, ...] = (
    "cleargbm",
    "cleargbm_rs",
    "lightgbm",
    "numpy",
    "scikit-learn",
    "xgboost",
)


def benchmark_fingerprint(
    determinism: DeterminismRecord, get_env: Callable[[str], str | None]
) -> RunFingerprint:
    """Describe the configuration a benchmark's numbers were produced under.

    Args:
        determinism: What the entry point pinned, from its own
            :func:`~platform_core.determinism_cpu.apply_cpu_determinism` call.
            Passed in rather than pinned here because pinning must happen
            before any native numeric library loads, which is above this
            module's own import.
        get_env: Reader for a process environment variable, for the image
            digest the launcher exports.

    Returns:
        The fingerprint, carrying the image, the pinned posture, the machine
        and the resolved versions of :data:`BENCHMARK_DISTRIBUTIONS`.

    Raises:
        PackageNotFoundError: When one of :data:`BENCHMARK_DISTRIBUTIONS` is
            not installed. Propagated rather than recorded as unknown: a
            benchmark comparing against a library the environment does not
            have is not the benchmark anyone meant to run.
    """
    return cpu_run_fingerprint(
        determinism,
        get_env,
        capture_host_record(_test_hooks.host_probe()),
        capture_package_versions(BENCHMARK_DISTRIBUTIONS, _test_hooks.installed_version),
    )


__all__ = ["BENCHMARK_DISTRIBUTIONS", "benchmark_fingerprint"]
