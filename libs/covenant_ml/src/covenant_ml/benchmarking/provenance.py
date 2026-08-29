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
from hashlib import sha256

from platform_core.comparability import RunFingerprint, cpu_run_fingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.environment_record import (
    capture_host_record,
    capture_package_versions,
)
from platform_core.json_utils import dump_json_str
from platform_core.run_record import Observation, RunRecord, run_record

from covenant_ml.benchmarking import _test_hooks
from covenant_ml.benchmarking.reporting import summarize_every_model, summarize_gap
from covenant_ml.benchmarking.types import BenchmarkManifest
from covenant_ml.benchmarking.types_codec import encode_seed_result

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


BENCHMARK_EXPERIMENT = "cleargbm-vs-lightgbm-fit-time"
"""What this benchmark IS, for pairing two of its records.

Stable across invocations and independent of the dataset or the seeds, both
of which vary between runs that are still the same experiment and both of
which belong in the label instead.
"""


def benchmark_observations(manifest: BenchmarkManifest) -> tuple[Observation, ...]:
    """Name every number this benchmark exists to produce.

    Args:
        manifest: The completed benchmark.

    Returns:
        One observation per headline number: each arm's mean and spread of
        fit time, its tree size, and its two quality metrics, plus the three
        ratios that are the comparison itself. Per-seed records are NOT here
        -- they are the payload, and the digest covers them.

        Arm-scoped names are prefixed with the arm, because ``mean_fit_s``
        alone would pair ClearGBM's number with LightGBM's in any contrast
        that read two records side by side.
    """
    gap = summarize_gap(manifest)
    observations: list[Observation] = [
        Observation(name="raw_ratio", value=gap.raw_ratio),
        Observation(name="leaf_ratio", value=gap.leaf_ratio),
        Observation(name="normalized_ratio", value=gap.normalized_ratio),
    ]
    for summary in summarize_every_model(manifest):
        observations.extend(
            [
                Observation(name=f"{summary.model}.mean_fit_s", value=summary.mean_fit_s),
                Observation(name=f"{summary.model}.stdev_fit_s", value=summary.stdev_fit_s),
                Observation(name=f"{summary.model}.mean_leaves", value=summary.mean_leaves),
                Observation(name=f"{summary.model}.mean_auc_roc", value=summary.mean_auc_roc),
                Observation(name=f"{summary.model}.mean_auc_pr", value=summary.mean_auc_pr),
            ]
        )
    return tuple(observations)


def benchmark_label(manifest: BenchmarkManifest) -> str:
    """Name which run within the experiment this is.

    Args:
        manifest: The completed benchmark.

    Returns:
        The dataset, the timing estimator and the seed count, joined. Those
        are the three things that differ between two runs of this benchmark
        that are still meant to be compared -- a different dataset is a
        different number, a median is not a minimum, and three seeds is not
        thirty. The dataset appears by its content digest rather than a
        filename, because a filename is not the bytes.
    """
    return (
        f"{manifest['dataset']['sha256'][:12]}"
        f"-{manifest['estimator']}"
        f"-{len(manifest['seeds'])}seeds"
    )


def benchmark_run_record(manifest: BenchmarkManifest) -> RunRecord:
    """Turn a completed benchmark into the shape every experiment emits.

    THE MANIFEST IS NOT THIS SHAPE, and that was the remaining gap after the
    fingerprint was added. A manifest says everything about one benchmark and
    nothing in a form another experiment's records can be read beside; the
    record says the few numbers someone will subtract, in the vocabulary
    :mod:`platform_core.run_record` already checks comparability in. Both are
    written, because neither contains the other: the manifest holds the
    per-seed detail, the record holds the claim.

    Args:
        manifest: The completed benchmark, carrying the fingerprint its entry
            point built before any numeric library loaded.

    Returns:
        The record, its observations in canonical order.

    Raises:
        ValueError: If two observations collide on a name, which would mean
            two arms share one, and the manifest's own decoder forbids that.
    """
    return run_record(
        experiment=BENCHMARK_EXPERIMENT,
        label=benchmark_label(manifest),
        fingerprint=manifest["fingerprint"],
        observations=benchmark_observations(manifest),
        # The per-seed records are the payload: 2 arms x N seeds of timing and
        # quality detail that no cross-experiment layer should have to read to
        # tell two runs apart. Digested over the canonical encoding rather
        # than the file, so a record built in memory carries the same digest
        # as one built from a manifest read back off disk.
        payload_digest=sha256(
            dump_json_str([encode_seed_result(result) for result in manifest["results"]]).encode(
                "utf-8"
            )
        ).hexdigest(),
    )


__all__ = [
    "BENCHMARK_DISTRIBUTIONS",
    "BENCHMARK_EXPERIMENT",
    "benchmark_fingerprint",
    "benchmark_label",
    "benchmark_observations",
    "benchmark_run_record",
]
