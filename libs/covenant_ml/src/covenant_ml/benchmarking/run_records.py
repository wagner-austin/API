"""A benchmark's numbers in the shape the rest of the workspace reads.

``BenchmarkManifest`` is not deleted here, and the reason matters. It is not
merely a competing envelope for the same content: it carries a per-seed,
per-model structure -- each arm's timing distribution, its quality metrics,
the rotating position that keeps one arm out of the cold-CPU slot -- that
``RunRecord``'s flat ``name -> float`` observations cannot hold. Replacing it
would lose evidence to gain comparability, which is a bad trade.

What was actually missing is that nothing could place a ClearGBM benchmark
beside another experiment's numbers. ``compare_run_records`` and
``agree_across_runs`` take ``RunRecord`` and nothing else, so a manifest, however
complete, was unreadable to them.

So this emits a ``RunRecord`` SIDECAR beside the manifest, which is the
pattern the rest of the workspace already uses: the LSTM writes one beside its
results CSV, Model-Trainer writes one beside a saved model, and code-style-eval
writes one beside its comparison. The domain artifact keeps the detail; the
sidecar makes the headline figures comparable. The manifest's own digest ties
the two together, so a sidecar can never quietly describe a different run than
the manifest it sits next to.

WHICH FIGURES BECOME OBSERVATIONS. The aggregates a later contrast would
actually read: each arm's mean fit time, its spread, its mean leaves, its two
AUCs, and the three ratios that are this benchmark's headline. Per-seed rows
stay in the manifest, because an observation named ``fit_s`` appearing eleven
times would make the pairing ambiguous, which :func:`run_record` refuses
outright.
"""

from __future__ import annotations

import hashlib

from platform_core.json_utils import dump_json_str
from platform_core.run_record import Observation, RunRecord, run_record

from .reporting import summarize_every_model, summarize_gap
from .types import BenchmarkManifest
from .types_codec import encode_benchmark_manifest

#: Name under which ClearGBM benchmark numbers are comparable with each other.
BENCHMARK_EXPERIMENT = "cleargbm-benchmark"


def manifest_digest(manifest: BenchmarkManifest) -> str:
    """Digest the manifest a record summarises.

    Encoded through the manifest's own codec, which builds its keys in a
    fixed order, so the same manifest digests the same. The digest is
    therefore stable across processes but NOT across a change to the codec's
    key order; it answers "are these two manifests the same content" for one
    schema version, which is what the sidecar needs it for, and it is not a
    content address that survives a schema change.

    Args:
        manifest: The finished manifest.

    Returns:
        A hex digest.
    """
    encoded = dump_json_str(encode_benchmark_manifest(manifest), compact=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def benchmark_observations(manifest: BenchmarkManifest) -> tuple[Observation, ...]:
    """Name the aggregate figures a later contrast would read.

    Every name is prefixed by its arm, so two models' means never collide.
    ``seeds`` travels with them because a mean over eleven seeds and a mean
    over one are not the same evidence, and a record carrying only the mean
    cannot tell them apart.

    The three ratios are included even though they are derivable, because
    ``normalized_ratio`` is this benchmark's actual claim -- per-leaf cost
    once tree size is held constant -- and a reader comparing two runs should
    not have to recompute the headline to see whether it moved.

    Args:
        manifest: The finished manifest.

    Returns:
        The observations, in any order; the record sorts them.

    Raises:
        ValueError: Propagated from :func:`summarize_gap` when the manifest
            holds no record for either model.
    """
    observations: list[Observation] = [
        Observation(name="seeds", value=float(len(manifest["seeds"]))),
    ]
    for summary in summarize_every_model(manifest):
        arm = summary.model
        observations.extend(
            (
                Observation(name=f"{arm}.mean_fit_s", value=summary.mean_fit_s),
                Observation(name=f"{arm}.stdev_fit_s", value=summary.stdev_fit_s),
                Observation(name=f"{arm}.mean_leaves", value=summary.mean_leaves),
                Observation(name=f"{arm}.mean_auc_roc", value=summary.mean_auc_roc),
                Observation(name=f"{arm}.mean_auc_pr", value=summary.mean_auc_pr),
            )
        )
    gap = summarize_gap(manifest)
    observations.extend(
        (
            Observation(name="raw_ratio", value=gap.raw_ratio),
            Observation(name="leaf_ratio", value=gap.leaf_ratio),
            Observation(name="normalized_ratio", value=gap.normalized_ratio),
        )
    )
    return tuple(observations)


def benchmark_run_record(manifest: BenchmarkManifest, label: str) -> RunRecord:
    """Build the record that belongs beside a benchmark manifest.

    The manifest's own fingerprint is carried through rather than recaptured.
    Recapturing would describe the machine writing the record instead of the
    machine that produced the timings, and for a benchmark whose headline is
    a fit time that is the axis that moves it most.

    Args:
        manifest: The finished manifest.
        label: Which run this was, e.g. the dataset and estimator it used.

    Returns:
        The record.

    Raises:
        ValueError: Propagated from :func:`run_record` when the label is
            empty, and from :func:`benchmark_observations` when the manifest
            holds no record for either model.
    """
    return run_record(
        experiment=BENCHMARK_EXPERIMENT,
        label=label,
        fingerprint=manifest["fingerprint"],
        observations=benchmark_observations(manifest),
        payload_digest=manifest_digest(manifest),
    )


__all__ = [
    "BENCHMARK_EXPERIMENT",
    "benchmark_observations",
    "benchmark_run_record",
    "manifest_digest",
]
