"""What produced a comparison, recorded beside the comparison itself.

A ``comparison.json`` carries pass rates and p-values and nothing that says
which models, which decoding parameters, or which machine produced them. Read
six months later it is a set of numbers with no way back to the run, and the
first question anyone asks of a result -- what was it measured on -- has no
answer on the page that states it.

This module emits the workspace's ONE record shape for that,
:class:`platform_core.run_record.RunRecord`, rather than a second one local to
this package. A private shape is what stops a number here being read beside a
number from another experiment.

The generated files and the per-arm generation manifests are covered by the
record's payload digest rather than copied into it. That is what carries the
generation configuration: the manifests record what each arm did, the digest
pins their bytes, and two comparisons whose digests agree were computed over
identical inputs including identical decoding settings.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.environment_record import (
    capture_host_record,
    capture_package_versions,
    installed_version,
    stdlib_host_probe,
)
from platform_core.run_record import Observation, RunRecord, run_record

from code_style_eval.contracts.outcomes import ComparisonReport

#: The libraries whose behaviour decides a guard-pass number, and the ONE
#: place that set is written.
#:
#: These are the checkers themselves, because a guard-pass rate is their
#: verdict: a ruff release that adds a rule moves the number without anything
#: about the models changing. ``platform-core`` is included because it decodes
#: the outcome records the comparison is computed from.
#:
#: The set is deliberately narrow. Recording every installed distribution
#: would make two runs differ over a dev-dependency bump that cannot reach a
#: verdict, and every spurious difference makes a real one harder to see.
SCORING_DISTRIBUTIONS: tuple[str, ...] = ("ruff", "mypy", "platform-core")

#: Name under which these numbers are comparable with each other.
EXPERIMENT = "code-style-guard-pass"


def _path_name(path: Path) -> str:
    """Return a path's final component, as a sort key.

    A named function rather than a lambda because the strict typing here
    rejects the untyped expression a lambda produces. Mirrors
    ``platform_core.run_record._observation_name``.

    Args:
        path: The path to key.

    Returns:
        The final component.
    """
    return path.name


def payload_digest(paths: Sequence[Path]) -> str:
    """Digest the files a comparison was computed from.

    Files are read in sorted order and each contributes its path name and its
    bytes, so a digest cannot collide across a rename or a reordering of the
    inputs.

    Args:
        paths: The files to cover, normally the two arms' outcome files and
            their generation manifests.

    Returns:
        A hex digest.

    Raises:
        ValueError: If no paths are given. A digest over nothing is a
            constant, and recording a constant as a payload digest would make
            every comparison look bit-identical to every other.
    """
    if not paths:
        raise ValueError("payload_digest needs at least one file to cover")
    digest = hashlib.sha256()
    for path in sorted(paths, key=_path_name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scoring_fingerprint() -> RunFingerprint:
    """Capture what this machine contributed to the numbers.

    Three of the six axes are recorded as unknown rather than invented.
    Scoring runs the checkers on the CPU in the caller's own environment: it
    is not in an image, it does not touch the GPU, and it pins no numerical
    determinism. Per the workspace convention an empty digest is a value that
    differs from every known digest, not a wildcard, and an explicitly
    unpinned determinism record says nobody pinned one rather than leaving a
    reader to guess.

    The generation step DID use a GPU, and that is not lost: it is recorded in
    the per-arm generation manifests, which the payload digest covers.

    Returns:
        The fingerprint.
    """
    return RunFingerprint(
        image_digest="",
        gpu_model="",
        driver_version="",
        determinism=determinism_record(UNPINNED_STACK, {}),
        host=capture_host_record(stdlib_host_probe(os.cpu_count)),
        packages=capture_package_versions(SCORING_DISTRIBUTIONS, installed_version),
    )


def comparison_observations(report: ComparisonReport) -> tuple[Observation, ...]:
    """Name every number the comparison concluded.

    Counts travel beside rates deliberately. Three passes out of three and
    thirty out of thirty are both a rate of one, and only one of them is
    evidence, so a record carrying only the rate cannot be read.

    Args:
        report: The computed comparison.

    Returns:
        The observations, in any order; the record sorts them.
    """
    counts = report["counts"]
    return (
        Observation(name="shared_items", value=float(report["shared_items"])),
        Observation(name="baseline_pass_rate", value=report["baseline_pass_rate"]),
        Observation(name="candidate_pass_rate", value=report["candidate_pass_rate"]),
        Observation(name="both_passed", value=float(counts["both_passed"])),
        Observation(name="baseline_only", value=float(counts["baseline_only"])),
        Observation(name="candidate_only", value=float(counts["candidate_only"])),
        Observation(name="neither", value=float(counts["neither"])),
        Observation(name="net_improvement", value=float(report["net_improvement"])),
        Observation(name="mid_p", value=report["mid_p"]),
        Observation(name="exact_p", value=report["exact_p"]),
    )


def comparison_run_record(
    report: ComparisonReport, label: str, covered: Sequence[Path]
) -> RunRecord:
    """Build the record that belongs beside a comparison.

    Args:
        report: The computed comparison.
        label: Which sweep this was, e.g. ``"sweep-v3-cap1536-reppen1.1"``.
        covered: The files the comparison was computed from.

    Returns:
        The record.

    Raises:
        ValueError: Propagated from :func:`run_record` when the label is
            empty, and from :func:`payload_digest` when nothing is covered.
    """
    return run_record(
        experiment=EXPERIMENT,
        label=label,
        fingerprint=scoring_fingerprint(),
        observations=comparison_observations(report),
        payload_digest=payload_digest(covered),
    )


__all__ = [
    "EXPERIMENT",
    "SCORING_DISTRIBUTIONS",
    "comparison_observations",
    "comparison_run_record",
    "payload_digest",
    "scoring_fingerprint",
]
