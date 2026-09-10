"""The benchmarking system: one path from a declared benchmark to its record.

WHY THIS EXISTS, AND WHY IT IS A SYSTEM RATHER THAN A HELPER. On 2026-09-09 a
power audit (board ``6d5536cc``) found that of the six ``benchmark_cleargbm_*``
entry points, exactly ONE wrote a ``RunRecord`` beside its manifest. The first
reading was five oversights. It is not: the six scripts carried 1,213 lines
between them, of which the same ~10 ``add_argument`` calls, the same seven
thread-pinning lines, the same fingerprint construction and the same manifest
write appeared in every one. Nothing OWNED provenance, so emitting it was
something each script had to REMEMBER, and five of six did not.

The ten unfingerprinted manifests found by the same audit are that shape, and
so is ``DEFAULT_SEEDS`` -- a constant documented as reproducing a TIMING
workload -- deciding what five quality comparisons could conclude. A defect
that recurs across every file of a kind is not a mistake in those files.

SO A BENCHMARK DECLARES WHAT IT IS AND RUNS NOTHING ITSELF. This module owns
pin, argument parsing, fingerprint, manifest write, observation building,
record construction and sidecar write, for every family. **An entry point
cannot omit its record, because writing is not a thing an entry point does.**

WHAT IS DATA AND WHAT IS NOT, stated because the boundary is the whole design.
A family's ARGUMENTS, its EXPERIMENT name, its ARM AXES and its METRIC NAMES
are data: ``goss`` is ``("model", "sampling")`` crossed with
``("auc", "log_loss")``, ``ranking`` is ``("model",)`` crossed with
``("mean_ndcg",)``, and ONE builder reads both. There is no per-family
observation builder, which is what the first attempt at this got wrong -- five
functions of identical structure differing only in string literals, which is
the duplication this task's own acceptance forbids.

What is NOT data is the config construction. ``--trees`` sets
``n_estimators``, ``--docs`` sets ``docs_per_query``, ``--early-stopping``
sets ``early_stopping_rounds``; the five families' configs are five distinct
TypedDicts with no common supertype, and building one from a flag mapping
would need a cast. So each family keeps one short typed constructor. Five
constructors for five different types is not a fork -- nothing is duplicated
between them, because they name different fields.

THE PIN ORDERING IS LOAD-BEARING AND THIS MODULE PRESERVES IT. The BLAS thread
count decides how a reduction is partitioned and floating-point addition is not
associative, so the count is an input to every number a benchmark produces --
measured at 865,498 of 16,777,216 matmul elements changing between 1, 8 and 24
threads. The pin must therefore be written before any native numeric library
loads. This module imports nothing numeric, and :func:`run_declared_benchmark`
calls the pin before it invokes the family's runner, which is where numpy
first arrives.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_float,
    require_list,
    require_str,
)
from platform_core.run_record import (
    Observation,
    encode_run_record,
    run_record,
    run_record_sidecar,
)
from typing_extensions import TypedDict


class ArgumentKind(StrEnum):
    """How one declared benchmark argument is parsed.

    A closed vocabulary rather than a callable, so a declaration stays data a
    reader can check against ``--help`` without executing anything.
    """

    INT = "int"
    FLOAT = "float"
    STRING = "string"
    PATH = "path"
    FLAG = "flag"


#: How each valued :class:`ArgumentKind` reaches argparse.
#:
#: A union of concrete types rather than ``Callable[[str], object]``: ``object``
#: in an annotation is the untyped escape hatch this repo forbids, and the four
#: parsers genuinely are these four types. :attr:`ArgumentKind.FLAG` is absent
#: because a flag takes no value; a kind added without a parser raises KeyError
#: here rather than silently parsing as one of the others.
VALUE_TYPES: dict[ArgumentKind, type[int] | type[float] | type[str] | type[Path]] = {
    ArgumentKind.INT: int,
    ArgumentKind.FLOAT: float,
    ArgumentKind.STRING: str,
    ArgumentKind.PATH: Path,
}


class BenchmarkArgument(TypedDict):
    """One flag a benchmark family accepts.

    Attributes:
        flag: The command-line flag, e.g. ``"--trees"``.
        kind: How to parse it.
        help: What it means, shown by ``--help``.
    """

    flag: str
    kind: ArgumentKind
    help: str


class BenchmarkDeclaration(TypedDict):
    """What one benchmark family IS.

    Attributes:
        experiment: The comparability key. Distinct per family, because
            :mod:`platform_core.comparability` reads it to decide whether two
            records may be subtracted -- and a mean NDCG must never be
            subtractable from an R-squared.
        description: Shown by ``--help``.
        arguments: The family's own flags, beyond the shared ones.
        arm_axes: The result keys whose combination names an arm. ``goss``
            measures each model under two sampling modes, so an observation
            named for the model alone would pair a model's GOSS number with
            its own full-sampling number.
        quality_metrics: Keys inside each result's ``quality`` record.
        result_metrics: Keys on the result itself, e.g. ``fit_seconds``.
    """

    experiment: str
    description: str
    arguments: tuple[BenchmarkArgument, ...]
    arm_axes: tuple[str, ...]
    quality_metrics: tuple[str, ...]
    result_metrics: tuple[str, ...]


def shared_parser(declaration: BenchmarkDeclaration) -> argparse.ArgumentParser:
    """Build the parser every benchmark shares, plus this family's own flags.

    Args:
        declaration: The family.

    Returns:
        A parser carrying ``--seeds`` and ``--out``, then the declared flags.
        The model hyperparameters are NOT shared here: five families take
        ``--learning-rate`` and one does not, and a flag that silently does
        nothing in one entry point is the defect class this package's own
        history is full of.
    """
    parser = argparse.ArgumentParser(description=declaration["description"])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Corpus seeds to measure. The count decides what can be concluded.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Manifest path. The run record is written beside it automatically.",
    )
    for argument in declaration["arguments"]:
        if argument["kind"] is ArgumentKind.FLAG:
            # A flag carries its meaning by being present, so it takes no
            # value and cannot be "required" -- absent IS the false case.
            parser.add_argument(argument["flag"], action="store_true", help=argument["help"])
            continue
        parser.add_argument(
            argument["flag"],
            type=VALUE_TYPES[argument["kind"]],
            required=True,
            help=argument["help"],
        )
    return parser


def arm_name(result: JSONObject, arm_axes: Sequence[str]) -> str:
    """Name the arm one result belongs to.

    Args:
        result: One encoded per-arm-per-seed record.
        arm_axes: The keys whose combination identifies an arm.

    Returns:
        The axis values joined by ``@``, e.g. ``"cleargbm@goss"``.

    Raises:
        JSONTypeError: If an axis is missing or is not a string, which would
            mean the manifest and the declaration disagree about the shape of
            an arm.
    """
    return "@".join(require_str(result, axis) for axis in arm_axes)


def declared_observations(
    declaration: BenchmarkDeclaration,
    results: Sequence[JSONObject],
) -> tuple[Observation, ...]:
    """Average every declared metric within every arm.

    The single observation builder for every family. What varies between them
    is the declaration, not this code.

    Args:
        declaration: The family, carrying its arm axes and metric names.
        results: Every encoded per-arm-per-seed record.

    Returns:
        One observation per arm and metric, named
        ``"mean_<arm>.<metric>"``, in first-seen order so two records of one
        experiment list their numbers identically.

    Raises:
        JSONTypeError: If a declared metric is absent from a result or is not
            a number -- the declaration claiming a measurement the manifest
            does not carry.
    """
    totals: dict[str, list[float]] = {}
    for result in results:
        arm = arm_name(result, declaration["arm_axes"])
        quality = _quality_of(result)
        for metric in declaration["quality_metrics"]:
            totals.setdefault(f"{arm}.{metric}", []).append(require_float(quality, metric))
        for metric in declaration["result_metrics"]:
            totals.setdefault(f"{arm}.{metric}", []).append(require_float(result, metric))
    return tuple(
        Observation(name=f"mean_{name}", value=sum(values) / len(values))
        for name, values in totals.items()
    )


def _quality_of(result: JSONObject) -> JSONObject:
    """Read one result's quality record.

    Args:
        result: One encoded per-arm-per-seed record.

    Returns:
        Its ``quality`` sub-record.

    Raises:
        JSONTypeError: If ``quality`` is absent or is not an object.
    """
    return narrow_json_to_dict(result["quality"])


def summary_lines(
    declaration: BenchmarkDeclaration,
    encoded_manifest: JSONValue,
    derived: tuple[Observation, ...] = (),
) -> tuple[str, ...]:
    """Render a finished benchmark for a human watching it run.

    Each of the six entry points hand-rolled its own report loop, in its own
    column widths, over its own metric names. They are one renderer because
    the thing being rendered is the declaration, and a report that drifts from
    the record it summarises is how a reader ends up trusting the wrong
    number.

    The lines are built from the SAME observations the record carries, so the
    two cannot disagree.

    Args:
        declaration: The family.
        encoded_manifest: The manifest as its family's encoder produced it.
        derived: Cross-arm observations, for the family that has them.

    Returns:
        One line per observation, longest name first-aligned.

    Raises:
        JSONTypeError: If the manifest carries no ``results`` list, or a result
            is missing a declared axis or metric.
    """
    observations = declared_observations(declaration, results_of(encoded_manifest)) + derived
    if not observations:
        return ()
    width = max(len(observation["name"]) for observation in observations)
    return tuple(
        f"  {observation['name']:<{width}}  {observation['value']:.6f}"
        for observation in observations
    )


def results_of(encoded_manifest: JSONValue) -> tuple[JSONObject, ...]:
    """Read the per-arm-per-seed records out of an encoded manifest.

    Args:
        encoded_manifest: The manifest as its family's encoder produced it.

    Returns:
        Its results, each validated as an object.

    Raises:
        JSONTypeError: If the manifest carries no ``results`` list, or an entry
            is not an object.
    """
    return tuple(
        narrow_json_to_dict(entry)
        for entry in require_list(narrow_json_to_dict(encoded_manifest), "results")
    )


def write_manifest_and_record(
    declaration: BenchmarkDeclaration,
    encoded_manifest: JSONValue,
    fingerprint: RunFingerprint,
    out_path: Path,
    seeds: Sequence[int],
    derived: tuple[Observation, ...] = (),
    label_prefix: str = "",
) -> Path:
    """Write a finished benchmark's manifest and its record together.

    ONE CALL, so neither can happen without the other. That is the whole point
    of this module: five of six entry points wrote a manifest and no record
    because the two were separate acts one of them had to remember.

    The manifest is taken already ENCODED rather than typed per family. The
    five manifest types share no supertype, and inventing one so this function
    could reach a single field would couple every family to every other; JSON
    is the boundary they already meet at, and every read of it here is
    validated rather than trusted.

    Args:
        declaration: The family, carrying its experiment, arm axes and metrics.
        encoded_manifest: The manifest as its family's encoder produced it.
        fingerprint: The configuration these numbers were produced under.
        out_path: Where the manifest goes; the record is named from it.
        seeds: The seeds measured, which name the run within its experiment.
        derived: Observations a family computes ACROSS arms rather than within
            one, appended after the declared per-arm means. Only
            ``vs_lightgbm`` has any: its headline is a fit-time RATIO between
            two arms, which is not a mean of anything a result carries. Empty
            for every other family, and empty is the honest default rather
            than a fallback -- a family with no cross-arm number passes none.

    Returns:
        The path the record was written to.

    Raises:
        JSONTypeError: If the manifest carries no ``results`` list, or a
            result is missing a declared axis or metric -- the declaration
            claiming a measurement the manifest does not contain.
        ValueError: If two observations collide on a name, meaning two arms
            shared one and a reader could not tell them apart.
    """
    out_path.write_text(dump_json_str(encoded_manifest, indent=1), encoding="utf-8")
    results = results_of(encoded_manifest)
    record = run_record(
        experiment=declaration["experiment"],
        label=f"{label_prefix}{len(seeds)}seeds",
        fingerprint=fingerprint,
        observations=declared_observations(declaration, results) + derived,
        payload_digest=sha256(dump_json_str(list(results)).encode("utf-8")).hexdigest(),
    )
    record_path = run_record_sidecar(out_path)
    record_path.write_text(dump_json_str(encode_run_record(record), indent=1), encoding="utf-8")
    return record_path


__all__ = [
    "VALUE_TYPES",
    "ArgumentKind",
    "BenchmarkArgument",
    "BenchmarkDeclaration",
    "arm_name",
    "declared_observations",
    "results_of",
    "shared_parser",
    "summary_lines",
    "write_manifest_and_record",
]
