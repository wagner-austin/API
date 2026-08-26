"""Whether two measured numbers may be subtracted.

Reproducibility standards answer whether a run can be REPEATED. They do not
answer whether two results may be COMPARED, and that is not an oversight: a
workflow description never sees the comparison, because the comparison happens
afterwards in the analysis. Refusing to subtract two numbers requires owning
the step where the subtraction happens.

The Common Workflow Language's own authors draw the line explicitly. They
write that containers can be seen as confirming a tool's execution is
reproducible under its declared runtime environment, and in the same passage
name variation in the operating-system kernel and variation in PROCESSOR
RESULTS as factors software containers do not control. So an image digest
fixes the software axis and, by the standard's own account, not the hardware
axis.

This module covers that gap for a project that owns submission through
contrast.

THE SHAPE IS A VERDICT, NOT A BOOLEAN, and that is the whole design. Two runs
that differ on some axis are not simply "incomparable": once someone has
measured what that axis is worth on a known input, the difference is a number
you subtract rather than a reason to stop. A boolean throws that away and
forces every caller to re-derive it. So a comparison returns which axes
differ, and either the offset that covers them or the fact that nothing does.

WHAT IT DELIBERATELY DOES NOT DO:

* It does not decide whether an offset is small enough to ignore. That is a
  judgement about the experiment's effect size, and it belongs to whoever
  knows what difference they are trying to read.
* It does not measure offsets. A calibration comes from running one known
  input on both configurations and comparing; this module consumes that
  result and cannot manufacture it.
* It does not compare metrics. It compares the CONFIGURATIONS two metrics
  were produced under, which is the part that silently differs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from typing_extensions import TypedDict

from platform_core.determinism_record import (
    DeterminismRecord,
    decode_determinism_record,
    encode_determinism_record,
    render_determinism_record,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_str,
)


class RunFingerprint(TypedDict):
    """The configuration a measured number was produced under.

    Every field is what the run RESOLVED to, never what it requested. A lock
    file records intent and a run manifest records fact; the two disagreed on
    this project's own published arms, and the manifest was right.

    Attributes:
        image_digest: Content digest of the image that ran, e.g.
            ``sha256:...``. Empty string means unknown, which is itself a
            difference from any known digest rather than a wildcard.
        gpu_model: The card, as the driver reports it.
        driver_version: The GPU driver. Named separately from the card
            because the same card under two drivers can select different
            kernels.
        determinism: What numerical determinism was actually in force, from
            whatever pinner the run's stack uses, e.g.
            :func:`platform_ml.determinism.apply_determinism` for torch.
    """

    image_digest: str
    gpu_model: str
    driver_version: str
    determinism: DeterminismRecord


IMAGE_DIGEST_ENV_VAR = "IMAGE_DIGEST"
"""What the launcher exports to tell a payload which image it is running in.

An image cannot compute its own digest from inside itself -- the digest covers
the whole squashfs, including whatever would be doing the computing. The
launcher knows it, because the job's spec pins it, so it exports it and the
payload reads it.

Named here rather than in each service because every research stack in this
monorepo eventually asks the same question, and a second spelling of the
variable is a silent "unknown image" in whichever copy drifts.
"""

NO_VALUE = ""
"""What an axis records when nothing is known about it.

Empty rather than absent, and never a wildcard: it differs from every real
value, so a run with no image never compares equal to one with an image, and
a cpu run never compares equal to a cuda one. An axis that were simply absent
would compare equal to any other record missing the same axis, which is the
one failure a comparability record must not have.
"""


def image_digest_from_env(get_env: Callable[[str], str | None]) -> str:
    """Read the digest of the image this process is running in.

    Args:
        get_env: Reader for a process environment variable, injected so a
            test can state what the launcher exported without touching the
            real environment.

    Returns:
        The digest, or :const:`NO_VALUE` when the variable is unset or empty.
        Both mean the same thing -- nobody told this process which image it
        is in -- and that is the honest answer for a run out of a directory
        environment, where there is no image and so no digest.
    """
    value = get_env(IMAGE_DIGEST_ENV_VAR)
    if value is None:
        return NO_VALUE
    return value


def cpu_run_fingerprint(
    determinism: DeterminismRecord, get_env: Callable[[str], str | None]
) -> RunFingerprint:
    """Describe the configuration of a run that uses no GPU.

    For the research in this monorepo that pulls no torch -- gradient
    boosting, transliteration, metabolomics. Those runs still have a
    configuration that decides their numbers: which image, and what the BLAS
    thread count was pinned to (see :mod:`platform_core.determinism_cpu`,
    where a 4096x4096 matmul changed 865,498 of 16,777,216 elements between
    thread counts).

    The card and driver are recorded as :const:`NO_VALUE` rather than
    omitted. A cpu run genuinely has no card, and saying so is what stops it
    comparing equal to a cuda run of the same code.

    Args:
        determinism: What was actually pinned, from
            :func:`~platform_core.determinism_cpu.apply_cpu_determinism`.
            Passed in rather than applied here: pinning writes process-global
            state and belongs to the job, while this only describes.
        get_env: Reader for a process environment variable.

    Returns:
        The fingerprint, comparable against any other by
        :func:`compare_configurations`.
    """
    return RunFingerprint(
        image_digest=image_digest_from_env(get_env),
        gpu_model=NO_VALUE,
        driver_version=NO_VALUE,
        determinism=determinism,
    )


class AxisDifference(TypedDict):
    """One axis on which two fingerprints disagree.

    Attributes:
        axis: Which axis, drawn from :const:`COMPARABILITY_AXES`.
        left: The left run's value, rendered for display.
        right: The right run's value, rendered for display.
    """

    axis: str
    left: str
    right: str


class Calibration(TypedDict):
    """A measured offset spanning one axis, from a known input run twice.

    Attributes:
        axis: The axis this calibration spans.
        left: The left value of that axis.
        right: The right value of that axis.
        offset: ``right`` minus ``left`` in the metric's own units, measured
            by running one known input under both.
        measured_by: What established it, e.g. a run id pair. Non-empty by
            construction at the decode boundary: an offset whose provenance
            is unrecorded cannot be audited later, and an un-auditable
            correction is worse than none.
    """

    axis: str
    left: str
    right: str
    offset: float
    measured_by: str


class IdenticalVerdict(TypedDict):
    """The configurations match on every axis, so the numbers subtract."""

    kind: Literal["identical"]


class OffsetVerdict(TypedDict):
    """The configurations differ, and every difference has a measurement.

    Attributes:
        kind: Discriminant.
        differences: The axes that differ, in :const:`COMPARABILITY_AXES`
            order.
        offset: The sum of the calibrated offsets covering them.
        calibrations: The measurements applied, so a reader can audit the
            correction rather than trust it.
    """

    kind: Literal["offset"]
    differences: tuple[AxisDifference, ...]
    offset: float
    calibrations: tuple[Calibration, ...]


class UncalibratedVerdict(TypedDict):
    """The configurations differ on an axis nobody has measured.

    Attributes:
        kind: Discriminant.
        differences: Every axis that differs.
        uncalibrated: The subset with no covering measurement. These are the
            ones that make a subtraction meaningless, and naming them tells
            the reader exactly which calibration run would fix it.
    """

    kind: Literal["uncalibrated"]
    differences: tuple[AxisDifference, ...]
    uncalibrated: tuple[AxisDifference, ...]


#: How each axis is read off a fingerprint, keyed by axis name.
#:
#: A mapping rather than a chain of comparisons, so there is no "unknown
#: axis" case to guard: the only axes that exist are the ones with a reader.
#: An earlier version raised on an unrecognised name, which was unreachable
#: through any caller and therefore dead code that a coverage gate correctly
#: refused. Removing the possibility beats defending against it.
_AXIS_READERS: Mapping[str, Callable[[RunFingerprint], str]] = {
    "image_digest": lambda f: f["image_digest"],
    "gpu_model": lambda f: f["gpu_model"],
    "driver_version": lambda f: f["driver_version"],
    "determinism": lambda f: render_determinism_record(f["determinism"]),
}

#: The axes a run's numbers depend on, in the order a verdict reports them.
#:
#: Derived from the readers rather than written out beside them, so the list
#: of axes and the ability to read them cannot drift apart. Insertion order
#: is the reporting order, which keeps two verdicts over the same pair
#: byte-identical.
COMPARABILITY_AXES: tuple[str, ...] = tuple(_AXIS_READERS)


def find_differences(left: RunFingerprint, right: RunFingerprint) -> tuple[AxisDifference, ...]:
    """Report every axis on which two fingerprints disagree.

    Args:
        left: One run's configuration.
        right: The other's.

    Returns:
        The differing axes in :const:`COMPARABILITY_AXES` order, empty when
        the configurations match.
    """
    found: list[AxisDifference] = []
    for axis, read in _AXIS_READERS.items():
        lhs = read(left)
        rhs = read(right)
        if lhs != rhs:
            found.append(AxisDifference(axis=axis, left=lhs, right=rhs))
    return tuple(found)


def _covering(
    difference: AxisDifference, calibrations: tuple[Calibration, ...]
) -> Calibration | None:
    """Find the calibration measuring one difference, in either direction.

    A calibration measured left-to-right also answers right-to-left with the
    sign flipped, so both orientations are accepted and the returned offset
    is oriented to the difference as asked.

    Args:
        difference: The axis disagreement to cover.
        calibrations: Available measurements.

    Returns:
        A calibration oriented to ``difference``, or None when none covers it.
    """
    for calibration in calibrations:
        if calibration["axis"] != difference["axis"]:
            continue
        if (
            calibration["left"] == difference["left"]
            and calibration["right"] == difference["right"]
        ):
            return calibration
        if (
            calibration["left"] == difference["right"]
            and calibration["right"] == difference["left"]
        ):
            return Calibration(
                axis=calibration["axis"],
                left=difference["left"],
                right=difference["right"],
                offset=-calibration["offset"],
                measured_by=calibration["measured_by"],
            )
    return None


def compare_configurations(
    left: RunFingerprint,
    right: RunFingerprint,
    calibrations: tuple[Calibration, ...],
) -> IdenticalVerdict | OffsetVerdict | UncalibratedVerdict:
    """Decide whether two runs' numbers may be subtracted.

    Named for CONFIGURATIONS rather than runs because it compares the
    fingerprints, not the results: it answers whether a comparison is
    licensed, never whether two runs agreed. ``navprobe.comparison`` already
    owns ``compare_runs`` for the other question -- do two runs of the SAME
    configuration produce the same numbers, and where did they first diverge
    -- and two functions of that name meaning opposite things is the drift
    this rename exists to prevent.

    Args:
        left: One run's resolved configuration.
        right: The other's.
        calibrations: Measured offsets available to bridge differences.

    Returns:
        ``identical`` when nothing differs; ``offset`` when everything that
        differs has a measurement, carrying their sum and the measurements
        applied; ``uncalibrated`` when at least one difference has none,
        naming exactly which.
    """
    differences = find_differences(left, right)
    if not differences:
        return IdenticalVerdict(kind="identical")

    applied: list[Calibration] = []
    missing: list[AxisDifference] = []
    for difference in differences:
        covering = _covering(difference, calibrations)
        if covering is None:
            missing.append(difference)
        else:
            applied.append(covering)

    if missing:
        return UncalibratedVerdict(
            kind="uncalibrated",
            differences=differences,
            uncalibrated=tuple(missing),
        )
    return OffsetVerdict(
        kind="offset",
        differences=differences,
        offset=sum(c["offset"] for c in applied),
        calibrations=tuple(applied),
    )


def describe_verdict(
    verdict: IdenticalVerdict | OffsetVerdict | UncalibratedVerdict,
) -> str:
    """Render a verdict as one line for a log or a run report.

    Args:
        verdict: The comparison outcome.

    Returns:
        A line naming the outcome and, for the two differing cases, the axes
        involved, so a reader who sees only the log knows which calibration
        would resolve it.
    """
    if verdict["kind"] == "identical":
        return "comparable: configurations identical"
    axes = ",".join(d["axis"] for d in verdict["differences"])
    if verdict["kind"] == "offset":
        return f"comparable with offset {verdict['offset']:+.4f} across {axes}"
    unmeasured = ",".join(d["axis"] for d in verdict["uncalibrated"])
    return f"NOT comparable: differs on {axes}; unmeasured: {unmeasured}"


def encode_run_fingerprint(fingerprint: RunFingerprint) -> JSONObject:
    """Encode a fingerprint for a run record.

    Args:
        fingerprint: The configuration to encode.

    Returns:
        A JSON object carrying every axis, with the determinism report
        nested rather than flattened, so the two decoders stay independent.
    """
    return {
        "image_digest": fingerprint["image_digest"],
        "gpu_model": fingerprint["gpu_model"],
        "driver_version": fingerprint["driver_version"],
        "determinism": encode_determinism_record(fingerprint["determinism"]),
    }


def decode_run_fingerprint(value: JSONValue) -> RunFingerprint:
    """Validate a JSON value as a run fingerprint.

    Args:
        value: The value to validate, typically read from a stored record.

    Returns:
        The validated fingerprint.

    Raises:
        JSONTypeError: When ``value`` is not an object, when any axis is
            absent or mistyped, or when the nested determinism report fails
            its own validation. Absence is rejected rather than defaulted:
            a fingerprint missing an axis would compare equal to another
            missing the same axis, reporting two differently-configured runs
            as identical.
    """
    obj = narrow_json_to_dict(value)
    return RunFingerprint(
        image_digest=require_str(obj, "image_digest"),
        gpu_model=require_str(obj, "gpu_model"),
        driver_version=require_str(obj, "driver_version"),
        determinism=decode_determinism_record(require_dict(obj, "determinism")),
    )


def encode_calibration(calibration: Calibration) -> JSONObject:
    """Encode a measured offset for storage.

    Args:
        calibration: The measurement to encode.

    Returns:
        A JSON object carrying the axis, both endpoints, the offset and its
        provenance.
    """
    return {
        "axis": calibration["axis"],
        "left": calibration["left"],
        "right": calibration["right"],
        "offset": calibration["offset"],
        "measured_by": calibration["measured_by"],
    }


def decode_calibration(value: JSONValue) -> Calibration:
    """Validate a JSON value as a calibration.

    Args:
        value: The value to validate.

    Returns:
        The validated calibration.

    Raises:
        JSONTypeError: When ``value`` is not an object or any field is absent
            or mistyped, when ``axis`` is not a known comparability axis, or
            when ``measured_by`` is empty. An offset naming an axis nothing
            compares would silently never apply, and an offset with no
            recorded provenance cannot be audited, which makes it worse than
            having none.
    """
    obj = narrow_json_to_dict(value)
    axis = require_str(obj, "axis")
    if axis not in COMPARABILITY_AXES:
        raise JSONTypeError(f"Field 'axis' must be one of {COMPARABILITY_AXES}, got {axis!r}")
    measured_by = require_str(obj, "measured_by")
    if measured_by == "":
        raise JSONTypeError("Field 'measured_by' must name what established the offset")
    return Calibration(
        axis=axis,
        left=require_str(obj, "left"),
        right=require_str(obj, "right"),
        offset=require_float(obj, "offset"),
        measured_by=measured_by,
    )


def encode_comparability_verdict(
    verdict: IdenticalVerdict | OffsetVerdict | UncalibratedVerdict,
) -> JSONObject:
    """Encode a verdict for a run report.

    Args:
        verdict: The comparison outcome.

    Returns:
        A JSON object whose ``kind`` discriminates the shape, carrying the
        differing axes and either the offset applied or the axes that lack a
        measurement.
    """
    if verdict["kind"] == "identical":
        return {"kind": "identical"}
    differences: JSONValue = [
        {"axis": d["axis"], "left": d["left"], "right": d["right"]} for d in verdict["differences"]
    ]
    if verdict["kind"] == "offset":
        return {
            "kind": "offset",
            "differences": differences,
            "offset": verdict["offset"],
            "calibrations": [encode_calibration(c) for c in verdict["calibrations"]],
        }
    return {
        "kind": "uncalibrated",
        "differences": differences,
        "uncalibrated": [
            {"axis": d["axis"], "left": d["left"], "right": d["right"]}
            for d in verdict["uncalibrated"]
        ],
    }


__all__ = [
    "COMPARABILITY_AXES",
    "IMAGE_DIGEST_ENV_VAR",
    "NO_VALUE",
    "AxisDifference",
    "Calibration",
    "IdenticalVerdict",
    "OffsetVerdict",
    "RunFingerprint",
    "UncalibratedVerdict",
    "compare_configurations",
    "cpu_run_fingerprint",
    "decode_calibration",
    "decode_run_fingerprint",
    "describe_verdict",
    "encode_calibration",
    "encode_comparability_verdict",
    "encode_run_fingerprint",
    "find_differences",
    "image_digest_from_env",
]
