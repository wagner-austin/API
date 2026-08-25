"""A cheap input whose output is known, and what checking it can conclude.

An image that still builds is not an image that still computes what it used
to. Nothing in the reproducibility standards validates the second: CWL
validates the DESCRIPTION, that a tool is well formed and its container
named, and MLflow describes entry points. Neither runs the thing and compares
the answer.

That gap has a measured cost on this project. A rebuilt image silently
changed its torch major version, the change went unnoticed, and it was found
only after a training run whose result could not be interpreted. A known
answer catches that on the first submission, in seconds, before anything is
staged.

WHAT A KNOWN ANSWER IS: a named cheap input, the value it produces, the
tolerance that value must land in, and THE CONFIGURATION IT WAS ESTABLISHED
UNDER. The last part is what makes it honest. An expected loss is not a
property of an experiment; it is a property of an experiment run on a
particular image on a particular card with particular determinism settings.
Recording the answer without the configuration produces a check that fails
correctly for the wrong reason the first time anyone changes hardware.

SO THERE ARE THREE OUTCOMES, NOT TWO. A value can match, or deviate, or the
question can fail to apply because the run happened under a configuration the
answer was never established for. Collapsing the third into "deviates" would
report a working image as broken every time it moved to a different card,
which trains everyone to ignore the check.

This module decides; it does not enforce. Whether a submission is refused on
a deviation belongs to whatever owns submission, which is not this layer.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_str,
)
from typing_extensions import TypedDict

from platform_ml.comparability import (
    AxisDifference,
    RunFingerprint,
    decode_run_fingerprint,
    encode_run_fingerprint,
    find_differences,
)


class KnownAnswer(TypedDict):
    """A cheap input with an expected output, pinned to its configuration.

    Attributes:
        label: What input produces this, e.g. an arm and seed and step count.
            Non-empty by construction at the decode boundary: a failing check
            that cannot name what it ran is not actionable.
        fingerprint: The configuration the expected value was established
            under. A value carries no meaning apart from it.
        expected: The value that configuration produces.
        tolerance: The absolute deviation still counted as a match. Zero
            means bit-exact, which is the right default WITHIN one
            configuration once determinism is pinned, and is why moving
            configuration is a separate outcome rather than a wider band.
    """

    label: str
    fingerprint: RunFingerprint
    expected: float
    tolerance: float


class AnswerMatches(TypedDict):
    """The observed value is inside tolerance for this configuration.

    Attributes:
        kind: Discriminant.
        observed: What the run produced.
        deviation: Absolute difference from the expected value, reported even
            on a match so a drift toward the edge of tolerance is visible
            before it crosses.
    """

    kind: Literal["matches"]
    observed: float
    deviation: float


class AnswerDeviates(TypedDict):
    """The observed value is outside tolerance for this configuration.

    This is the image-is-broken signal, and it is only trustworthy because
    the configuration matched first.

    Attributes:
        kind: Discriminant.
        observed: What the run produced.
        deviation: Absolute difference from the expected value.
        tolerance: The band it failed to land in, carried so a reader does
            not have to fetch the answer to judge the size of the miss.
    """

    kind: Literal["deviates"]
    observed: float
    deviation: float
    tolerance: float


class AnswerNotApplicable(TypedDict):
    """The run happened under a configuration this answer never covered.

    Not a failure. A known answer establishes what one configuration
    produces, so under a different one it has nothing to say, and treating
    silence as a deviation would condemn a working image for moving cards.

    Attributes:
        kind: Discriminant.
        differences: The axes on which the run differs from the answer's
            configuration, so the reader knows whether to establish a new
            answer or to fix the run.
    """

    kind: Literal["configuration_differs"]
    differences: tuple[AxisDifference, ...]


def check_known_answer(
    known: KnownAnswer, observed_fingerprint: RunFingerprint, observed: float
) -> AnswerMatches | AnswerDeviates | AnswerNotApplicable:
    """Check an observed value against a known answer.

    The configuration is checked FIRST. A value produced under a different
    image, card, driver or determinism setting cannot confirm or refute an
    answer established elsewhere, so no comparison of the numbers is
    attempted and none is reported.

    Args:
        known: The answer to check against.
        observed_fingerprint: The configuration the observed value was
            produced under.
        observed: The value produced.

    Returns:
        ``configuration_differs`` when the run's configuration is not the
        answer's, naming the axes; otherwise ``matches`` or ``deviates``
        according to the absolute deviation and the answer's tolerance.
    """
    differences = find_differences(known["fingerprint"], observed_fingerprint)
    if differences:
        return AnswerNotApplicable(kind="configuration_differs", differences=differences)

    deviation = abs(observed - known["expected"])
    if deviation <= known["tolerance"]:
        return AnswerMatches(kind="matches", observed=observed, deviation=deviation)
    return AnswerDeviates(
        kind="deviates",
        observed=observed,
        deviation=deviation,
        tolerance=known["tolerance"],
    )


def encode_known_answer(known: KnownAnswer) -> JSONObject:
    """Encode a known answer for storage beside an experiment.

    Args:
        known: The answer to encode.

    Returns:
        A JSON object carrying the label, the expected value, the tolerance
        and the nested configuration.
    """
    return {
        "label": known["label"],
        "fingerprint": encode_run_fingerprint(known["fingerprint"]),
        "expected": known["expected"],
        "tolerance": known["tolerance"],
    }


def decode_known_answer(value: JSONValue) -> KnownAnswer:
    """Validate a JSON value as a known answer.

    Args:
        value: The value to validate.

    Returns:
        The validated answer.

    Raises:
        JSONTypeError: When ``value`` is not an object, any field is absent
            or mistyped, the nested fingerprint fails its own validation, the
            label is empty, or the tolerance is negative. A negative
            tolerance admits no value at all, so a check carrying one can
            only ever report a deviation and would read as a broken image
            rather than a broken answer.
    """
    obj = narrow_json_to_dict(value)
    label = require_str(obj, "label")
    if label == "":
        raise JSONTypeError("Field 'label' must name the input that produces the value")
    tolerance = require_float(obj, "tolerance")
    if tolerance < 0.0:
        raise JSONTypeError(f"Field 'tolerance' must not be negative, got {tolerance}")
    return KnownAnswer(
        label=label,
        fingerprint=decode_run_fingerprint(require_dict(obj, "fingerprint")),
        expected=require_float(obj, "expected"),
        tolerance=tolerance,
    )


def describe_known_answer_outcome(
    known: KnownAnswer,
    outcome: AnswerMatches | AnswerDeviates | AnswerNotApplicable,
) -> str:
    """Render an outcome as one line for a submission log.

    Args:
        known: The answer that was checked, for its label.
        outcome: What the check concluded.

    Returns:
        A line naming the answer and the conclusion, including the axes when
        the answer did not apply, so a reader who sees only the log knows
        whether to establish a new answer or to reject the image.
    """
    label = known["label"]
    if outcome["kind"] == "matches":
        return f"known answer {label!r}: matches (deviation {outcome['deviation']:.6g})"
    if outcome["kind"] == "deviates":
        return (
            f"known answer {label!r}: DEVIATES by {outcome['deviation']:.6g}, "
            f"tolerance {outcome['tolerance']:.6g}"
        )
    axes = ",".join(d["axis"] for d in outcome["differences"])
    return f"known answer {label!r}: does not apply, configuration differs on {axes}"


__all__ = [
    "AnswerDeviates",
    "AnswerMatches",
    "AnswerNotApplicable",
    "KnownAnswer",
    "check_known_answer",
    "decode_known_answer",
    "describe_known_answer_outcome",
    "encode_known_answer",
]
