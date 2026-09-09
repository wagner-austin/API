"""What the guard-pass instrument records, per item and per arm.

The unit of measurement is ONE generated file under ONE checker, and the
records keep it that way to the end. A guard-pass rate is a summary of these
rows and never a substitute for them: a paired comparison needs to know
which items each arm passed, not how many.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)
from platform_core.power_records import (
    decode_mcnemar_power,
    decode_net_difference_power,
    encode_mcnemar_power,
    encode_net_difference_power,
)
from platform_core.power_types import McNemarPower, NetDifferencePower
from typing_extensions import TypedDict

#: The three checkers a generated file is scored under, and the ONE place the
#: set is written. ``scripts/guard.py`` fails the lint when a checker is added
#: to the pipeline without appearing here.
#:
#: They are not interchangeable. ``ruff`` scores syntax and lint idiom,
#: ``mypy`` scores the typing discipline the house standard is mostly about,
#: and ``guards`` scores the monorepo's own architectural rules, which no
#: off-the-shelf tool encodes. A model can pass the first and fail the third
#: badly, so the rate is reported per checker as well as combined.
CHECKERS: tuple[Literal["ruff", "mypy", "guards"], ...] = ("ruff", "mypy", "guards")


def as_checker(raw: str, field: str) -> Literal["ruff", "mypy", "guards"]:
    """Narrow a string to a checker name, or refuse it.

    Args:
        raw: The string to narrow.
        field: Field name for the error message.

    Returns:
        The narrowed checker name.

    Raises:
        JSONTypeError: If the string names no known checker.
    """
    for known in CHECKERS:
        if raw == known:
            return known
    raise JSONTypeError(f"Field '{field}' must be one of {CHECKERS}, got '{raw}'")


class CheckOutcome(TypedDict):
    """One checker's verdict on one generated file.

    Attributes:
        checker: Which checker ran.
        passed: Whether it reported no findings.
        exit_code: The checker's own exit status, kept because a crash and a
            clean run are both "did not report findings" to a caller reading
            only ``passed``, and they are not the same event.
        detail: The first line of output, or the empty string on a pass. Kept
            short deliberately: this is an index into the run's logs, not a
            replacement for them.
    """

    checker: Literal["ruff", "mypy", "guards"]
    passed: bool
    exit_code: int
    detail: str


class ItemOutcome(TypedDict):
    """Every checker's verdict on one generated file, under one arm.

    Attributes:
        item_id: The held-out file this completion was generated for. Shared
            across arms, which is what makes the comparison paired.
        arm: Which model produced the completion.
        checks: One outcome per checker, in ``CHECKERS`` order.
        all_passed: True when every checker passed. Stored rather than
            recomputed so a decoded record cannot disagree with the encoder
            about what it means.
    """

    item_id: str
    arm: str
    checks: tuple[CheckOutcome, ...]
    all_passed: bool


class PairedCounts(TypedDict):
    """The 2x2 table two arms produce over the same items.

    Attributes:
        both_passed: Items both arms passed.
        baseline_only: Items only the baseline passed.
        candidate_only: Items only the candidate passed.
        neither: Items neither passed.
    """

    both_passed: int
    baseline_only: int
    candidate_only: int
    neither: int


class ComparisonReport(TypedDict):
    """Everything a two-arm comparison concluded, in one record.

    The p-values are carried as a PAIR rather than one number. The mid-p
    value is the one to read, and the exact conditional value is kept beside
    it because it is the guaranteed-level reference mid-p is derived from,
    and because a reader who expects the conservative figure should be able
    to see it rather than wonder which was computed.

    ``net_improvement`` is stored alongside them because a p-value on two
    discordant pairs still describes two files, and a report that carried
    significance without effect size would invite reading the first as the
    second.

    ``payload_digest`` IS PART OF THE REPORT AND NOT A SIDECAR, and that is
    the whole reason this class was changed on 2026-09-09. Every other field
    here is a number. Numbers do not say what produced them, and two runs of
    this package can agree on every one of them by coincidence: ``gen-v1``
    and ``sweep-v1`` both scored 226 shared items, and a reader holding "226
    items, mid-p 0.688" has no way to tell which run it came from. Until this
    field existed a ``comparison.json`` was identified ONLY by the directory
    it sat in, so the identity was lost the instant a figure was quoted onto
    a wiki page, into a board post, or into a script -- and a figure that
    cannot be traced back to its bytes gets traced back by resemblance
    instead, which finds the wrong run. That happened on 2026-09-09 and the
    recomputation looked entirely correct.

    The digest was always being computed; it was written to
    ``*.runrecord.json`` beside the report rather than into it. Beside is not
    good enough for the one property that has to survive being quoted, and
    two of this package's six committed runs had no sidecar at all.

    Attributes:
        baseline_arm: Name recorded on the baseline outcomes.
        candidate_arm: Name recorded on the candidate outcomes.
        shared_items: Items both arms produced a completion for. The
            denominator every figure below is computed over.
        baseline_pass_rate: Fraction of shared items the baseline passed.
        candidate_pass_rate: Fraction of shared items the candidate passed.
        counts: The 2x2 table.
        net_improvement: Items fixed minus items broken.
        mid_p: Two-sided McNemar mid-p value.
        exact_p: Two-sided exact conditional McNemar p-value.
        net_power: Whether the OBSERVED NET could ever have been significant,
            from
            :func:`platform_core.minimum_detectable_effect.net_difference_power`.
            A SECOND and independent floor: ``power`` asks whether this
            discordant count could ever reject, this asks whether this net
            could. A comparison passes the first and fails the second whenever
            the discordant count is large and the split near even, and four of
            this package's own published rows were in exactly that state while
            the first floor called them fine.
        power: What this comparison could ever have resolved, from
            :func:`platform_core.minimum_detectable_effect.mcnemar_power`.
            Carried BESIDE the p-values so a reader never has to ask whether
            a null was a finding or a sample size -- the record answers it.
        payload_digest: Name-paired sha256 over the two outcome files this
            was computed from, from
            :func:`code_style_eval.core.provenance.payload_digest`. Never
            empty: a report that cannot say which bytes produced it is the
            anonymous artifact this field exists to abolish.
    """

    baseline_arm: str
    candidate_arm: str
    shared_items: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    counts: PairedCounts
    net_improvement: int
    mid_p: float
    exact_p: float
    payload_digest: str
    power: McNemarPower
    net_power: NetDifferencePower


def encode_comparison_report(report: ComparisonReport) -> JSONObject:
    """Encode a ComparisonReport to a JSON object.

    Args:
        report: The report to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "baseline_arm": report["baseline_arm"],
        "candidate_arm": report["candidate_arm"],
        "shared_items": report["shared_items"],
        "baseline_pass_rate": report["baseline_pass_rate"],
        "candidate_pass_rate": report["candidate_pass_rate"],
        "counts": encode_paired_counts(report["counts"]),
        "net_improvement": report["net_improvement"],
        "mid_p": report["mid_p"],
        "exact_p": report["exact_p"],
        "payload_digest": report["payload_digest"],
        "power": encode_mcnemar_power(report["power"]),
        "net_power": encode_net_difference_power(report["net_power"]),
    }


def decode_comparison_report(obj: JSONObject) -> ComparisonReport:
    """Decode a JSON object to a ComparisonReport.

    Args:
        obj: The object to decode.

    Returns:
        The validated report.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, the stored
            table does not sum to the stored item count, or the payload
            digest is empty.
    """
    counts = decode_paired_counts(require_dict(obj, "counts"))
    shared_items = require_int(obj, "shared_items")
    payload_digest = require_str(obj, "payload_digest")
    if not payload_digest:
        raise JSONTypeError(
            "Field 'payload_digest' is empty; a comparison that cannot name "
            "the bytes it was computed from is indistinguishable from one "
            "computed over different bytes, which is the confusion this field "
            "exists to prevent"
        )
    total = (
        counts["both_passed"]
        + counts["baseline_only"]
        + counts["candidate_only"]
        + counts["neither"]
    )
    if total != shared_items:
        raise JSONTypeError(
            f"Field 'shared_items' is {shared_items} but the 2x2 table sums to "
            f"{total}; a report whose denominator disagrees with its own table "
            f"cannot be read"
        )
    return ComparisonReport(
        baseline_arm=require_str(obj, "baseline_arm"),
        candidate_arm=require_str(obj, "candidate_arm"),
        shared_items=shared_items,
        baseline_pass_rate=require_float(obj, "baseline_pass_rate"),
        candidate_pass_rate=require_float(obj, "candidate_pass_rate"),
        counts=counts,
        net_improvement=require_int(obj, "net_improvement"),
        mid_p=require_float(obj, "mid_p"),
        exact_p=require_float(obj, "exact_p"),
        payload_digest=payload_digest,
        power=decode_mcnemar_power(require_dict(obj, "power")),
        net_power=decode_net_difference_power(require_dict(obj, "net_power")),
    )


def encode_check_outcome(outcome: CheckOutcome) -> JSONObject:
    """Encode a CheckOutcome to a JSON object.

    Args:
        outcome: The outcome to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "checker": outcome["checker"],
        "passed": outcome["passed"],
        "exit_code": outcome["exit_code"],
        "detail": outcome["detail"],
    }


def decode_check_outcome(obj: JSONObject) -> CheckOutcome:
    """Decode a JSON object to a CheckOutcome.

    Args:
        obj: The object to decode.

    Returns:
        The validated outcome.

    Raises:
        JSONTypeError: If a field is missing or has the wrong type.
    """
    return CheckOutcome(
        checker=as_checker(require_str(obj, "checker"), "checker"),
        passed=require_bool(obj, "passed"),
        exit_code=require_int(obj, "exit_code"),
        detail=require_str(obj, "detail"),
    )


def encode_item_outcome(outcome: ItemOutcome) -> JSONObject:
    """Encode an ItemOutcome to a JSON object.

    Args:
        outcome: The outcome to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "item_id": outcome["item_id"],
        "arm": outcome["arm"],
        "checks": [encode_check_outcome(check) for check in outcome["checks"]],
        "all_passed": outcome["all_passed"],
    }


def decode_item_outcome(obj: JSONObject) -> ItemOutcome:
    """Decode a JSON object to an ItemOutcome.

    Args:
        obj: The object to decode.

    Returns:
        The validated outcome.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, or the
            stored ``all_passed`` disagrees with the checks it summarises.
    """
    checks = tuple(
        decode_check_outcome(narrow_json_to_dict(entry)) for entry in require_list(obj, "checks")
    )
    all_passed = require_bool(obj, "all_passed")
    if all_passed != all(check["passed"] for check in checks):
        raise JSONTypeError(
            "Field 'all_passed' disagrees with 'checks'; a summary that "
            "contradicts its own rows cannot be compared against another arm"
        )
    return ItemOutcome(
        item_id=require_str(obj, "item_id"),
        arm=require_str(obj, "arm"),
        checks=checks,
        all_passed=all_passed,
    )


def encode_paired_counts(counts: PairedCounts) -> JSONObject:
    """Encode PairedCounts to a JSON object.

    Args:
        counts: The counts to encode.

    Returns:
        The JSON-serializable form.
    """
    return {
        "both_passed": counts["both_passed"],
        "baseline_only": counts["baseline_only"],
        "candidate_only": counts["candidate_only"],
        "neither": counts["neither"],
    }


def decode_paired_counts(obj: JSONObject) -> PairedCounts:
    """Decode a JSON object to PairedCounts.

    Args:
        obj: The object to decode.

    Returns:
        The validated counts.

    Raises:
        JSONTypeError: If a field is missing or has the wrong type.
    """
    return PairedCounts(
        both_passed=require_int(obj, "both_passed"),
        baseline_only=require_int(obj, "baseline_only"),
        candidate_only=require_int(obj, "candidate_only"),
        neither=require_int(obj, "neither"),
    )


__all__ = [
    "CHECKERS",
    "CheckOutcome",
    "ComparisonReport",
    "ItemOutcome",
    "PairedCounts",
    "as_checker",
    "decode_check_outcome",
    "decode_comparison_report",
    "decode_item_outcome",
    "decode_paired_counts",
    "encode_check_outcome",
    "encode_comparison_report",
    "encode_item_outcome",
    "encode_paired_counts",
]
