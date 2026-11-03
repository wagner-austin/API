"""Count every panel path's first refusal under the frozen design, without fitting.

Qualification answers one question before any candidate is fitted: can the
frozen evaluator value every eligible member of every decision whose window
ends before the final period? It walks the panel exactly as label
construction does (:func:`~covenant_ml.markets.crsp_study.crsp_known_panel`,
then each decision's eligible securities in liquidity order) and asks
:func:`~covenant_ml.markets.crsp_holdings.crsp_resolve_window_path` for each
path. Nothing is fitted, selected or released.

A refused path is reported by its one-based ordinal in that walk, its decision
index, its period and the refusal's counting key, with SHA-256 digests of its
entry and exit daily records so a reviewer can join it to earlier diagnostics.
No security identifier, calendar date, price or return is published. Every
label depends on its decision's full-universe benchmark, so the report also
counts the decisions, and the rows in them, that any refusal leaves unlabelled.

An unfilled order (an entry record without a closing trade) is valued as cash,
so its row is labelled against the benchmark with a multiplier of exactly one:
positive in every month the benchmark falls. That is a changed target, the
order's outcome rather than the security's, so each decision's unfilled count
is published beside its size and counted by period.

The record digests are unsalted, so a holder of the same vendor row can confirm
it; they are kept only to join the version-3 diagnosis, which published the
same digests.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import pairwise

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .crsp_extract import CrspSources
from .crsp_holdings import (
    CrspOutcomes,
    CrspPath,
    crsp_resolve_window_path,
    index_crsp_outcomes,
)
from .crsp_study import crsp_known_panel
from .validation import require_nonempty

#: The two periods a train-validation path can belong to.
PERIODS = ("training", "validation")


class CrspRefusedPath(TypedDict):
    """One path the frozen evaluator cannot value, without its observations.

    Attributes:
        ordinal: One-based position in the decision-then-liquidity walk.
        decision: Zero-based index of its decision in the panel.
        period: ``training`` or ``validation``, by the decision date.
        reason: The refusal's counting key.
        entry_sha256: Digest of the entry close's daily record, None when absent.
        exit_sha256: Digest of the exit close's daily record, None when absent.
    """

    ordinal: ReadOnly[int]
    decision: ReadOnly[int]
    period: ReadOnly[str]
    reason: ReadOnly[str]
    entry_sha256: ReadOnly[str | None]
    exit_sha256: ReadOnly[str | None]


class CrspQualification(TypedDict):
    """Every path's outcome under the frozen evaluator, refused paths in full.

    Attributes:
        members: Eligible securities per decision, in panel order.
        unfilled: Unfilled orders per decision, valued as cash, in panel order.
        training_decisions: How many leading decisions precede validation.
        refused: Every refused path, in walk order.
    """

    members: ReadOnly[tuple[int, ...]]
    unfilled: ReadOnly[tuple[int, ...]]
    training_decisions: ReadOnly[int]
    refused: ReadOnly[tuple[CrspRefusedPath, ...]]


def _digest(outcomes: CrspOutcomes, permno: int, day: str) -> str | None:
    """Hash one daily record's canonical JSON, or None when the vendor has none."""
    row = outcomes.returns.get((permno, day))
    return None if row is None else sha256(dump_json_str(row).encode()).hexdigest()


def qualify_crsp_panel(sources: CrspSources) -> CrspQualification:
    """Resolve every known decision's every eligible path and keep each refusal.

    Args:
        sources: Verified train-validation evidence carrying the frozen design.

    Returns:
        Decision sizes, unfilled orders and every refused path; an empty
        refusal tuple means every label and every holding the design can
        need is valued.

    Raises:
        ValueError: CRSP-002/CRSP-009 for a calendar fault the panel or the
            window closes cannot pass; a security's own missing evidence is
            reported, never raised.
    """
    outcomes = index_crsp_outcomes(sources)
    validation = sources.protocol["validation_start"]
    members: list[int] = []
    unfilled: list[int] = []
    refused: list[CrspRefusedPath] = []
    ordinal = 0
    panel = crsp_known_panel(sources)
    for index, decision in enumerate(panel):
        window = decision.window
        members.append(len(decision.features))
        unfilled.append(0)
        for feature in decision.features:
            ordinal += 1
            path = crsp_resolve_window_path(outcomes, feature.permno, window)
            if isinstance(path, CrspPath):
                unfilled[-1] += 0 if path.filled else 1
            else:
                refused.append(
                    CrspRefusedPath(
                        ordinal=ordinal,
                        decision=index,
                        period="validation" if window.decision >= validation else "training",
                        reason=path.reason,
                        entry_sha256=_digest(outcomes, feature.permno, window.entry),
                        exit_sha256=_digest(outcomes, feature.permno, window.exit),
                    )
                )
    return CrspQualification(
        members=tuple(members),
        unfilled=tuple(unfilled),
        training_decisions=sum(1 for item in panel if item.window.decision < validation),
        refused=tuple(refused),
    )


def crsp_qualification_counts(value: CrspQualification) -> JSONObject:
    """Summarize a qualification for publication and for the shared record.

    Args:
        value: A complete qualification.

    Returns:
        Path and decision totals, refusals by reason and period, unfilled
        orders by period, and the decisions and rows whose full-universe
        labels a refusal leaves unknown.
    """
    split = value["training_decisions"]
    by_reason = Counter(path["reason"] for path in value["refused"])
    by_period = Counter(path["period"] for path in value["refused"])
    reasons: JSONObject = {}
    for reason, count in sorted(by_reason.items()):
        reasons[reason] = count
    periods: JSONObject = {period: by_period[period] for period in PERIODS}
    dependent = {path["decision"] for path in value["refused"]}
    return {
        "decisions": len(value["members"]),
        "paths": sum(value["members"]),
        "refused": len(value["refused"]),
        "reasons": reasons,
        "periods": periods,
        "unfilled_training": sum(value["unfilled"][:split]),
        "unfilled_validation": sum(value["unfilled"][split:]),
        "dependent_decisions": len(dependent),
        "dependent_rows": sum(value["members"][index] for index in dependent),
    }


def encode_crsp_qualification(value: CrspQualification) -> JSONObject:
    """Encode decision sizes and refused paths.

    Args:
        value: A complete qualification.

    Returns:
        Canonical document; refused paths are the published ``results``.
    """
    members: list[JSONValue] = list(value["members"])
    unfilled: list[JSONValue] = list(value["unfilled"])
    results: list[JSONValue] = []
    for path in value["refused"]:
        encoded: JSONObject = {
            "ordinal": path["ordinal"],
            "decision": path["decision"],
            "period": path["period"],
            "reason": path["reason"],
            "entry_sha256": path["entry_sha256"],
            "exit_sha256": path["exit_sha256"],
        }
        results.append(encoded)
    return {
        "members": members,
        "unfilled": unfilled,
        "training_decisions": value["training_decisions"],
        "results": results,
    }


def _optional_digest(document: JSONObject, key: str) -> str | None:
    """Require a lowercase SHA-256 or an explicit null."""
    if document[key] is None:
        return None
    digest = require_str(document, key)
    if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ValueError(f"CRSP_QUALIFICATION: {key} must be a SHA-256 or null")
    return digest


def _decode_path(document: JSONObject, decisions: int, training: int) -> CrspRefusedPath:
    """Require one refused path's exact fields, bounds and period."""
    if set(document) != set(CrspRefusedPath.__annotations__):
        raise ValueError("CRSP_QUALIFICATION: exact refused-path fields required")
    period = require_str(document, "period")
    decision = require_int(document, "decision")
    if period not in PERIODS or not 0 <= decision < decisions:
        raise ValueError("CRSP_QUALIFICATION: refused path names no panel decision or period")
    if period != ("validation" if decision >= training else "training"):
        raise ValueError("CRSP_QUALIFICATION: a refused path's period disagrees with its decision")
    return CrspRefusedPath(
        ordinal=require_int(document, "ordinal"),
        decision=decision,
        period=period,
        reason=require_nonempty(document, "reason"),
        entry_sha256=_optional_digest(document, "entry_sha256"),
        exit_sha256=_optional_digest(document, "exit_sha256"),
    )


def decode_crsp_qualification(document: JSONObject) -> CrspQualification:
    """Decode a published qualification and recheck its internal consistency.

    Args:
        document: Untrusted ``encode_crsp_qualification`` output.

    Returns:
        The qualification.

    Raises:
        ValueError: CRSP_QUALIFICATION for unknown fields, a negative decision
            size, unfilled counts that do not fit their decisions, a training
            split outside the panel, a refused path outside the panel or in
            the wrong period, more unfilled and refused paths than members,
            or ordinals that are not strictly increasing within the walk.
    """
    if set(document) != {"members", "unfilled", "training_decisions", "results"}:
        raise ValueError("CRSP_QUALIFICATION: exact qualification fields required")
    members = _sizes(document, "members")
    unfilled = _sizes(document, "unfilled")
    if any(size < 0 for size in members):
        raise ValueError("CRSP_QUALIFICATION: decision sizes must be nonnegative")
    if len(unfilled) != len(members) or any(
        not 0 <= count <= size for count, size in zip(unfilled, members, strict=True)
    ):
        raise ValueError("CRSP_QUALIFICATION: unfilled counts must fit their decisions")
    training = require_int(document, "training_decisions")
    if not 0 <= training <= len(members):
        raise ValueError("CRSP_QUALIFICATION: the training split lies outside the panel")
    refused = tuple(
        _decode_path(narrow_json_to_dict(item), len(members), training)
        for item in require_list(document, "results")
    )
    starts = [sum(members[:index]) for index in range(len(members))]
    for path in refused:
        first = starts[path["decision"]]
        if not first < path["ordinal"] <= first + members[path["decision"]]:
            raise ValueError("CRSP_QUALIFICATION: an ordinal lies outside its decision's walk")
    if any(later["ordinal"] <= earlier["ordinal"] for earlier, later in pairwise(refused)):
        raise ValueError("CRSP_QUALIFICATION: ordinals must strictly increase")
    refusals = Counter(path["decision"] for path in refused)
    if any(unfilled[index] + count > members[index] for index, count in refusals.items()):
        raise ValueError("CRSP_QUALIFICATION: a decision has more unfilled and refused paths")
    return CrspQualification(
        members=members, unfilled=unfilled, training_decisions=training, refused=refused
    )


def _sizes(document: JSONObject, key: str) -> tuple[int, ...]:
    """Require a list of integers, one per decision."""
    return tuple(require_int({key: size}, key) for size in require_list(document, key))


__all__ = [
    "CrspQualification",
    "CrspRefusedPath",
    "crsp_qualification_counts",
    "decode_crsp_qualification",
    "encode_crsp_qualification",
    "qualify_crsp_panel",
]
