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
    CrspPathRefusal,
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
        refused: Every refused path, in walk order.
    """

    members: ReadOnly[tuple[int, ...]]
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
        Decision sizes and every refused path; an empty refusal tuple means
        every label and every holding the design can need is valued.

    Raises:
        ValueError: CRSP-002/CRSP-009 for a calendar fault the panel or the
            window closes cannot pass; a security's own missing evidence is
            reported, never raised.
    """
    outcomes = index_crsp_outcomes(sources)
    validation = sources.protocol["validation_start"]
    members: list[int] = []
    refused: list[CrspRefusedPath] = []
    ordinal = 0
    for index, decision in enumerate(crsp_known_panel(sources)):
        window = decision.window
        members.append(len(decision.features))
        for feature in decision.features:
            ordinal += 1
            path = crsp_resolve_window_path(outcomes, feature.permno, window)
            if isinstance(path, CrspPathRefusal):
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
    return CrspQualification(members=tuple(members), refused=tuple(refused))


def crsp_qualification_counts(value: CrspQualification) -> JSONObject:
    """Summarize a qualification for publication and for the shared record.

    Args:
        value: A complete qualification.

    Returns:
        Path and decision totals, refusals by reason and period, and the
        decisions and rows whose full-universe labels a refusal leaves unknown.
    """
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
    return {"members": members, "results": results}


def _optional_digest(document: JSONObject, key: str) -> str | None:
    """Require a lowercase SHA-256 or an explicit null."""
    if document[key] is None:
        return None
    digest = require_str(document, key)
    if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        raise ValueError(f"CRSP_QUALIFICATION: {key} must be a SHA-256 or null")
    return digest


def _decode_path(document: JSONObject, decisions: int) -> CrspRefusedPath:
    """Require one refused path's exact fields and bounds."""
    if set(document) != set(CrspRefusedPath.__annotations__):
        raise ValueError("CRSP_QUALIFICATION: exact refused-path fields required")
    period = require_str(document, "period")
    decision = require_int(document, "decision")
    if period not in PERIODS or not 0 <= decision < decisions:
        raise ValueError("CRSP_QUALIFICATION: refused path names no panel decision or period")
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
            size, a refused path outside the panel, or ordinals that are not
            strictly increasing within the walk.
    """
    if set(document) != {"members", "results"}:
        raise ValueError("CRSP_QUALIFICATION: exact qualification fields required")
    sizes = require_list(document, "members")
    members = tuple(require_int({"size": size}, "size") for size in sizes)
    if any(size < 0 for size in members):
        raise ValueError("CRSP_QUALIFICATION: decision sizes must be nonnegative")
    refused = tuple(
        _decode_path(narrow_json_to_dict(item), len(members))
        for item in require_list(document, "results")
    )
    starts = [sum(members[:index]) for index in range(len(members))]
    for path in refused:
        first = starts[path["decision"]]
        if not first < path["ordinal"] <= first + members[path["decision"]]:
            raise ValueError("CRSP_QUALIFICATION: an ordinal lies outside its decision's walk")
    if any(later["ordinal"] <= earlier["ordinal"] for earlier, later in pairwise(refused)):
        raise ValueError("CRSP_QUALIFICATION: ordinals must strictly increase")
    return CrspQualification(members=members, refused=refused)


__all__ = [
    "CrspQualification",
    "CrspRefusedPath",
    "crsp_qualification_counts",
    "decode_crsp_qualification",
    "encode_crsp_qualification",
    "qualify_crsp_panel",
]
