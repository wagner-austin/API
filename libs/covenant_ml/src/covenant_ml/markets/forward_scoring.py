"""Score a frozen forward plan from a campaign's saved captures and settlements.

The campaign evaluator verifies every acquisition and replays every settlement
from its archived exchange responses. This module adds only what the forward plan
declares: it re-derives each scored capture's quotes and predictions from that
capture's archived responses and requires them to equal the sealed report, then
compares the plan's model arm with its market reference on the declared dates
alone, and runs the existing quoted taker policy over the same contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
    require_str_list,
)
from platform_core.minimum_detectable_effect import MIN_REPLICATES, paired_continuous_power
from platform_core.power_records import encode_paired_continuous_power
from platform_core.power_types import PairedContinuousPower
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.codec_records import encode_interval
from ..evaluation.paired import paired_mean_difference
from ..evaluation.resampling import build_resampling
from ..evaluation.types import IntervalEstimate, ResamplingUnit
from ..metrics import brier_per_row
from . import _test_hooks
from .archive import ResponseArchive
from .campaign_evaluation import evaluate_campaign
from .campaign_inputs import load_campaign_inputs
from .campaign_records import Campaign
from .evidence_location import locate_evidence
from .forecast_model import float_values
from .forecast_policy import quoted_policy
from .forward_plan import ForwardPlan, encode_forward_plan
from .paper_collection import PaperReader, collect_paper_predictions
from .paper_records import PaperPlan
from .reporting import read_verified_market_report
from .settlement_records import SettlementState, decode_settlement
from .types import DOLLAR
from .validation import read_document


class ForwardContract(TypedDict):
    """One eligible contract: its archived quote, final outcome and both saved arms."""

    ticker: ReadOnly[str]
    series: ReadOnly[str]
    event_day: ReadOnly[str]
    bid: ReadOnly[int]
    ask: ReadOnly[int]
    yes: ReadOnly[int]
    model: ReadOnly[float]
    reference: ReadOnly[float]


def encode_forward_contract(row: ForwardContract) -> JSONObject:
    """Encode a scored contract for the report.

    Args:
        row: Validated contract.

    Returns:
        JSON object.
    """
    return {
        "ticker": row["ticker"],
        "series": row["series"],
        "event_day": row["event_day"],
        "bid": row["bid"],
        "ask": row["ask"],
        "yes": row["yes"],
        "model": row["model"],
        "reference": row["reference"],
    }


def decode_forward_contract(document: JSONObject) -> ForwardContract:
    """Read a scored contract back from a report.

    Args:
        document: Encoded contract.

    Returns:
        Validated contract.

    Raises:
        ValueError: FORWARD-006 for an outcome other than 0 or 1, a crossed or
            out-of-range quote, or a probability outside [0, 1].
    """
    bid = require_int(document, "bid")
    ask = require_int(document, "ask")
    yes = require_int(document, "yes")
    model = require_float(document, "model")
    reference = require_float(document, "reference")
    if (
        yes not in (0, 1)
        or not 0 <= bid <= ask <= DOLLAR
        or not 0 <= model <= 1
        or not 0 <= reference <= 1
    ):
        raise ValueError("[FORWARD-006] contract needs a binary outcome, a quote and probabilities")
    return ForwardContract(
        ticker=require_str(document, "ticker"),
        series=require_str(document, "series"),
        event_day=require_str(document, "event_day"),
        bid=bid,
        ask=ask,
        yes=yes,
        model=model,
        reference=reference,
    )


def replay_capture(
    root: Path,
    campaign: Campaign,
    plan: PaperPlan,
    market_archive: Callable[[Path], ResponseArchive],
    forecast_archive: Callable[[Path], ResponseArchive],
) -> tuple[JSONObject, tuple[Path, ...]]:
    """Re-derive one capture from its archived responses and require the sealed report.

    Args:
        root: Campaign directory.
        campaign: Verified frozen manifest.
        plan: The capture's frozen plan.
        market_archive: Opens an archive of exchange responses, offline.
        forecast_archive: Opens an archive of forecast responses, offline.

    Returns:
        The verified saved capture report and every path read for it.

    Raises:
        ValueError: FORWARD-005 when the replayed quotes or predictions differ
            from the saved report; report verification errors propagate.
        OSError: An archived response or the fitted report is absent.
    """
    path = root / "reports" / (plan["name"] + "-capture.json")
    saved, sources, _ = read_verified_market_report(path, "FORWARD-005")
    relocations = _test_hooks.evidence_relocations()
    model_path = locate_evidence(
        next(
            model["path"] for model in campaign["models"] if model["hour"] == plan["forecast_hour"]
        ),
        relocations,
    )
    archive = root / "archives" / (plan["name"] + "-capture")
    replay = collect_paper_predictions(
        plan,
        read_document(model_path),
        PaperReader(market_archive(archive / "markets"), plan, False),
        PaperReader(forecast_archive(archive / "forecasts"), plan, False),
    )
    replay["implementation_sha256"] = require_str(saved, "implementation_sha256")
    core = {
        key: value
        for key, value in saved.items()
        if key not in ("recorded_at", "sources", "fingerprint")
    }
    if replay != core:
        raise ValueError("[FORWARD-005] archived responses do not reproduce the saved capture")
    return saved, (*sources, model_path)


def eligible_contracts(
    saved: JSONObject, outcomes: JSONObject, plan: ForwardPlan, series: str
) -> tuple[ForwardContract, ...]:
    """Join a capture's saved quotes and arms with its binary unchanged-rule outcomes.

    Args:
        saved: Verified capture report.
        outcomes: Replayed settlement snapshot with no pending contract.
        plan: Forward plan naming the model and reference arms.
        series: The capture's series.

    Returns:
        Contracts in capture order; scalar and rule-changed contracts are left out.

    Raises:
        ValueError: FORWARD-004 when the capture's model features differ from the
            plan's. The snapshot is replayed from the capture's own predictions, so
            every settled ticker is a captured one.
    """
    features = tuple(
        require_str_list(
            require_dict(require_dict(saved, "models"), plan["model"]["arm"]), "features"
        )
    )
    if features != plan["model"]["features"]:
        raise ValueError("[FORWARD-004] capture's model features differ from the plan's")
    predictions = {
        require_str(prediction, "ticker"): prediction
        for prediction in (
            narrow_json_to_dict(value) for value in require_list(saved, "predictions")
        )
    }
    rows: list[ForwardContract] = []
    for value in require_list(outcomes, "settlements"):
        settlement = decode_settlement(narrow_json_to_dict(value))
        if settlement["state"] is SettlementState.BINARY and not settlement["rules_changed"]:
            prediction = predictions[settlement["ticker"]]
            book = require_dict(prediction, "inputs")
            rows.append(
                decode_forward_contract(
                    {
                        "ticker": settlement["ticker"],
                        "series": series,
                        "event_day": require_str(require_dict(saved, "plan"), "event_day"),
                        "bid": require_int(book, "bid"),
                        "ask": require_int(book, "ask"),
                        "yes": int(settlement["payout"] == DOLLAR),
                        "model": require_float(prediction, plan["model"]["arm"]),
                        "reference": require_float(prediction, plan["reference_arm"]),
                    }
                )
            )
    return tuple(rows)


def mean_brier(rows: Sequence[ForwardContract], probabilities: Sequence[float]) -> float:
    """Average one arm's squared error over contracts.

    Args:
        rows: Nonempty eligible contracts.
        probabilities: That arm's probability for each contract, in the same order.

    Returns:
        Mean Brier score.
    """
    outcomes: NDArray[np.int64] = np.asarray([row["yes"] for row in rows], dtype=np.int64)
    predicted: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
    return sum(float_values(brier_per_row(outcomes, predicted))) / len(rows)


def compare_dates(
    plan: ForwardPlan, differences: Sequence[float]
) -> tuple[IntervalEstimate, PairedContinuousPower]:
    """Interval and minimum detectable effect of the equal-date mean difference.

    Args:
        plan: Declared confidence, alpha and smallest effect of interest.
        differences: One paired Brier difference per scored date, at least
            ``MIN_REPLICATES`` of them.

    Returns:
        The t interval with one date as one unit, and the paired power record.
    """
    values: NDArray[np.float64] = np.asarray(differences, dtype=np.float64)
    groups: NDArray[np.int64] = np.zeros(len(differences), dtype=np.int64)
    units = build_resampling(ResamplingUnit.ROW, len(differences), groups, 1)
    return paired_mean_difference(values, units, plan["confidence"]), paired_continuous_power(
        differences, plan["alpha"], plan["smallest_effect_of_interest"]
    )


def score_forward_plan(
    plan: ForwardPlan,
    plan_sha256: str,
    root: Path,
    observed_at: str,
    market_archive: Callable[[Path], ResponseArchive],
    forecast_archive: Callable[[Path], ResponseArchive],
) -> tuple[JSONObject, tuple[Path, ...]]:
    """Score exactly the plan's declared dates from saved campaign evidence.

    Args:
        plan: Frozen plan, already verified against its pinned digest.
        plan_sha256: That digest, recorded in the report.
        root: Campaign directory.
        observed_at: Explicit analysis observation timestamp.
        market_archive: Opens an archive of exchange responses, offline.
        forecast_archive: Opens an archive of forecast responses, offline.

    Returns:
        Per-date dispositions and scores, the comparison, the trades, the decision
        criteria and every scored contract, with all evidence paths read.

    Raises:
        ValueError: FORWARD-004 when the campaign is not the one the plan binds or
            lacks a declared plan; FORWARD-005 when a capture does not reproduce;
            campaign evaluator errors propagate.
        OSError: Required evidence is absent.
    """
    campaign = load_campaign_inputs(root, plan["campaign"]["manifest_sha256"])
    models = {model["hour"]: model["digest"] for model in campaign["models"]}
    if (
        campaign["name"] != plan["campaign"]["name"]
        or campaign["implementation_digest"] != plan["campaign"]["collector_implementation_sha256"]
        or campaign["protocol_digest"] != plan["campaign"]["protocol_sha256"]
        or models.get(plan["model"]["forecast_hour"]) != plan["model"]["report_sha256"]
    ):
        raise ValueError("[FORWARD-004] campaign identity differs from the plan's")
    evaluation, evidence = evaluate_campaign(
        root, plan["campaign"]["manifest_sha256"], observed_at, market_archive
    )
    rows = {
        require_str(row, "name"): row
        for row in (narrow_json_to_dict(value) for value in require_list(evaluation, "plans"))
    }
    sources = set(evidence)
    dates: list[JSONValue] = []
    differences: list[float] = []
    contracts: list[ForwardContract] = []
    for day in plan["event_dates"]:
        entries: list[JSONValue] = []
        scored: list[ForwardContract] = []
        for series in plan["series"]:
            frozen = [
                candidate
                for candidate in campaign["plans"]
                if (candidate["event_day"], candidate["series"], candidate["forecast_hour"])
                == (day, series, plan["model"]["forecast_hour"])
            ]
            if len(frozen) != 1:
                raise ValueError(f"[FORWARD-004] campaign holds no single plan for {day} {series}")
            row = rows[frozen[0]["name"]]
            outcomes = row["outcomes"]
            found: tuple[ForwardContract, ...] = ()
            if outcomes is not None and require_bool(narrow_json_to_dict(outcomes), "complete"):
                saved, paths = replay_capture(
                    root, campaign, frozen[0], market_archive, forecast_archive
                )
                sources.update(paths)
                found = eligible_contracts(saved, narrow_json_to_dict(outcomes), plan, series)
            scored.extend(found)
            entries.append(
                {
                    "name": frozen[0]["name"],
                    "series": series,
                    "capture": row["capture"],
                    "settlement": row["settlement"],
                    "eligible": len(found),
                }
            )
        difference: float | None = None
        model: float | None = None
        reference: float | None = None
        if scored:
            model = mean_brier(scored, [contract["model"] for contract in scored])
            reference = mean_brier(scored, [contract["reference"] for contract in scored])
            difference = model - reference
            differences.append(difference)
            contracts.extend(scored)
        dates.append(
            {
                "event_day": day,
                "plans": entries,
                "contracts": len(scored),
                "model_brier": model,
                "reference_brier": reference,
                "paired_brier_difference": difference,
            }
        )
    return {
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "plan": encode_forward_plan(plan),
        "observed_at": observed_at,
        "declared_dates": len(plan["event_dates"]),
        "scored_dates": len(differences),
        "eligible_contracts": len(contracts),
        "dates": dates,
        **decide(plan, differences, contracts),
        "contracts": [encode_forward_contract(row) for row in contracts],
        "interpretation": (
            "Forward paper evidence on dates declared before their outcomes existed. "
            "Trades are hypothetical fills at the archived quote; depth, queue position, "
            "latency and actual fees are unverified, and no order was placed."
        ),
    }, tuple(sorted(sources))


def decide(
    plan: ForwardPlan, differences: Sequence[float], contracts: Sequence[ForwardContract]
) -> JSONObject:
    """Apply the declared decision rule, failing every criterion without enough dates.

    Args:
        plan: Declared statistics.
        differences: One paired Brier difference per scored date, in date order.
        contracts: Every eligible contract of those dates.

    Returns:
        ``mean_paired_brier_difference``, ``comparison``, ``policy`` and ``criteria``;
        the first three are null with fewer than ``MIN_REPLICATES`` scored dates.
    """
    if len(differences) < MIN_REPLICATES:
        return {
            "mean_paired_brier_difference": None,
            "comparison": None,
            "policy": None,
            "criteria": {
                "mean_below_zero": False,
                "upper_bound_below_zero": False,
                "stressed_pnl_positive": False,
                "forward_lead": False,
                "reason": f"fewer than {MIN_REPLICATES} scored dates",
            },
        }
    mean = sum(differences) / len(differences)
    interval, power = compare_dates(plan, differences)
    policy = quoted_policy(contracts, [row["model"] for row in contracts])
    below = mean < 0
    upper = interval["ci_high"] < 0
    stressed = require_int(policy, "extra_cent_pnl") > 0
    return {
        "mean_paired_brier_difference": mean,
        "comparison": {
            "interval": encode_interval(interval),
            "power": encode_paired_continuous_power(power),
        },
        "policy": policy,
        "criteria": {
            "mean_below_zero": below,
            "upper_bound_below_zero": upper,
            "stressed_pnl_positive": stressed,
            "forward_lead": below and upper and stressed,
            "reason": "scored",
        },
    }


__all__ = [
    "ForwardContract",
    "compare_dates",
    "decide",
    "decode_forward_contract",
    "eligible_contracts",
    "encode_forward_contract",
    "mean_brier",
    "replay_capture",
    "score_forward_plan",
]
