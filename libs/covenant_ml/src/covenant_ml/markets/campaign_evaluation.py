"""Offline prospective study accounting and equal-event-date probability comparisons."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)

from .archive import ResponseArchive
from .campaign_evidence import read_campaign_attempt, replay_campaign_outcomes
from .campaign_inputs import load_campaign_inputs
from .paper_records import decode_paper_plan
from .paper_scoring import load_verified_predictions
from .settlement_records import ARMS, SavedPrediction
from .validation import parse_timestamp


def summarize_dates(rows: Sequence[tuple[str, int, JSONObject]]) -> JSONObject:
    """Weight contracts within dates, then weight dates equally for all four arms.

    Args:
        rows: Event date, eligible contract count and replayed proper-score objects.

    Returns:
        Per-date scores, counts and descriptive paired Brier contrast. No inference
        or profitability claim is produced, including for an incomplete study.
    """
    dates: list[JSONValue] = []
    contrasts: list[float] = []
    arm_dates: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for day in sorted({row[0] for row in rows}):
        selected = [(count, scores) for event, count, scores in rows if event == day]
        contracts = sum(count for count, _ in selected)
        scores: JSONObject = {}
        for arm in ARMS:
            score = (
                sum(
                    count * require_float(require_dict(values, arm), "brier")
                    for count, values in selected
                )
                / contracts
            )
            scores[arm] = score
            arm_dates[arm].append(score)
        contrast = require_float(scores, "market_plus_forecast") - require_float(
            scores, "market_calibrated"
        )
        contrasts.append(contrast)
        dates.append(
            {
                "event_day": day,
                "contracts": contracts,
                "brier": scores,
                "paired_brier_difference": contrast,
            }
        )
    return {
        "eligible_dates": len(dates),
        "eligible_contracts": sum(row[1] for row in rows),
        "dates": dates,
        "mean_brier": {arm: sum(values) / len(values) for arm, values in arm_dates.items()}
        if dates
        else None,
        "paired_brier_difference": sum(contrasts) / len(contrasts) if contrasts else None,
    }


def evaluate_campaign(
    root: Path,
    digest: str,
    observed_at: str,
    archive_factory: Callable[[Path], ResponseArchive],
) -> tuple[JSONObject, tuple[Path, ...]]:
    """Verify saved acquisitions and replay settlements without any source requests.

    Args:
        root: Original campaign directory.
        digest: Externally pinned manifest digest.
        observed_at: Explicit analysis observation timestamp.
        archive_factory: Creates an archive reader for the saved market origin.

    Returns:
        Descriptive study report and all original evidence paths for provenance.

    Raises:
        ValueError: CAMPAIGN-013 for changed plan or settlement prediction binding;
            CAMPAIGN-014 when a saved settlement differs from offline replay.
        OSError: Required evidence is missing; no acquisition is attempted.
    """
    campaign = load_campaign_inputs(root, digest)
    observed = parse_timestamp(observed_at)
    sources = {root / "campaign.json", Path(campaign["protocol_path"])}
    sources.update(Path(model["path"]) for model in campaign["models"])
    rows: list[JSONValue] = []
    counts: dict[str, int] = {}
    scoring: dict[int, list[tuple[str, int, JSONObject]]] = {0: [], 12: []}
    for plan in campaign["plans"]:
        sources.add(root / "plans" / (plan["name"] + ".json"))
        capture_state, capture, capture_paths = read_campaign_attempt(
            root, campaign, digest, plan, "capture", observed
        )
        state, settlement, settlement_paths = read_campaign_attempt(
            root, campaign, digest, plan, "settlement", observed
        )
        sources.update((*capture_paths, *settlement_paths))
        for disposition in (capture_state, state):
            counts[disposition] = counts.get(disposition, 0) + 1
        row: JSONObject = {
            "name": plan["name"],
            "event_day": plan["event_day"],
            "forecast_hour": plan["forecast_hour"],
            "series": plan["series"],
            "capture": capture_state,
            "settlement": state,
            "capture_exclusions": None,
            "outcomes": None,
        }
        predictions_path = root / "reports" / (plan["name"] + "-capture.json")
        predictions: tuple[SavedPrediction, ...] = ()
        if capture is not None:
            if decode_paper_plan(require_dict(capture, "plan")) != plan:
                raise ValueError("[CAMPAIGN-013] capture does not match frozen plan")
            predictions, _ = load_verified_predictions(predictions_path)
            if parse_timestamp(require_str(capture, "recorded_at")) > observed:
                raise ValueError("[CAMPAIGN-011] capture is later than analysis observation")
            row["capture_exclusions"] = require_list(capture, "exclusions")
        if settlement is not None:
            if capture is None:
                raise ValueError("[CAMPAIGN-013] settlement lacks completed capture")
            archive = archive_factory(root / "archives" / (plan["name"] + "-settlement"))
            replay, replay_sources = replay_campaign_outcomes(
                settlement, predictions_path, predictions, archive, observed
            )
            sources.update(replay_sources)
            row["outcomes"] = replay
            if replay["scores"] is not None:
                scoring[plan["forecast_hour"]].append(
                    (
                        plan["event_day"],
                        require_int(replay, "eligible_binary"),
                        require_dict(replay, "scores"),
                    )
                )
        rows.append(row)
    return {
        "schema_version": 1,
        "campaign": campaign["name"],
        "manifest_sha256": digest,
        "collector_implementation_sha256": campaign["implementation_digest"],
        "observed_at": observed_at,
        "acquisitions": dict(counts),
        "plans": rows,
        "primary_12utc": summarize_dates(scoring[12]),
        "secondary_00utc": summarize_dates(scoring[0]),
        "interpretation": (
            "Descriptive available-case probability scores; dates have equal weight. "
            "Negative paired Brier differences favor forecast augmentation. "
            "Missing and pending observations remain explicit. "
            "No significance, fills, returns or proven edge are inferred."
        ),
    }, tuple(sorted(sources))


__all__ = ["evaluate_campaign", "summarize_dates"]
