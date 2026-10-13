"""Read frozen campaign attempts without running, repairing or retrying acquisition."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, require_int, require_str

from .archive import ResponseArchive
from .campaign_inputs import acquisition_window
from .campaign_records import AttemptState, Campaign, decode_attempt
from .paper_records import PaperPlan
from .paper_scoring import reconcile_predictions
from .reporting import read_verified_market_report
from .settlement_records import SavedPrediction
from .validation import parse_timestamp, read_document


def read_campaign_attempt(
    root: Path,
    campaign: Campaign,
    digest: str,
    plan: PaperPlan,
    operation: str,
    observed_ms: int,
) -> tuple[str, JSONObject | None, tuple[Path, ...]]:
    """Verify terminal evidence or explicitly classify an unfinished window.

    Args:
        root: Campaign evidence directory.
        campaign: Verified frozen manifest.
        digest: Externally pinned manifest identity.
        plan: Expected acquisition plan.
        operation: Capture or settlement operation.
        observed_ms: Analysis observation timestamp in epoch milliseconds.

    Returns:
        Disposition, verified completed report if present, and evidence paths.

    Raises:
        ValueError: CAMPAIGN-011 for changed attempt identity or future evidence;
            CAMPAIGN-012 for changed report or collector identity.
        OSError: Required saved evidence cannot be read.
    """
    name = f"{plan['name']}-{operation}"
    path = root / "attempts" / (name + ".json")
    if not path.exists():
        marker = root / "attempts" / (name + ".started.json")
        if marker.exists():
            started = read_document(marker)
            if (
                require_str(started, "campaign_digest") != digest
                or parse_timestamp(require_str(started, "started_at")) > observed_ms
            ):
                raise ValueError("[CAMPAIGN-011] invalid acquisition start evidence")
            return "started_unfinished", None, (marker,)
        start, end = acquisition_window(plan, operation)
        state = (
            "future"
            if observed_ms < start
            else "unrecorded_overdue"
            if observed_ms > end
            else "due"
        )
        return state, None, ()
    attempt = decode_attempt(read_document(path))
    if (
        attempt["name"] != plan["name"]
        or attempt["operation"] != operation
        or parse_timestamp(attempt["observed_at"]) > observed_ms
    ):
        raise ValueError("[CAMPAIGN-011] attempt identity or observation time differs")
    if attempt["state"] is not AttemptState.COMPLETE:
        return attempt["state"].value, None, (path,)
    report, sources, report_digest = read_verified_market_report(
        root / "reports" / (name + ".json"), "CAMPAIGN-012"
    )
    if (
        report_digest != attempt["report_digest"]
        or require_str(report, "implementation_sha256") != campaign["implementation_digest"]
    ):
        raise ValueError("[CAMPAIGN-012] completed report or frozen collector identity changed")
    return attempt["state"].value, report, (path, *sources)


def replay_campaign_outcomes(
    settlement: JSONObject,
    predictions_path: Path,
    predictions: tuple[SavedPrediction, ...],
    archive: ResponseArchive,
    observed_ms: int,
) -> tuple[JSONObject, tuple[Path, ...]]:
    """Verify settlement binding and reproduce every reported outcome from native bytes.

    Args:
        settlement: Provenance-verified settlement report.
        predictions_path: Original sealed prediction report.
        predictions: Validated original predictions.
        archive: Offline native settlement archive.
        observed_ms: Analysis observation timestamp.

    Returns:
        Replayed report and the exact archive paths read.

    Raises:
        ValueError: CAMPAIGN-013 for changed prediction binding, CAMPAIGN-011 for
            future receipts, or CAMPAIGN-014 for disagreement with native replay.
        OSError: Required archived bytes are unavailable.
    """
    if (
        require_str(settlement, "prediction_report_sha256")
        != sha256(predictions_path.read_bytes()).hexdigest()
    ):
        raise ValueError("[CAMPAIGN-013] settlement refers to different predictions")
    replay = reconcile_predictions(predictions, archive, False)
    if require_int(replay, "last_received_ms") > observed_ms:
        raise ValueError("[CAMPAIGN-011] settlement is later than analysis observation")
    saved: JSONObject = {
        key: value
        for key, value in settlement.items()
        if key
        not in ("prediction_report_sha256", "implementation_sha256", "sources", "fingerprint")
    }
    if saved != replay:
        raise ValueError("[CAMPAIGN-014] settlement report differs from archived outcomes")
    return replay, tuple(
        archive.file_for(archive.request_url("/markets/" + prediction["ticker"], {}))
        for prediction in predictions
    )


__all__ = ["read_campaign_attempt", "replay_campaign_outcomes"]
