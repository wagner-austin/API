"""Verify sealed predictions and score only fully reconciled finalized outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_int,
    require_list,
    require_str,
)

from ..metrics import brier_per_row, log_loss_per_row
from .archive import ResponseArchive, decode_capture
from .forecast_model import float_values
from .paper_records import decode_paper_plan
from .reporting import read_verified_market_report
from .settlement_records import (
    ARMS,
    SavedPrediction,
    SettlementState,
    decode_native_settlement,
    decode_saved_prediction,
    encode_settlement,
)
from .types import DOLLAR
from .validation import parse_timestamp


def load_verified_predictions(path: Path) -> tuple[tuple[SavedPrediction, ...], tuple[Path, ...]]:
    """Verify report, provenance, every source file and the original sealed prediction bytes.

    Args:
        path: Original prospective report, beside its RunRecord sidecar.

    Returns:
        Immutable saved predictions and all verified source paths.

    Raises:
        ValueError: SETTLEMENT-004 for changed evidence; SETTLEMENT-005 for
            malformed prediction timing, schema or identity.
        OSError: Required evidence is unavailable; it is never refetched.
        TypeError: Required native fields have incorrect types.
    """
    report, sources, _ = read_verified_market_report(path, "SETTLEMENT-004")
    seals = [source for source in sources[2:] if source.name == "prediction.json"]
    if len(seals) != 1:
        raise ValueError("[SETTLEMENT-004] require exactly one original prediction seal")
    seal = decode_capture(narrow_json_to_dict(load_json_str(seals[0].read_text("utf-8"))))
    core: JSONObject = {
        key: value
        for key, value in report.items()
        if key not in ("sources", "fingerprint", "recorded_at")
    }
    plan = decode_paper_plan(require_dict(core, "plan"))
    if (
        seal["body"] != dump_json_str(core)
        or seal["captured_at"] != require_str(report, "recorded_at")
        or seal["url"] != "paper://" + plan["name"]
    ):
        raise ValueError("[SETTLEMENT-004] report differs from original sealed predictions")
    recorded = parse_timestamp(seal["captured_at"])
    latest = require_int(core, "last_received_ms")
    if (
        require_int(core, "schema_version") != 1
        or require_int(core, "orders_submitted") != 0
        or not parse_timestamp(plan["window_start"])
        <= latest
        <= recorded
        <= parse_timestamp(plan["window_end"])
        or recorded - latest > 60000
    ):
        raise ValueError("[SETTLEMENT-005] invalid original prediction schema or chronology")
    predictions: list[SavedPrediction] = []
    for value in require_list(core, "predictions"):
        native = narrow_json_to_dict(value)
        predictions.append(
            decode_saved_prediction(
                {
                    "ticker": native["ticker"],
                    "series": plan["series"],
                    "event_day": plan["event_day"],
                    "recorded_ms": recorded,
                    "rules_primary": native["rules_primary"],
                    "probabilities": [native[arm] for arm in ARMS],
                }
            )
        )
    if len({row["ticker"] for row in predictions}) != len(predictions):
        raise ValueError("[SETTLEMENT-005] duplicate saved prediction identity")
    return tuple(predictions), tuple(sources)


def reconcile_predictions(
    predictions: Sequence[SavedPrediction],
    archive: ResponseArchive,
    capture: bool,
) -> JSONObject:
    """Join native final cash outcomes without treating provisional results as labels.

    Args:
        predictions: Hash-verified frozen predictions from one prospective report.
        archive: New settlement archive or existing evidence for explicit replay.
        capture: True for fresh acquisition, False for offline replay.

    Returns:
        Settlement states, exclusions and aggregate proper scores only after all
        selected contracts finalize; aggregate scores otherwise remain null.

    Raises:
        ValueError: Invalid native evidence or SETTLEMENT-006 for clock regression.
        HTTPStatusError: Source acquisition failed without retry or substitution.
        OSError: An archive operation failed.
    """
    rows: list[JSONValue] = []
    labels: list[int] = []
    probabilities: dict[str, list[float]] = {arm: [] for arm in ARMS}
    pending, scalar, changed = 0, 0, 0
    latest = 0
    for prediction in predictions:
        path = "/markets/" + prediction["ticker"]
        if capture:
            archive.capture(path, {})
        envelope = archive.load_capture(path, {})
        received = parse_timestamp(envelope["captured_at"])
        if received < latest:
            raise ValueError("[SETTLEMENT-006] settlement receipt clock regressed")
        latest = received
        document = require_dict(narrow_json_to_dict(load_json_str(envelope["body"])), "market")
        settlement = decode_native_settlement(document, prediction, received)
        rows.append(encode_settlement(settlement))
        pending += int(settlement["state"] is SettlementState.PENDING)
        scalar += int(settlement["state"] is SettlementState.SCALAR)
        changed += int(settlement["rules_changed"])
        if settlement["state"] is SettlementState.BINARY and not settlement["rules_changed"]:
            labels.append(int(settlement["payout"] == DOLLAR))
            for arm, probability in zip(ARMS, prediction["probabilities"], strict=True):
                probabilities[arm].append(probability)
    scores: JSONObject | None = None
    if pending == 0 and labels:
        scores = {}
        y: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
        for arm in ARMS:
            p: NDArray[np.float64] = np.asarray(probabilities[arm], dtype=np.float64)
            scores[arm] = {
                "brier": sum(float_values(brier_per_row(y, p))) / len(labels),
                "log_loss": sum(float_values(log_loss_per_row(y, p))) / len(labels),
            }
    return {
        "schema_version": 1,
        "settlements": rows,
        "selected": len(predictions),
        "pending": pending,
        "scalar": scalar,
        "rules_changed": changed,
        "eligible_binary": len(labels),
        "complete": pending == 0,
        "last_received_ms": latest,
        "scores": scores,
        "interpretation": (
            "Finalized binary probability scores only; no fills or returns measured."
        ),
    }


__all__ = ["load_verified_predictions", "reconcile_predictions"]
