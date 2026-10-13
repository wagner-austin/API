"""Real HTTP payout reconciliation with exact proper scores and pending gates."""

from __future__ import annotations

import math
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from platform_core.json_utils import narrow_json_to_dict, require_dict, require_list

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.paper_scoring import reconcile_predictions
from covenant_ml.markets.settlement_records import ARMS

from ._server import ApiServer, api_server
from .test_settlement_records import final_market, saved_prediction

__all__ = ["api_server"]


@pytest.mark.parametrize("kind", ["binary", "pending", "scalar", "changed"])
def test_actual_http_reconciliation_and_replay(
    api_server: ApiServer, tmp_path: Path, kind: str
) -> None:
    predictions = (saved_prediction(1), saved_prediction(0))
    for index, prediction in enumerate(predictions):
        document = final_market(1 - index)
        if index == 1:
            if kind == "pending":
                document["status"] = "determined"
            elif kind == "scalar":
                document.update(result="scalar", settlement_value_dollars="0.0100")
            elif kind == "changed":
                document["rules_primary"] = "Different terms"
        api_server.documents["/markets/" + prediction["ticker"]] = [{"market": document}]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        report = reconcile_predictions(predictions, archive, True)
        assert reconcile_predictions(predictions, archive, False) == report
    assert len(api_server.requests) == 2
    assert report["selected"] == 2
    assert report["pending"] == int(kind == "pending")
    assert report["scalar"] == int(kind == "scalar")
    assert report["rules_changed"] == int(kind == "changed")
    assert report["eligible_binary"] == (2 if kind == "binary" else 1)
    assert report["complete"] == (kind != "pending")
    if kind == "pending":
        assert report["scores"] is None
    else:
        scores = require_dict(report, "scores")
        for arm, p in zip(ARMS, predictions[0]["probabilities"], strict=True):
            expected_brier = ((1 - p) ** 2 + p**2) / 2 if kind == "binary" else (1 - p) ** 2
            expected_loss = (
                (-math.log(p) - math.log(1 - p)) / 2 if kind == "binary" else -math.log(p)
            )
            assert require_dict(scores, arm)["brier"] == pytest.approx(expected_brier, abs=1e-15)
            assert require_dict(scores, arm)["log_loss"] == pytest.approx(expected_loss, abs=1e-15)
    final = narrow_json_to_dict(require_list(report, "settlements")[1])
    assert final["payout"] == (None if kind == "pending" else 100 if kind == "scalar" else 0)


def test_empty_panel_has_no_scores(api_server: ApiServer, tmp_path: Path) -> None:
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        report = reconcile_predictions((), archive, False)
    assert report["selected"] == 0
    assert report["scores"] is None
    assert report["settlements"] == []
    assert report["complete"] is True
    assert api_server.requests == []


def test_real_http_failure_propagates(api_server: ApiServer, tmp_path: Path) -> None:
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(httpx.HTTPStatusError, match="404"):
            reconcile_predictions((saved_prediction(),), archive, True)
    assert api_server.requests == ["/markets/KXHIGHTBOS-26APR03-B60.5?"]
    assert list(tmp_path.glob("*.json")) == []


def test_receipt_clock_regression(api_server: ApiServer, tmp_path: Path) -> None:
    predictions = (saved_prediction(1), saved_prediction(0))
    for index, prediction in enumerate(predictions):
        api_server.documents["/markets/" + prediction["ticker"]] = [
            {"market": final_market(1 - index)}
        ]
    timestamps = iter(("2026-04-06T00:00:00Z", "2026-04-05T00:00:00Z"))
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path, api_server.url, client, lambda: next(timestamps), _test_hooks.pause
        )
        with pytest.raises(ValueError, match="SETTLEMENT-006"):
            reconcile_predictions(predictions, archive, True)
    assert len(api_server.requests) == 2
