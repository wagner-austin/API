"""Real HTTP acquisition, sealed replay and failure propagation for stock studies."""

from __future__ import annotations

from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import dump_json_str, require_list

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.equity_collection import collect_equity_outcomes
from covenant_ml.markets.equity_study import decode_equity_study
from covenant_ml.markets.validation import read_document

from ._server import ApiServer, api_server
from .test_equity_outcomes import captured
from .test_equity_study import study_document

__all__ = ["api_server"]


def test_real_http_complete_replay_and_corruption(api_server: ApiServer, tmp_path: Path) -> None:
    for symbol in ("AAPL", "MSFT"):
        api_server.documents["/v8/finance/chart/" + symbol] = [captured(symbol)]
    root = tmp_path / "archive"
    study = decode_equity_study(study_document())
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            root, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        report, paths = collect_equity_outcomes(study, archive, True)
        assert collect_equity_outcomes(study, archive, False) == (report, paths)
        assert report["complete"] is True
        assert report["returns"] == [
            {"symbol": "AAPL", "adjusted_return": 125.05754089355469 / 120.9556655883789 - 1},
            {"symbol": "MSFT", "adjusted_return": 214.47921752929688 / 217.6935577392578 - 1},
        ]
        assert len(require_list(report, "scores")) == 2
        assert len(paths) == 3
        assert len(api_server.requests) == 2
        assert (
            api_server.requests[0] == "/v8/finance/chart/AAPL?events=div%2Csplits&interval=1d"
            "&period1=1598572800&period2=1598918400"
        )
        with pytest.raises(FileExistsError, match="EQUITY-018"):
            collect_equity_outcomes(study, archive, True)
        paths[0].write_text("{}", encoding="utf-8")
        with pytest.raises(TypeError):
            collect_equity_outcomes(study, archive, False)
    assert len(api_server.requests) == 2


@pytest.mark.parametrize("change", ["identity", "body", "finish", "receipt"])
def test_pending_seal_validation(api_server: ApiServer, tmp_path: Path, change: str) -> None:
    study = decode_equity_study(
        dict(
            study_document(),
            decision_at="2099-08-28T00:00:00Z",
            expected_days=["2099-08-28", "2099-08-31"],
        )
    )
    root = tmp_path / "archive"
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            root, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        report, paths = collect_equity_outcomes(study, archive, True)
        assert report["complete"] is False
        assert report["scores"] == report["returns"] == []
        assert collect_equity_outcomes(study, archive, False) == (report, paths)
        assert api_server.requests == []
        seal = read_document(paths[0])
        if change == "identity":
            seal["url"] = "equity://wrong"
        elif change == "body":
            report["orders_submitted"] = 1
        elif change == "finish":
            seal["captured_at"] = "2000-01-01T00:00:00Z"
        else:
            report["started_at"] = "2100-01-01T00:00:00Z"
            # A mature replay must require actual price files; no HTTP fallback.
        body = dump_json_str(report)
        seal["body"], seal["digest"] = body, sha256(body.encode()).hexdigest()
        paths[0].write_text(dump_json_str(seal), encoding="utf-8")
        if change == "receipt":
            with pytest.raises(FileNotFoundError):
                collect_equity_outcomes(study, archive, False)
        else:
            with pytest.raises(ValueError, match="EQUITY-019"):
                collect_equity_outcomes(study, archive, False)
    assert api_server.requests == []


def test_real_http_failure(api_server: ApiServer, tmp_path: Path) -> None:
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path / "archive", api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        with pytest.raises(HTTPStatusError, match="404"):
            collect_equity_outcomes(decode_equity_study(study_document()), archive, True)
    assert len(api_server.requests) == 1
    assert list(tmp_path.iterdir()) == []


def test_actual_http_receipt_regression(api_server: ApiServer, tmp_path: Path) -> None:
    api_server.documents["/v8/finance/chart/AAPL"] = [captured("AAPL")]
    times = iter(("2026-10-14T00:00:01Z", "2026-10-14T00:00:00Z"))
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path / "archive", api_server.url, client, lambda: next(times), _test_hooks.pause
        )
        with pytest.raises(ValueError, match="EQUITY-019"):
            collect_equity_outcomes(decode_equity_study(study_document()), archive, True)
    assert len(api_server.requests) == 1


def test_acquisition_deadline_exact_boundary(api_server: ApiServer, tmp_path: Path) -> None:
    study = decode_equity_study(study_document())
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        before = ResponseArchive(
            tmp_path / "before",
            api_server.url,
            client,
            lambda: "2020-09-01T23:59:59Z",
            _test_hooks.pause,
        )
        report, _ = collect_equity_outcomes(study, before, True)
        assert report["complete"] is False
        assert report["scores"] == []
        assert api_server.requests == []
        at = ResponseArchive(
            tmp_path / "at",
            api_server.url,
            client,
            lambda: "2020-09-02T00:00:00Z",
            _test_hooks.pause,
        )
        with pytest.raises(HTTPStatusError, match="404"):
            collect_equity_outcomes(study, at, True)
    assert len(api_server.requests) == 1


def test_missing_unselected_member_never_writes_complete_report(
    api_server: ApiServer,
    tmp_path: Path,
) -> None:
    api_server.documents["/v8/finance/chart/AAPL"] = [captured("AAPL")]
    root = tmp_path / "archive"
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            root,
            api_server.url,
            client,
            _test_hooks.clock,
            _test_hooks.pause,
        )
        with pytest.raises(HTTPStatusError, match="404"):
            collect_equity_outcomes(decode_equity_study(study_document()), archive, True)
    assert len(api_server.requests) == 2
    assert len(list(root.glob("*.json"))) == 1
    assert (root / "observation.json").exists() is False
