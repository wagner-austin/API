"""Real archived Apple prices exercise disclosure timing and action accounting."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_list,
)

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.disclosure_history import decode_submission_columns
from covenant_ml.markets.equity_history import adjusted_forward_return, decode_equity_chart
from covenant_ml.markets.equity_records import (
    EquityPrice,
    decode_equity_price,
    encode_equity_price,
)
from covenant_ml.markets.validation import read_document

from ._server import ApiServer, api_server

__all__ = ["api_server"]

RECEIVED = "2026-10-14T05:34:39Z"


def captured(year: int = 2009) -> JSONObject:
    """Read the unchanged native price capture, including its action ledger."""
    return read_document(Path(__file__).parent / f"fixtures/equity_apple_{year}.json")


def native_result(document: JSONObject) -> JSONObject:
    """Select the source result for deliberately malformed boundary cases."""
    return narrow_json_to_dict(require_list(require_dict(document, "chart"), "result")[0])


def prices(year: int = 2009) -> tuple[EquityPrice, ...]:
    """Decode captured evidence through the production native decoder."""
    return decode_equity_chart(captured(year), "AAPL", "USD", received_at=RECEIVED)


def test_real_disclosure_to_future_close_join() -> None:
    history = prices()
    assert len(history) == 83
    for row in history:
        assert decode_equity_price(encode_equity_price(row)) == row
    document = read_document(Path(__file__).parent / "fixtures/disclosure_apple_assets_2009.json")
    filing = next(
        row
        for row in decode_submission_columns(require_dict(document, "submissions"), 320193)
        if row["accession"] == "0001193125-09-214859"
    )
    entry: EquityPrice = {
        "symbol": "AAPL",
        "currency": "USD",
        "session_start_s": 1256736600,
        "split_adjusted_close": 6.871428966522217,
        "distribution_adjusted_close": 5.754613399505615,
    }
    exit_price: EquityPrice = {
        "symbol": "AAPL",
        "currency": "USD",
        "session_start_s": 1259159400,
        "split_adjusted_close": 7.292500019073486,
        "distribution_adjusted_close": 6.1072468757629395,
    }
    assert adjusted_forward_return(
        history, available_at=filing["accepted_at"], holding_bars=20
    ) == (entry, exit_price, 6.1072468757629395 / 5.754613399505615 - 1)


def test_split_and_dividend_are_not_applied_twice() -> None:
    history = prices(2020)
    assert len(history) == 64
    events = require_dict(native_result(captured(2020)), "events")
    assert require_dict(events, "dividends") == {
        "1596807000": {"amount": 0.205, "date": 1596807000}
    }
    assert require_dict(events, "splits") == {
        "1598880600": {
            "date": 1598880600,
            "numerator": 4.0,
            "denominator": 1.0,
            "splitRatio": "4:1",
        }
    }
    assert adjusted_forward_return(
        history, available_at="2020-08-06T00:00:00Z", holding_bars=1
    ) == (history[25], history[26], 107.68331909179688 / 110.18853759765625 - 1)
    assert adjusted_forward_return(
        history, available_at="2020-08-28T00:00:00Z", holding_bars=1
    ) == (history[41], history[42], 125.05754089355469 / 120.9556655883789 - 1)


def test_availability_boundaries_and_incomplete_windows() -> None:
    history = prices()
    for available in ("2009-10-28T13:30:00Z", "2009-10-28T13:30:00.001Z", "2009-10-28T23:59:59Z"):
        assert adjusted_forward_return(history, available_at=available, holding_bars=1) == (
            history[20],
            history[21],
            history[21]["distribution_adjusted_close"] / history[20]["distribution_adjusted_close"]
            - 1,
        )
    for rows, available, horizon in (
        (history, "2010-02-01T00:00:00Z", 1),
        (history, "2010-01-29T00:00:00Z", 1),
        (history, "2009-10-01T00:00:00Z", 83),
        ((), "2009-10-01T00:00:00Z", 1),
    ):
        assert adjusted_forward_return(rows, available_at=available, holding_bars=horizon) is None
    with pytest.raises(ValueError, match="MARKET-001"):
        adjusted_forward_return(history, available_at=RECEIVED, holding_bars=0)


def test_provider_errors_and_result_cardinality() -> None:
    charts: tuple[JSONObject, ...] = (
        {"result": []},
        {"error": {"code": "Not Found"}},
        {"error": None, "result": []},
    )
    for chart in charts:
        document: JSONObject = {"chart": chart}
        with pytest.raises(ValueError, match="EQUITY-002"):
            decode_equity_chart(document, "AAPL", "USD", received_at=RECEIVED)
    for key in ("quote", "adjclose"):
        document = captured()
        require_dict(native_result(document), "indicators")[key] = []
        with pytest.raises(ValueError, match="EQUITY-002"):
            decode_equity_chart(document, "AAPL", "USD", received_at=RECEIVED)


@pytest.mark.parametrize(
    "key,value",
    [("symbol", "MSFT"), ("currency", "EUR"), ("instrumentType", "ETF"), ("dataGranularity", "1h")],
)
def test_native_identity_mismatch(key: str, value: str) -> None:
    document = captured()
    require_dict(native_result(document), "meta")[key] = value
    with pytest.raises(ValueError, match="EQUITY-003"):
        decode_equity_chart(document, "AAPL", "USD", received_at=RECEIVED)


def test_history_array_order_and_completion_failures() -> None:
    malformed: tuple[list[JSONValue], ...] = (
        [],
        [1],
        [1256736600] * 83,
        list(reversed(require_list(native_result(captured()), "timestamp"))),
    )
    for values in malformed:
        document = captured()
        native_result(document)["timestamp"] = values
        with pytest.raises(ValueError, match="EQUITY-004"):
            decode_equity_chart(document, "AAPL", "USD", received_at=RECEIVED)
    document = captured()
    with pytest.raises(ValueError, match="EQUITY-004"):
        decode_equity_chart(document, "AAPL", "USD", received_at="2009-10-01T13:30:00Z")
    indicators = require_dict(native_result(document), "indicators")
    quote = narrow_json_to_dict(require_list(indicators, "quote")[0])
    require_list(quote, "close")[0] = None
    with pytest.raises(TypeError, match="split_adjusted_close"):
        decode_equity_chart(document, "AAPL", "USD", received_at=RECEIVED)


def test_return_identity_order_and_numeric_limits() -> None:
    history = prices()
    changed = encode_equity_price(history[0])
    changed["symbol"] = "MSFT"
    for rows in (
        (decode_equity_price(changed), *history[1:]),
        tuple(reversed(history)),
        (history[0], *history),
    ):
        with pytest.raises(ValueError, match="EQUITY-005"):
            adjusted_forward_return(rows, available_at="2009-01-01T00:00:00Z", holding_bars=1)
    for first, second in ((1e-300, 1e300), (1e300, 1e-300)):
        entry, exit_price = encode_equity_price(history[0]), encode_equity_price(history[1])
        entry["distribution_adjusted_close"] = first
        exit_price["distribution_adjusted_close"] = second
        with pytest.raises(ValueError, match="EQUITY-006"):
            adjusted_forward_return(
                (decode_equity_price(entry), decode_equity_price(exit_price)),
                available_at="2009-01-01T00:00:00Z",
                holding_bars=1,
            )


def test_real_price_transport_and_offline_replay(api_server: ApiServer, tmp_path: Path) -> None:
    api_server.documents["/prices"] = [captured(2020)]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        online = decode_equity_chart(
            archive.capture("/prices", {}), "AAPL", "USD", received_at=RECEIVED
        )
        replay = decode_equity_chart(
            archive.replay("/prices", {}), "AAPL", "USD", received_at=RECEIVED
        )
        assert online == replay == prices(2020)
        assert adjusted_forward_return(
            online, available_at="2020-08-28T00:00:00Z", holding_bars=1
        ) == (
            replay[41],
            replay[42],
            125.05754089355469 / 120.9556655883789 - 1,
        )
        archive.file_for(archive.request_url("/prices", {})).write_text("{}", encoding="utf-8")
        with pytest.raises(TypeError, match="body"):
            archive.replay("/prices", {})
        with pytest.raises(HTTPStatusError, match="404"):
            archive.capture("/delisted", {})
        assert not archive.file_for(archive.request_url("/delisted", {})).exists()
    assert len(api_server.requests) == 2
