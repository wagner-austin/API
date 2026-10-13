"""Real HTTP account reads, exact codecs and fail-closed preflight integration."""

from __future__ import annotations

from base64 import b64decode
from contextlib import closing
from urllib.parse import urlsplit

import pytest
from platform_core.http_client import build_client
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.kalshi_demo import DemoAccountClient, decode_preflight, encode_preflight

from ._kalshi_fixtures import KEY_ID, account_clock, balance, serve_account, signing_key
from ._server import ApiServer, api_server

__all__ = ["api_server"]


def preflight_document() -> JSONObject:
    return {
        "credential_fingerprint": "a" * 64,
        "balance_microdollars": 20_000_099,
        "portfolio_value_cents": 0,
        "balance_updated_seconds": 100,
        "started_ms": 120_000,
        "completed_ms": 120_000,
    }


def test_authenticated_socket_preflight(api_server: ApiServer) -> None:
    serve_account(api_server)
    key, signer = signing_key()
    with closing(build_client(5)) as client:
        service = DemoAccountClient(client, signer, account_clock, origin=api_server.url)
        observed = encode_preflight(service.preflight())
        expected = preflight_document()
        expected["credential_fingerprint"] = signer.fingerprint
        assert observed == expected
        assert encode_preflight(decode_preflight(observed)) == observed
        assert _test_hooks.demo_account_factory is DemoAccountClient
        DemoAccountClient(client, signer, _test_hooks.clock)
    assert api_server.requests == [
        "/trade-api/v2/portfolio/balance?subaccount=0",
        "/trade-api/v2/portfolio/orders?subaccount=0&limit=1",
        "/trade-api/v2/portfolio/positions?subaccount=0&limit=1&settlement_status=unsettled",
        "/trade-api/v2/portfolio/balance?subaccount=0",
    ]
    assert len(api_server.request_headers) == 4
    for request, headers in zip(api_server.requests, api_server.request_headers, strict=True):
        assert headers["KALSHI-ACCESS-KEY"] == KEY_ID
        assert headers["KALSHI-ACCESS-TIMESTAMP"] == "120000"
        signature = b64decode(headers["KALSHI-ACCESS-SIGNATURE"], validate=True)
        key.public_key().verify(signature, ("120000GET" + urlsplit(request).path).encode())


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("credential_fingerprint", "g" * 64, "KALSHI-DEMO-001"),
        ("completed_ms", 119999, "KALSHI-DEMO-001"),
        ("completed_ms", 150001, "KALSHI-DEMO-001"),
        ("balance_updated_seconds", 121, "MARKET-001"),
        ("balance_microdollars", -1, "MARKET-001"),
        ("portfolio_value_cents", 1, "MARKET-001"),
    ],
)
def test_saved_preflight_validation(field: str, value: JSONValue, code: str) -> None:
    document = preflight_document()
    document[field] = value
    with pytest.raises(ValueError, match=code):
        decode_preflight(document)


@pytest.mark.parametrize(
    "origin",
    [
        "https://external-api.kalshi.com",
        "https://external-api.demo.kalshi.co.evil.example",
        "https://user:pass@external-api.demo.kalshi.co",
        "http://192.168.1.1:80",
    ],
)
def test_production_and_other_origins_refused(origin: str) -> None:
    _, signer = signing_key()
    with closing(build_client(5)) as client, pytest.raises(ValueError, match="KALSHI-DEMO-002"):
        DemoAccountClient(client, signer, account_clock, origin=origin)


def test_actual_http_failure_no_retry(api_server: ApiServer) -> None:
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        service = DemoAccountClient(client, signer, account_clock, origin=api_server.url)
        with pytest.raises(ValueError, match=r"KALSHI-DEMO-003.*HTTP 404"):
            service.preflight()
    assert api_server.requests == ["/trade-api/v2/portfolio/balance?subaccount=0"]


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
def test_http_status_failures_redact_body_and_do_not_retry(
    api_server: ApiServer, status: int
) -> None:
    path = "/trade-api/v2/portfolio/balance"
    api_server.documents[path] = [{"details": "sensitive server diagnostic"}]
    api_server.statuses[path] = status
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        service = DemoAccountClient(client, signer, account_clock, origin=api_server.url)
        with pytest.raises(ValueError) as raised:
            service.preflight()
    assert str(raised.value) == (
        f"[KALSHI-DEMO-003] /portfolio/balance returned HTTP {status}; preflight failed"
    )
    assert api_server.requests == [path + "?subaccount=0"]


@pytest.mark.parametrize(
    ("endpoint", "field", "value", "code", "requests"),
    [
        ("orders", "cursor", "more", "KALSHI-DEMO-004", 2),
        ("orders", "orders", [{"order_id": "untracked"}], "KALSHI-DEMO-004", 2),
        ("positions", "cursor", "more", "KALSHI-DEMO-004", 3),
        ("positions", "market_positions", [{"ticker": "position"}], "KALSHI-DEMO-004", 3),
        ("positions", "event_positions", [{"event_ticker": "event"}], "KALSHI-DEMO-004", 3),
        ("balance", "balance", 1999, "KALSHI-DEMO-006", 1),
        ("balance", "balance_dollars", "20.0000001", "MARKET-005", 1),
        ("balance", "portfolio_value", 1, "KALSHI-DEMO-005", 4),
    ],
)
def test_native_account_refusals(
    api_server: ApiServer,
    endpoint: str,
    field: str,
    value: JSONValue,
    code: str,
    requests: int,
) -> None:
    serve_account(api_server)
    api_server.documents["/trade-api/v2/portfolio/" + endpoint][0][field] = value
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        service = DemoAccountClient(client, signer, account_clock, origin=api_server.url)
        with pytest.raises(ValueError, match=code):
            service.preflight()
    assert len(api_server.requests) == requests


@pytest.mark.parametrize("endpoint", ["orders", "positions"])
def test_absent_cursor_is_incomplete_evidence(api_server: ApiServer, endpoint: str) -> None:
    serve_account(api_server)
    del api_server.documents["/trade-api/v2/portfolio/" + endpoint][0]["cursor"]
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        service = DemoAccountClient(client, signer, account_clock, origin=api_server.url)
        with pytest.raises(TypeError, match="cursor"):
            service.preflight()


def test_nonzero_portfolio_even_with_empty_pages(api_server: ApiServer) -> None:
    serve_account(api_server)
    document = balance()
    document["portfolio_value"] = 1
    api_server.documents["/trade-api/v2/portfolio/balance"] = [document, document]
    _, signer = signing_key()
    with closing(build_client(5)) as client:
        service = DemoAccountClient(client, signer, account_clock, origin=api_server.url)
        with pytest.raises(ValueError, match=r"MARKET-001.*portfolio_value_cents"):
            service.preflight()
