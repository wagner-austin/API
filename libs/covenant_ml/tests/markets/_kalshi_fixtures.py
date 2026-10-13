"""Real generated signing keys and native account responses for socket tests."""

from __future__ import annotations

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from platform_core.json_utils import JSONObject

from covenant_ml.markets.kalshi_auth import KalshiSigner

from ._server import ApiServer

KEY_ID = "00000000-0000-4000-8000-000000000001"


def private_pem(key: ed25519.Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def signing_key() -> tuple[ed25519.Ed25519PrivateKey, KalshiSigner]:
    key = ed25519.Ed25519PrivateKey.generate()
    return key, KalshiSigner(KEY_ID, private_pem(key))


def balance() -> JSONObject:
    return {
        "balance": 2000,
        "balance_dollars": "20.000099",
        "portfolio_value": 0,
        "updated_ts": 100,
    }


def serve_account(server: ApiServer) -> None:
    server.documents = {
        "/trade-api/v2/portfolio/balance": [balance(), balance()],
        "/trade-api/v2/portfolio/orders": [{"orders": [], "cursor": ""}],
        "/trade-api/v2/portfolio/positions": [
            {"market_positions": [], "event_positions": [], "cursor": ""}
        ],
    }


def account_clock() -> str:
    return "1970-01-01T00:02:00+00:00"
