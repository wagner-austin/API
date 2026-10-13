"""Actual Ed25519/RSA signature verification, including altered-message failures."""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

from covenant_ml.markets.kalshi_auth import KalshiSigner

from ._kalshi_fixtures import KEY_ID, private_pem, signing_key


def test_ed25519_signatures_exclude_queries_and_bind_every_signed_field() -> None:
    key, signer = signing_key()
    headers = signer.headers("GET", "/trade-api/v2/portfolio/balance?subaccount=0", 120000)
    assert headers["KALSHI-ACCESS-KEY"] == KEY_ID
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "120000"
    signature = b64decode(headers["KALSHI-ACCESS-SIGNATURE"], validate=True)
    message = b"120000GET/trade-api/v2/portfolio/balance"
    key.public_key().verify(signature, message)
    for altered in (
        b"120001GET/trade-api/v2/portfolio/balance",
        b"120000POST/trade-api/v2/portfolio/balance",
        b"120000GET/trade-api/v2/portfolio/orders",
        message + b"?subaccount=0",
    ):
        with pytest.raises(InvalidSignature):
            key.public_key().verify(signature, altered)
    assert signer.headers("GET", "/trade-api/v2/portfolio/balance", 120000) == headers
    public = key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    assert signer.fingerprint == sha256(KEY_ID.encode() + b"\0" + public).hexdigest()
    other_id = "00000000-0000-4000-8000-000000000002"
    other = KalshiSigner(other_id, private_pem(key))
    assert other.fingerprint == sha256(other_id.encode() + b"\0" + public).hexdigest()


def test_rsa_pss_sha256_digest_length_salt() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    signer = KalshiSigner(KEY_ID, pem)
    signature = b64decode(
        signer.headers("POST", "/trade-api/v2/portfolio/events/orders", 123)[
            "KALSHI-ACCESS-SIGNATURE"
        ],
        validate=True,
    )
    pss = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32)
    key.public_key().verify(
        signature, b"123POST/trade-api/v2/portfolio/events/orders", pss, hashes.SHA256()
    )
    with pytest.raises(InvalidSignature):
        key.public_key().verify(signature, b"changed", pss, hashes.SHA256())


@pytest.mark.parametrize("kind", ["ec", "small_rsa", "bad_pem", "encrypted", "key_id"])
def test_private_key_validation(kind: str) -> None:
    key, _ = signing_key()
    pem = private_pem(key)
    key_id = KEY_ID
    expected = "KALSHI-AUTH-002"
    exception: type[ValueError] | type[TypeError] = ValueError
    if kind == "ec":
        pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    elif kind == "small_rsa":
        pem = rsa.generate_private_key(public_exponent=65537, key_size=1024).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    elif kind == "bad_pem":
        pem = b"invalid PEM"
        expected = "Unable to load PEM"
    elif kind == "encrypted":
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"test-password"),
        )
        exception = TypeError
        expected = "Password was not given"
    else:
        key_id = "00000000000040008000000000000001"
        expected = "KALSHI-AUTH-001"
    with pytest.raises(exception, match=expected):
        KalshiSigner(key_id, pem)


@pytest.mark.parametrize(
    ("path", "timestamp"),
    [
        ("https://kalshi.com/trade-api/v2/test", 1),
        ("/trade-api/v2/test#fragment", 1),
        ("/trade-api/v2/test\n", 1),
        ("/trade-api/v2/tést", 1),
        ("/trade-api/v2/test", -1),
    ],
)
def test_invalid_signature_input(path: str, timestamp: int) -> None:
    _, signer = signing_key()
    with pytest.raises(ValueError, match="KALSHI-AUTH-003"):
        signer.headers("GET", path, timestamp)
