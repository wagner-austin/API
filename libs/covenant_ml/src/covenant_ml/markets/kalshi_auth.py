"""Kalshi request signatures using validated private keys and exact signed paths."""

from __future__ import annotations

import re
from base64 import b64encode
from hashlib import sha256
from typing import Literal

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa


class KalshiSigner:
    """Sign requests without retaining PEM text or exposing private key material."""

    def __init__(self, key_id: str, pem: bytes) -> None:
        """Load an unencrypted Ed25519 or RSA-2048 key.

        Args:
            key_id: Canonical UUID identifying the exchange API key.
            pem: Private key PEM bytes read at the application boundary.

        Raises:
            ValueError: KALSHI-AUTH-001 for a noncanonical identifier, or
                KALSHI-AUTH-002 for an unsupported algorithm or RSA size.
            ValueError: Cryptography cannot decode the PEM.
            TypeError: The PEM requires a password.
        """
        if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", key_id) is None:
            raise ValueError("[KALSHI-AUTH-001] API key identifier must be a canonical UUID")
        key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(key, ed25519.Ed25519PrivateKey | rsa.RSAPrivateKey):
            raise ValueError("[KALSHI-AUTH-002] only Ed25519 and RSA-2048 keys are supported")
        if isinstance(key, rsa.RSAPrivateKey) and key.key_size != 2048:
            raise ValueError("[KALSHI-AUTH-002] RSA key must contain exactly 2048 bits")
        self._key = key
        self._key_id = key_id
        public = key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.fingerprint = sha256(key_id.encode("ascii") + b"\0" + public).hexdigest()

    def headers(
        self, method: Literal["GET", "POST"], path: str, timestamp_ms: int
    ) -> dict[str, str]:
        """Sign the timestamp, verb and absolute API path, excluding query text.

        Args:
            method: Uppercase HTTP verb.
            path: Absolute trade API path, optionally followed by a query.
            timestamp_ms: Nonnegative Unix timestamp in milliseconds.

        Returns:
            Authentication headers; callers must not log these values.

        Raises:
            ValueError: KALSHI-AUTH-003 for an invalid path or timestamp.
        """
        if (
            not path.startswith("/trade-api/v2/")
            or "#" in path
            or any(ord(char) < 33 or ord(char) > 126 for char in path)
            or timestamp_ms < 0
        ):
            raise ValueError("[KALSHI-AUTH-003] invalid signature path or timestamp")
        timestamp = str(timestamp_ms)
        message = (timestamp + method + path.split("?", 1)[0]).encode("ascii")
        if isinstance(self._key, ed25519.Ed25519PrivateKey):
            signature = self._key.sign(message)
        else:
            signature = self._key.sign(
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                hashes.SHA256(),
            )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": b64encode(signature).decode("ascii"),
        }
