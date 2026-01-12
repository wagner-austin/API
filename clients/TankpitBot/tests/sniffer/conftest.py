"""Shared fixtures and helpers for sniffer tests."""

from __future__ import annotations

import base64


def make_payload(body: bytes) -> str:
    """Create a base64 payload with 2-byte length header.

    Args:
        body: Raw message body bytes.

    Returns:
        Base64-encoded payload with length header.
    """
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()
