"""Tests for probe error classes and validation functions."""

from __future__ import annotations

from tankpit_bot.browser import BrowserError
from tankpit_bot.probe import (
    GameNotJoinedError,
    PlaywrightNotInstalledError,
    ProbeError,
    _extract_magic_from_payload,
    _is_valid_base64,
)

# =============================================================================
# Base64 Validation Tests
# =============================================================================


def test_is_valid_base64_empty_string() -> None:
    """Test _is_valid_base64 returns False for empty string."""
    assert _is_valid_base64("") is False


def test_is_valid_base64_invalid_chars() -> None:
    """Test _is_valid_base64 returns False for invalid characters."""
    assert _is_valid_base64("not!valid@base64") is False


def test_is_valid_base64_wrong_length() -> None:
    """Test _is_valid_base64 returns False for wrong length."""
    # Valid chars but not multiple of 4
    assert _is_valid_base64("abc") is False


def test_is_valid_base64_valid() -> None:
    """Test _is_valid_base64 returns True for valid base64."""
    assert _is_valid_base64("YWJj") is True
    assert _is_valid_base64("YWJjZA==") is True


# =============================================================================
# Magic Extraction Tests
# =============================================================================


def test_extract_magic_from_payload_invalid_base64() -> None:
    """Test _extract_magic_from_payload returns None for invalid base64."""
    assert _extract_magic_from_payload("not!valid") is None


def test_extract_magic_from_payload_not_auth() -> None:
    """Test _extract_magic_from_payload returns None for non-AUTH message."""
    import base64

    body = "HELLO test message"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

    assert _extract_magic_from_payload(payload) is None


def test_extract_magic_from_payload_valid_auth() -> None:
    """Test _extract_magic_from_payload extracts magic from valid AUTH."""
    import base64

    body = "%AUTH !be session|hash|ts test_magic_key_12345"
    body_bytes = body.encode("utf-8")
    length_prefix = len(body_bytes).to_bytes(2, "little")
    payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

    assert _extract_magic_from_payload(payload) == "test_magic_key_12345"


# =============================================================================
# Error Class Tests
# =============================================================================


def test_probe_error_is_exception() -> None:
    """Test ProbeError is an Exception."""
    assert issubclass(ProbeError, Exception)
    err = ProbeError("test error")
    assert str(err) == "test error"


def test_playwright_not_installed_error_is_browser_error() -> None:
    """Test PlaywrightNotInstalledError is a BrowserError."""
    assert issubclass(PlaywrightNotInstalledError, BrowserError)


def test_game_not_joined_error_is_browser_error() -> None:
    """Test GameNotJoinedError is a BrowserError."""
    assert issubclass(GameNotJoinedError, BrowserError)
