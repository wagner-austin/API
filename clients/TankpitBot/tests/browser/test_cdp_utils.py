"""Tests for the CDP payload helpers.

:func:`_is_valid_base64` gates every capture payload before it reaches
a decoder, so the shapes it must reject are the shapes that would
otherwise reach ``base64.b64decode``. The empty-string case is the one
the pattern alone cannot cover: ``_BASE64_PATTERN`` is
``^[A-Za-z0-9+/]*={0,2}$`` and ``*`` matches zero characters, so ``""``
satisfies the pattern AND ``len("") % 4 == 0``. Only the explicit
emptiness check rejects it.
"""

from __future__ import annotations

from tankpit_bot.browser.cdp_utils import _is_valid_base64


class TestIsValidBase64:
    """Payload validation ahead of any decode attempt."""

    def test_empty_payload_is_rejected(self) -> None:
        """An empty payload is not valid base64 despite matching the pattern."""
        assert _is_valid_base64("") is False

    def test_well_formed_payload_is_accepted(self) -> None:
        """A padded four-character group is accepted."""
        assert _is_valid_base64("YWJj") is True

    def test_padded_payload_is_accepted(self) -> None:
        """Trailing ``=`` padding is part of the alphabet."""
        assert _is_valid_base64("YQ==") is True

    def test_payload_with_illegal_characters_is_rejected(self) -> None:
        """Characters outside the base64 alphabet are rejected."""
        assert _is_valid_base64("ab!d") is False

    def test_payload_with_a_length_not_a_multiple_of_four_is_rejected(self) -> None:
        """Base64 arrives in four-character groups."""
        assert _is_valid_base64("YWJjZA") is False
