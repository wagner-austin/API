"""Tests for platform_email.types.draft module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_email.types.draft import (
    Draft,
    decode_draft,
    encode_draft,
)
from platform_email.types.email import EmailAddress


def _make_test_draft() -> Draft:
    """Create a test draft for testing."""
    return Draft(
        id="draft-123",
        subject="Draft Subject",
        body="Draft body content",
        body_type="text",
        to=(EmailAddress(address="recipient@test.com", name="Recipient"),),
        cc=(),
        bcc=(),
    )


class TestDraft:
    """Tests for Draft encode/decode functions."""

    def test_encode_draft(self) -> None:
        """Test encoding a Draft to JSON."""
        draft = _make_test_draft()
        result = encode_draft(draft)

        assert result["id"] == "draft-123"
        assert result["subject"] == "Draft Subject"
        assert result["body"] == "Draft body content"
        assert result["body_type"] == "text"
        # Verify list lengths by decoding back
        decoded = decode_draft(result)
        assert len(decoded["to"]) == 1
        assert len(decoded["cc"]) == 0
        assert len(decoded["bcc"]) == 0

    def test_encode_draft_with_multiple_recipients(self) -> None:
        """Test encoding draft with multiple to/cc/bcc."""
        draft = Draft(
            id="draft-multi",
            subject="Multi-recipient draft",
            body="<p>HTML body</p>",
            body_type="html",
            to=(
                EmailAddress(address="a@test.com", name="A"),
                EmailAddress(address="b@test.com", name="B"),
            ),
            cc=(EmailAddress(address="c@test.com", name="C"),),
            bcc=(
                EmailAddress(address="d@test.com", name="D"),
                EmailAddress(address="e@test.com", name="E"),
            ),
        )
        result = encode_draft(draft)

        # Verify list lengths by decoding back
        decoded = decode_draft(result)
        assert len(decoded["to"]) == 2
        assert len(decoded["cc"]) == 1
        assert len(decoded["bcc"]) == 2
        assert decoded["body_type"] == "html"

    def test_decode_draft(self) -> None:
        """Test decoding a Draft from JSON."""
        data: JSONObject = {
            "id": "draft-decoded",
            "subject": "Decoded Subject",
            "body": "Decoded body",
            "body_type": "html",
            "to": [{"address": "to@test.com", "name": "To"}],
            "cc": [{"address": "cc@test.com", "name": "CC"}],
            "bcc": [],
        }
        result = decode_draft(data)

        assert result["id"] == "draft-decoded"
        assert result["body_type"] == "html"
        assert len(result["to"]) == 1
        assert len(result["cc"]) == 1
        assert len(result["bcc"]) == 0

    def test_decode_draft_raises_for_missing_id(self) -> None:
        """Test that missing id raises JSONTypeError."""
        data: JSONObject = {
            "subject": "Test",
            "body": "Body",
            "body_type": "text",
            "to": [],
            "cc": [],
            "bcc": [],
        }
        with pytest.raises(JSONTypeError):
            decode_draft(data)

    def test_decode_draft_raises_for_invalid_body_type(self) -> None:
        """Test that invalid body_type raises JSONTypeError."""
        data: JSONObject = {
            "id": "draft-err",
            "subject": "Test",
            "body": "Body",
            "body_type": "markdown",
            "to": [],
            "cc": [],
            "bcc": [],
        }
        with pytest.raises(JSONTypeError) as exc_info:
            decode_draft(data)
        assert "must be text/html" in str(exc_info.value)

    def test_roundtrip(self) -> None:
        """Test encode then decode preserves data."""
        original = _make_test_draft()
        encoded = encode_draft(original)
        decoded = decode_draft(encoded)
        assert decoded == original

    def test_roundtrip_with_html_and_recipients(self) -> None:
        """Test roundtrip with HTML body and multiple recipients."""
        original = Draft(
            id="roundtrip-draft",
            subject="Roundtrip Test",
            body="<html><body>Test</body></html>",
            body_type="html",
            to=(EmailAddress(address="to@test.com", name="To"),),
            cc=(EmailAddress(address="cc@test.com", name="CC"),),
            bcc=(EmailAddress(address="bcc@test.com", name="BCC"),),
        )
        encoded = encode_draft(original)
        decoded = decode_draft(encoded)
        assert decoded == original
