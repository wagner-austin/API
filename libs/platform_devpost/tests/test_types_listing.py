"""Tests for types: HackathonListMeta."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_devpost.types import (
    HackathonListMeta,
    HackathonListResponse,
    decode_list_meta,
    decode_list_response,
    encode_list_meta,
    encode_list_response,
)


class TestHackathonListMeta:
    """Tests for HackathonListMeta type and encode/decode."""

    def test_meta_creation(self) -> None:
        """Test creating a HackathonListMeta instance."""
        meta = HackathonListMeta(total_count=100, per_page=10)
        assert meta.total_count == 100
        assert meta.per_page == 10

    def test_encode_list_meta(self) -> None:
        """Test encoding HackathonListMeta to dict."""
        meta = HackathonListMeta(total_count=50, per_page=20)
        result = encode_list_meta(meta)
        assert result == {"total_count": 50, "per_page": 20}

    def test_decode_list_meta(self) -> None:
        """Test decoding HackathonListMeta from dict."""
        data: JSONObject = {"total_count": 200, "per_page": 25}
        meta = decode_list_meta(data)
        assert meta.total_count == 200
        assert meta.per_page == 25


class TestHackathonListResponse:
    """Tests for HackathonListResponse type and encode/decode."""

    def test_response_creation(self) -> None:
        """Test creating a HackathonListResponse instance."""
        resp = HackathonListResponse(
            hackathons=(),
            meta=HackathonListMeta(total_count=0, per_page=10),
        )
        assert len(resp.hackathons) == 0
        assert resp.meta.total_count == 0

    def test_encode_list_response(self) -> None:
        """Test encoding HackathonListResponse to dict."""
        resp = HackathonListResponse(
            hackathons=(),
            meta=HackathonListMeta(total_count=0, per_page=10),
        )
        result = encode_list_response(resp)
        assert result["hackathons"] == []
        assert result["meta"] == {"total_count": 0, "per_page": 10}

    def test_decode_list_response(self) -> None:
        """Test decoding HackathonListResponse from dict."""
        data: JSONObject = {
            "hackathons": [
                {
                    "id": 1,
                    "title": "Test",
                    "url": "https://example.com",
                    "thumbnail_url": "https://example.com/img.jpg",
                    "organization_name": "Org",
                    "displayed_location": {"icon": "x", "location": "x"},
                    "open_state": "open",
                    "time_left_to_submission": "1 day",
                    "submission_period_dates": "Jan 1-2",
                    "themes": [],
                    "prize_amount": "$0",
                    "registrations_count": 0,
                    "featured": False,
                    "winners_announced": False,
                    "invite_only": False,
                }
            ],
            "meta": {"total_count": 1, "per_page": 10},
        }
        resp = decode_list_response(data)
        assert len(resp.hackathons) == 1
        assert resp.meta.total_count == 1

    def test_decode_list_response_invalid_hackathon_type(self) -> None:
        """Test decode_list_response raises when hackathon is not a dict."""
        data: JSONObject = {
            "hackathons": ["not a dict"],
            "meta": {"total_count": 1, "per_page": 10},
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_list_response(data)

    def test_decode_list_response_invalid_meta_type(self) -> None:
        """Test decode_list_response raises when meta is not a dict."""
        data: JSONObject = {
            "hackathons": [],
            "meta": "not a dict",
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_list_response(data)
