"""Tests for validators: DecodeCapabilitiesRequest."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError

from github_stats_api.api.validators.stats import (
    decode_capabilities_request,
    decode_hero_request,
)


class TestDecodeCapabilitiesRequest:
    """Tests for decode_capabilities_request function."""

    def test_decode_capabilities_request_minimal(self) -> None:
        """Test decoding with just repo."""
        req = decode_capabilities_request(
            repo="owner/repo",
            theme=None,
            hide_border=None,
            disable_animations=None,
        )

        assert req["repo"] == "owner/repo"
        assert req["theme"] == "default"
        assert req["hide_border"] is False
        assert req["disable_animations"] is False

    def test_decode_capabilities_request_all_options(self) -> None:
        """Test decoding with all options specified."""
        req = decode_capabilities_request(
            repo="wagner-austin/API",
            theme="dracula",
            hide_border="true",
            disable_animations="true",
        )

        assert req["repo"] == "wagner-austin/API"
        assert req["theme"] == "dracula"
        assert req["hide_border"] is True
        assert req["disable_animations"] is True

    def test_decode_capabilities_request_missing_repo_raises(self) -> None:
        """Test that missing repo raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo=None,
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "repo is required" in exc_info.value.message

    def test_decode_capabilities_request_empty_repo_raises(self) -> None:
        """Test that empty repo raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="   ",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "repo is required" in exc_info.value.message

    def test_decode_capabilities_request_invalid_format_no_slash_raises(self) -> None:
        """Test that repo without slash raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="just-a-repo",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "owner/repo" in exc_info.value.message

    def test_decode_capabilities_request_invalid_format_too_many_slashes_raises(self) -> None:
        """Test that repo with too many slashes raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="owner/repo/extra",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "owner/repo" in exc_info.value.message

    def test_decode_capabilities_request_empty_owner_raises(self) -> None:
        """Test that empty owner raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="/repo",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "owner/repo" in exc_info.value.message

    def test_decode_capabilities_request_empty_repo_name_raises(self) -> None:
        """Test that empty repo name raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="owner/",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "owner/repo" in exc_info.value.message

    def test_decode_capabilities_request_owner_too_long_raises(self) -> None:
        """Test that owner over 39 chars raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="a" * 40 + "/repo",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "39 characters" in exc_info.value.message

    def test_decode_capabilities_request_owner_invalid_char_raises(self) -> None:
        """Test that invalid owner character raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="owner_name/repo",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "invalid character" in exc_info.value.message

    def test_decode_capabilities_request_repo_name_too_long_raises(self) -> None:
        """Test that repo name over 100 chars raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="owner/" + "a" * 101,
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "100 characters" in exc_info.value.message

    def test_decode_capabilities_request_repo_name_invalid_char_raises(self) -> None:
        """Test that invalid repo name character raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="owner/repo@name",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "invalid character" in exc_info.value.message

    def test_decode_capabilities_request_accepts_valid_repo_names(self) -> None:
        """Test that valid repo name formats are accepted."""
        # Test various valid repo name formats
        valid_repos = [
            "owner/my-repo",
            "owner/my_repo",
            "owner/my.repo",
            "owner/my-repo_name.v2",
            "owner123/repo456",
        ]
        for repo in valid_repos:
            req = decode_capabilities_request(
                repo=repo,
                theme=None,
                hide_border=None,
                disable_animations=None,
            )
            assert req["repo"] == repo

    def test_decode_capabilities_request_invalid_theme_raises(self) -> None:
        """Test that invalid theme raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_capabilities_request(
                repo="owner/repo",
                theme="invalid-theme",
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "theme must be one of" in exc_info.value.message


class TestDecodeHeroRequest:
    """Tests for decode_hero_request function."""

    def test_decode_hero_request_minimal(self) -> None:
        """Test decoding with minimal required fields."""
        req = decode_hero_request(
            name="Austin Wagner",
            subtitle=None,
            lines=None,
            theme=None,
            disable_animations=None,
        )

        assert req["name"] == "Austin Wagner"
        assert req["subtitle"] == ""
        assert req["lines"] == ()
        assert req["theme"] == "default"
        assert req["disable_animations"] is False

    def test_decode_hero_request_all_options(self) -> None:
        """Test decoding with all options specified."""
        req = decode_hero_request(
            name="Austin Wagner",
            subtitle="Full-Stack Dev | ML Researcher",
            lines="Location: Irvine|Education: UC Irvine",
            theme="cyberpunk",
            disable_animations="true",
        )

        assert req["name"] == "Austin Wagner"
        assert req["subtitle"] == "Full-Stack Dev | ML Researcher"
        assert req["lines"] == ("Location: Irvine", "Education: UC Irvine")
        assert req["theme"] == "cyberpunk"
        assert req["disable_animations"] is True

    def test_decode_hero_request_missing_name_raises(self) -> None:
        """Test that missing name raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_hero_request(
                name=None,
                subtitle=None,
                lines=None,
                theme=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "name is required" in exc_info.value.message

    def test_decode_hero_request_empty_name_raises(self) -> None:
        """Test that empty name raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_hero_request(
                name="   ",
                subtitle=None,
                lines=None,
                theme=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "name is required" in exc_info.value.message

    def test_decode_hero_request_name_too_long_raises(self) -> None:
        """Test that name exceeding 40 chars raises AppError."""
        long_name = "x" * 41
        with pytest.raises(AppError) as exc_info:
            decode_hero_request(
                name=long_name,
                subtitle=None,
                lines=None,
                theme=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "name must be 40 characters" in exc_info.value.message

    def test_decode_hero_request_subtitle_too_long_raises(self) -> None:
        """Test that subtitle exceeding 80 chars raises AppError."""
        long_subtitle = "x" * 81
        with pytest.raises(AppError) as exc_info:
            decode_hero_request(
                name="Test",
                subtitle=long_subtitle,
                lines=None,
                theme=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "subtitle must be 80 characters" in exc_info.value.message

    def test_decode_hero_request_too_many_lines_raises(self) -> None:
        """Test that too many lines raises AppError."""
        lines = "|".join([f"Line {i}" for i in range(9)])
        with pytest.raises(AppError) as exc_info:
            decode_hero_request(
                name="Test",
                subtitle=None,
                lines=lines,
                theme=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "at most 8 lines" in exc_info.value.message

    def test_decode_hero_request_line_too_long_raises(self) -> None:
        """Test that line exceeding 80 chars raises AppError."""
        long_line = "x" * 81
        with pytest.raises(AppError) as exc_info:
            decode_hero_request(
                name="Test",
                subtitle=None,
                lines=long_line,
                theme=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "exceeds 80 characters" in exc_info.value.message
