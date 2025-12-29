from __future__ import annotations

import pytest
from platform_core.errors import AppError

from github_stats_api.api.validators.stats import (
    decode_capabilities_request,
    decode_langs_request,
    decode_stats_request,
)


class TestDecodeStatsRequest:
    """Tests for decode_stats_request function."""

    def test_decode_stats_request_minimal(self) -> None:
        """Test decoding with just username."""
        req = decode_stats_request(
            username="testuser",
            theme=None,
            hide_border=None,
            show_icons=None,
            include_all_commits=None,
            hide=None,
            disable_animations=None,
        )

        assert req["username"] == "testuser"
        assert req["theme"] == "default"
        assert req["hide_border"] is False
        assert req["show_icons"] is True
        assert req["include_all_commits"] is False
        assert req["hide"] == ()
        assert req["disable_animations"] is False

    def test_decode_stats_request_all_options(self) -> None:
        """Test decoding with all options specified."""
        req = decode_stats_request(
            username="wagner-austin",
            theme="dracula",
            hide_border="true",
            show_icons="false",
            include_all_commits="true",
            hide="stars,commits",
            disable_animations="true",
        )

        assert req["username"] == "wagner-austin"
        assert req["theme"] == "dracula"
        assert req["hide_border"] is True
        assert req["show_icons"] is False
        assert req["include_all_commits"] is True
        assert req["hide"] == ("stars", "commits")
        assert req["disable_animations"] is True

    def test_decode_stats_request_missing_username_raises(self) -> None:
        """Test that missing username raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username=None,
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "username is required" in exc_info.value.message

    def test_decode_stats_request_empty_username_raises(self) -> None:
        """Test that empty username raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="   ",
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "username is required" in exc_info.value.message

    def test_decode_stats_request_username_too_long_raises(self) -> None:
        """Test that username over 39 chars raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="a" * 40,
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "39 characters" in exc_info.value.message

    def test_decode_stats_request_username_invalid_char_raises(self) -> None:
        """Test that invalid username character raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="test_user",
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "invalid character" in exc_info.value.message

    def test_decode_stats_request_username_leading_hyphen_raises(self) -> None:
        """Test that leading hyphen raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="-testuser",
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "start or end with hyphen" in exc_info.value.message

    def test_decode_stats_request_username_trailing_hyphen_raises(self) -> None:
        """Test that trailing hyphen raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="testuser-",
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "start or end with hyphen" in exc_info.value.message

    def test_decode_stats_request_username_double_hyphen_raises(self) -> None:
        """Test that double hyphen raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="test--user",
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "consecutive hyphens" in exc_info.value.message

    def test_decode_stats_request_invalid_theme_raises(self) -> None:
        """Test that invalid theme raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="testuser",
                theme="invalid-theme",
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "theme must be one of" in exc_info.value.message

    def test_decode_stats_request_invalid_hide_raises(self) -> None:
        """Test that invalid hide value raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="testuser",
                theme=None,
                hide_border=None,
                show_icons=None,
                include_all_commits=None,
                hide="invalid-stat",
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "invalid hide value" in exc_info.value.message

    def test_decode_stats_request_bool_variations(self) -> None:
        """Test various boolean value formats."""
        for true_val in ["true", "1", "yes", "TRUE", "True"]:
            req = decode_stats_request(
                username="testuser",
                theme=None,
                hide_border=true_val,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )
            assert req["hide_border"] is True

        for false_val in ["false", "0", "no", "FALSE", "False"]:
            req = decode_stats_request(
                username="testuser",
                theme=None,
                hide_border=false_val,
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )
            assert req["hide_border"] is False


class TestDecodeLangsRequest:
    """Tests for decode_langs_request function."""

    def test_decode_langs_request_minimal(self) -> None:
        """Test decoding with just username."""
        req = decode_langs_request(
            username="testuser",
            theme=None,
            hide_border=None,
            layout=None,
            langs_count=None,
            hide=None,
            disable_animations=None,
        )

        assert req["username"] == "testuser"
        assert req["theme"] == "default"
        assert req["hide_border"] is False
        assert req["layout"] == "default"
        assert req["langs_count"] == 8
        assert req["hide"] == ()
        assert req["disable_animations"] is False

    def test_decode_langs_request_all_options(self) -> None:
        """Test decoding with all options specified."""
        req = decode_langs_request(
            username="testuser",
            theme="github_dark",
            hide_border="true",
            layout="compact",
            langs_count="10",
            hide="HTML,CSS",
            disable_animations="true",
        )

        assert req["username"] == "testuser"
        assert req["theme"] == "github_dark"
        assert req["hide_border"] is True
        assert req["layout"] == "compact"
        assert req["langs_count"] == 10
        assert req["hide"] == ("HTML", "CSS")
        assert req["disable_animations"] is True

    def test_decode_langs_request_rejects_low_langs_count(self) -> None:
        """Test that langs_count below 1 raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_langs_request(
                username="testuser",
                theme=None,
                hide_border=None,
                layout=None,
                langs_count="0",
                hide=None,
                disable_animations=None,
            )
        assert exc_info.value.http_status == 400
        assert "at least 1" in exc_info.value.message

    def test_decode_langs_request_rejects_high_langs_count(self) -> None:
        """Test that langs_count above 20 raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_langs_request(
                username="testuser",
                theme=None,
                hide_border=None,
                layout=None,
                langs_count="100",
                hide=None,
                disable_animations=None,
            )
        assert exc_info.value.http_status == 400
        assert "at most 20" in exc_info.value.message

    def test_decode_langs_request_invalid_layout_raises(self) -> None:
        """Test that invalid layout raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_langs_request(
                username="testuser",
                theme=None,
                hide_border=None,
                layout="invalid",
                langs_count=None,
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "layout must be one of" in exc_info.value.message

    def test_decode_langs_request_invalid_langs_count_raises(self) -> None:
        """Test that non-integer langs_count raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_langs_request(
                username="testuser",
                theme=None,
                hide_border=None,
                layout=None,
                langs_count="not-a-number",
                hide=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "langs_count must be an integer" in exc_info.value.message

    def test_decode_stats_request_transparent_theme(self) -> None:
        """Test transparent theme is accepted."""
        req = decode_stats_request(
            username="testuser",
            theme="transparent",
            hide_border=None,
            show_icons=None,
            include_all_commits=None,
            hide=None,
            disable_animations=None,
        )
        assert req["theme"] == "transparent"

    def test_decode_stats_request_github_dark_theme(self) -> None:
        """Test github_dark theme is accepted."""
        req = decode_stats_request(
            username="testuser",
            theme="github_dark",
            hide_border=None,
            show_icons=None,
            include_all_commits=None,
            hide=None,
            disable_animations=None,
        )
        assert req["theme"] == "github_dark"

    def test_decode_stats_request_default_theme_explicit(self) -> None:
        """Test explicit 'default' theme is accepted."""
        req = decode_stats_request(
            username="testuser",
            theme="default",
            hide_border=None,
            show_icons=None,
            include_all_commits=None,
            hide=None,
            disable_animations=None,
        )
        assert req["theme"] == "default"

    def test_decode_stats_request_dark_theme(self) -> None:
        """Test dark theme is accepted."""
        req = decode_stats_request(
            username="testuser",
            theme="dark",
            hide_border=None,
            show_icons=None,
            include_all_commits=None,
            hide=None,
            disable_animations=None,
        )
        assert req["theme"] == "dark"

    def test_decode_langs_request_donut_layout(self) -> None:
        """Test donut layout is accepted."""
        req = decode_langs_request(
            username="testuser",
            theme=None,
            hide_border=None,
            layout="donut",
            langs_count=None,
            hide=None,
            disable_animations=None,
        )
        assert req["layout"] == "donut"

    def test_decode_langs_request_pie_layout(self) -> None:
        """Test pie layout is accepted."""
        req = decode_langs_request(
            username="testuser",
            theme=None,
            hide_border=None,
            layout="pie",
            langs_count=None,
            hide=None,
            disable_animations=None,
        )
        assert req["layout"] == "pie"

    def test_decode_langs_request_default_layout_explicit(self) -> None:
        """Test explicit 'default' layout is accepted."""
        req = decode_langs_request(
            username="testuser",
            theme=None,
            hide_border=None,
            layout="default",
            langs_count=None,
            hide=None,
            disable_animations=None,
        )
        assert req["layout"] == "default"

    def test_decode_langs_request_empty_langs_count_uses_default(self) -> None:
        """Test empty langs_count string uses default."""
        req = decode_langs_request(
            username="testuser",
            theme=None,
            hide_border=None,
            layout=None,
            langs_count="  ",
            hide=None,
            disable_animations=None,
        )
        assert req["langs_count"] == 8

    def test_decode_stats_request_invalid_bool_raises(self) -> None:
        """Test invalid boolean parameter raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_stats_request(
                username="testuser",
                theme=None,
                hide_border="maybe",
                show_icons=None,
                include_all_commits=None,
                hide=None,
                disable_animations=None,
            )
        assert exc_info.value.http_status == 400
        assert "must be true/false" in exc_info.value.message


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
