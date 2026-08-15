"""Tests for validators: DecodeSkillsRequest."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError

from github_stats_api.api.validators.stats import (
    decode_skills_request,
)


class TestDecodeSkillsRequest:
    """Tests for decode_skills_request function."""

    def test_decode_skills_request_basic(self) -> None:
        """Test decoding with basic skills list."""
        req = decode_skills_request(
            skills="Python,TypeScript,FastAPI",
            theme=None,
            hide_border=None,
            disable_animations=None,
        )

        assert req["skills"] == ("Python", "TypeScript", "FastAPI")
        assert req["theme"] == "default"
        assert req["hide_border"] is False
        assert req["disable_animations"] is False

    def test_decode_skills_request_all_options(self) -> None:
        """Test decoding with all options specified."""
        req = decode_skills_request(
            skills="Python,React",
            theme="cyberpunk",
            hide_border="true",
            disable_animations="true",
        )

        assert req["skills"] == ("Python", "React")
        assert req["theme"] == "cyberpunk"
        assert req["hide_border"] is True
        assert req["disable_animations"] is True

    def test_decode_skills_request_missing_skills_raises(self) -> None:
        """Test that missing skills raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_skills_request(
                skills=None,
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "skills is required" in exc_info.value.message

    def test_decode_skills_request_empty_skills_raises(self) -> None:
        """Test that empty skills raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_skills_request(
                skills="   ",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "skills is required" in exc_info.value.message

    def test_decode_skills_request_only_commas_raises(self) -> None:
        """Test that skills with only commas raises AppError."""
        with pytest.raises(AppError) as exc_info:
            decode_skills_request(
                skills=",,,",
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "skills is required" in exc_info.value.message

    def test_decode_skills_request_too_many_skills_raises(self) -> None:
        """Test that too many skills raises AppError."""
        skills = ",".join([f"Skill{i}" for i in range(21)])
        with pytest.raises(AppError) as exc_info:
            decode_skills_request(
                skills=skills,
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "at most 20 items" in exc_info.value.message

    def test_decode_skills_request_skill_too_long_raises(self) -> None:
        """Test that skill exceeding 30 chars raises AppError."""
        long_skill = "x" * 31
        with pytest.raises(AppError) as exc_info:
            decode_skills_request(
                skills=long_skill,
                theme=None,
                hide_border=None,
                disable_animations=None,
            )

        assert exc_info.value.http_status == 400
        assert "exceeds 30 characters" in exc_info.value.message

    def test_decode_skills_request_trims_whitespace(self) -> None:
        """Test that skills are trimmed of whitespace."""
        req = decode_skills_request(
            skills="  Python  ,  React  ",
            theme=None,
            hide_border=None,
            disable_animations=None,
        )

        assert req["skills"] == ("Python", "React")
