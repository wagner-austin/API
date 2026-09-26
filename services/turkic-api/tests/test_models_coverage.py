"""Tests for API models to reach 100% coverage."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError

from turkic_api.api import models
from turkic_api.core.models import Language, Script, Source

_INVALID_STATUS_MESSAGE = (
    "Invalid status 'invalid': must be one of 'queued', 'processing', 'completed', 'failed'"
)


def test_parse_job_create_ignores_unknown_fields() -> None:
    result = models.parse_job_create(
        b'{"user_id": 42, "source": "oscar", "language": "kk", "max_sentences": 1,'
        b' "transliterate": true, "confidence_threshold": 0.9,'
        b' "extra_list": ["item1", 123, null, true, {"nested": [1]}]}'
    )
    assert result["source"] is Source.OSCAR
    assert result["language"] is Language.KK


def test_parse_job_response_json_not_dict() -> None:
    # Test JSON is not a dict
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        models.parse_job_response_json('"not a dict"')


def test_parse_job_response_json_invalid_status() -> None:
    # Test invalid job status value
    with pytest.raises(JSONTypeError) as excinfo:
        models.parse_job_response_json(
            '{"status": "invalid", "job_id": "x", "user_id": 42, '
            '"created_at": "2024-01-01T00:00:00"}'
        )
    assert str(excinfo.value) == _INVALID_STATUS_MESSAGE


def test_parse_job_status_json_not_dict() -> None:
    # Test JSON is not a dict
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        models.parse_job_status_json("123")


def test_parse_job_status_json_invalid_status() -> None:
    # Test invalid job status value in result
    with pytest.raises(JSONTypeError) as excinfo:
        models.parse_job_status_json(
            '{"status": "invalid", "job_id": "x", "user_id": 42, "progress": 0, '
            '"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}'
        )
    assert str(excinfo.value) == _INVALID_STATUS_MESSAGE


def test_decode_source_literal_all_values() -> None:
    for source in Source:
        assert models._decode_source_literal(source.value) is source


def test_decode_language_literal_all_values() -> None:
    for language in Language:
        assert models._decode_language_literal(language.value) is language


def test_decode_script_literal_all_values() -> None:
    assert models._decode_script_literal(None) is None
    for script in Script:
        assert models._decode_script_literal(script.value) is script


def test_decode_job_create_from_unknown_user_id_not_int() -> None:
    """Cover user_id must be an integer in _decode_job_create_from_unknown."""
    with pytest.raises(AppError) as exc_info:
        models._decode_job_create_from_unknown(
            {
                "user_id": "42",  # string, not int
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            }
        )
    assert exc_info.value.http_status == 400
    assert "user_id must be an integer" in exc_info.value.message


def test_decode_job_create_from_unknown_user_id_none() -> None:
    """Cover user_id None triggers the error."""
    with pytest.raises(AppError) as exc_info:
        models._decode_job_create_from_unknown(
            {
                "user_id": None,
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": True,
                "confidence_threshold": 0.9,
            }
        )
    assert exc_info.value.http_status == 400
    assert "user_id must be an integer" in exc_info.value.message


def test_parse_job_response_json_user_id_not_int() -> None:
    """Cover user_id must be an integer in parse_job_response_json."""
    with pytest.raises(JSONTypeError, match="user_id must be an integer"):
        models.parse_job_response_json(
            '{"job_id": "x", "user_id": "42", "status": "queued", '
            '"created_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_response_json_user_id_null() -> None:
    """Cover user_id null is not an integer."""
    with pytest.raises(JSONTypeError, match="user_id must be an integer"):
        models.parse_job_response_json(
            '{"job_id": "x", "user_id": null, "status": "queued", '
            '"created_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_status_json_user_id_not_int() -> None:
    """Cover user_id must be an integer in parse_job_status_json."""
    with pytest.raises(JSONTypeError, match="user_id must be an integer"):
        models.parse_job_status_json(
            '{"job_id": "x", "user_id": "42", "status": "queued", "progress": 0, '
            '"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_status_json_user_id_null() -> None:
    """Cover user_id null is not an integer."""
    with pytest.raises(JSONTypeError, match="user_id must be an integer"):
        models.parse_job_status_json(
            '{"job_id": "x", "user_id": null, "status": "queued", "progress": 0, '
            '"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}'
        )
