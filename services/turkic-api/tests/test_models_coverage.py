"""Tests for API models to reach 100% coverage."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue

from turkic_api import _test_hooks
from turkic_api.api import models


def test_decode_source_literal_invalid_after_validation() -> None:
    # Test defensive error path in _decode_source_literal
    # Override decode_required_literal to return an unexpected value
    orig_decode = _test_hooks.decode_required_literal

    def mock_decode(
        val: JSONValue,
        field: str,
        allowed: frozenset[str],
    ) -> str:
        return "invalid"

    _test_hooks.decode_required_literal = mock_decode
    try:
        with pytest.raises(AppError) as exc_info:
            models._decode_source_literal("anything")
        assert exc_info.value.http_status == 400
        assert "Invalid source" in exc_info.value.message
    finally:
        _test_hooks.decode_required_literal = orig_decode


def test_decode_language_literal_invalid_after_validation() -> None:
    # Test defensive error path in _decode_language_literal
    orig_decode = _test_hooks.decode_required_literal

    def mock_decode(
        val: JSONValue,
        field: str,
        allowed: frozenset[str],
    ) -> str:
        return "invalid"

    _test_hooks.decode_required_literal = mock_decode
    try:
        with pytest.raises(AppError) as exc_info:
            models._decode_language_literal("anything")
        assert exc_info.value.http_status == 400
        assert "Invalid language" in exc_info.value.message
    finally:
        _test_hooks.decode_required_literal = orig_decode


def test_decode_job_create_map_lookup_failures() -> None:
    # Test defensive map lookup failures
    orig_source_map = _test_hooks.source_map
    orig_language_map = _test_hooks.language_map

    # First test source map failure (empty map = None lookup result)
    _test_hooks.source_map = {}
    try:
        with pytest.raises(AppError) as exc_info:
            models._decode_job_create_from_unknown(
                {
                    "user_id": 42,
                    "source": "oscar",
                    "language": "kk",
                    "max_sentences": 1,
                    "transliterate": True,
                    "confidence_threshold": 0.9,
                }
            )
        assert exc_info.value.http_status == 400
        assert "Invalid source" in exc_info.value.message
    finally:
        _test_hooks.source_map = orig_source_map

    # Now test language map failure (empty map = None lookup result)
    _test_hooks.language_map = {}
    try:
        with pytest.raises(AppError) as exc_info:
            models._decode_job_create_from_unknown(
                {
                    "user_id": 42,
                    "source": "oscar",
                    "language": "kk",
                    "max_sentences": 1,
                    "transliterate": True,
                    "confidence_threshold": 0.9,
                }
            )
        assert exc_info.value.http_status == 400
        assert "Invalid language" in exc_info.value.message
    finally:
        _test_hooks.language_map = orig_language_map


def test_decode_source_literal_final_raise() -> None:
    """Test defensive final raise in _decode_source_literal (line 149)."""
    # Need to make source_map return a non-None value that isn't oscar/wikipedia/culturax
    orig_source_map = _test_hooks.source_map
    _test_hooks.source_map = {"oscar": "unknown_source"}  # Pass None check, fail literal narrow
    try:
        with pytest.raises(AppError) as exc_info:
            models._decode_source_literal("oscar")
        assert exc_info.value.http_status == 400
        assert "Invalid source" in exc_info.value.message
    finally:
        _test_hooks.source_map = orig_source_map


def test_decode_language_literal_final_raise() -> None:
    """Test defensive final raise in _decode_language_literal (line 182)."""
    # Need to make language_map return a non-None value that isn't one of the literals
    orig_language_map = _test_hooks.language_map
    _test_hooks.language_map = {"kk": "unknown_lang"}  # Pass None check, fail literal narrow
    try:
        with pytest.raises(AppError) as exc_info:
            models._decode_language_literal("kk")
        assert exc_info.value.http_status == 400
        assert "Invalid language" in exc_info.value.message
    finally:
        _test_hooks.language_map = orig_language_map


def test_parse_job_create_with_list_values() -> None:
    # Test the else branch - handle list values in payload
    from turkic_api.api.types import JsonDict

    payload: JsonDict = {
        "user_id": 42,
        "source": "oscar",
        "language": "kk",
        "max_sentences": 1,
        "transliterate": True,
        "confidence_threshold": 0.9,
        "extra_list": ["item1", "item2", 123, None, True],
    }
    result = models.parse_job_create(payload)
    assert result["source"] == "oscar"
    assert result["language"] == "kk"


def test_parse_job_response_json_not_dict() -> None:
    # Test JSON is not a dict
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        models.parse_job_response_json('"not a dict"')


def test_parse_job_response_json_invalid_status() -> None:
    # Test invalid job status value
    with pytest.raises(JSONTypeError, match="Invalid job status"):
        models.parse_job_response_json(
            '{"status": "invalid", "job_id": "x", "user_id": 42, '
            '"created_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_status_json_not_dict() -> None:
    # Test JSON is not a dict
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        models.parse_job_status_json("123")


def test_parse_job_status_json_invalid_status() -> None:
    # Test invalid job status value in result
    with pytest.raises(JSONTypeError, match="Invalid job status"):
        models.parse_job_status_json(
            '{"status": "invalid", "job_id": "x", "user_id": 42, "progress": 0, '
            '"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}'
        )


def test_decode_source_literal_all_values() -> None:
    # Test all source literal branches
    assert models._decode_source_literal("oscar") == "oscar"
    assert models._decode_source_literal("wikipedia") == "wikipedia"
    assert models._decode_source_literal("culturax") == "culturax"


def test_decode_language_literal_all_values() -> None:
    # Test all language literal branches
    assert models._decode_language_literal("kk") == "kk"
    assert models._decode_language_literal("ky") == "ky"
    assert models._decode_language_literal("uz") == "uz"
    assert models._decode_language_literal("tr") == "tr"
    assert models._decode_language_literal("ug") == "ug"
    assert models._decode_language_literal("fi") == "fi"
    assert models._decode_language_literal("az") == "az"
    assert models._decode_language_literal("en") == "en"


def test_decode_script_literal_all_values() -> None:
    # Test all script literal branches including None
    assert models._decode_script_literal(None) is None
    assert models._decode_script_literal("Latn") == "Latn"
    assert models._decode_script_literal("Cyrl") == "Cyrl"
    assert models._decode_script_literal("Arab") == "Arab"


def test_decode_script_literal_fallback() -> None:
    # Test defensive return None when validation passes but no branches match
    orig_decode = _test_hooks.decode_optional_literal

    def mock_decode_optional(
        val: JSONValue,
        field: str,
        allowed: frozenset[str],
    ) -> str | None:
        return "unexpected" if isinstance(val, str) else None

    _test_hooks.decode_optional_literal = mock_decode_optional
    try:
        result = models._decode_script_literal("unexpected")
        assert result is None
    finally:
        _test_hooks.decode_optional_literal = orig_decode


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
