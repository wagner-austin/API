"""Tests for API models to reach 100% coverage."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError

from turkic_api.api import models


def test_decode_source_literal_invalid_after_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test defensive error path in _decode_source_literal line 117
    # Mock _decode_required_literal to return an unexpected value
    def mock_decode(
        _v: str | int | float | bool | None | list[str] | dict[str, str],
        _f: str,
        _a: frozenset[str],
    ) -> str:
        return "invalid"

    monkeypatch.setattr(models, "_decode_required_literal", mock_decode)

    with pytest.raises(AppError) as exc_info:
        models._decode_source_literal("anything")
    assert exc_info.value.http_status == 400
    assert "Invalid source" in exc_info.value.message


def test_decode_language_literal_invalid_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test defensive error path in _decode_language_literal line 132
    def mock_decode(
        _v: str | int | float | bool | None | list[str] | dict[str, str],
        _f: str,
        _a: frozenset[str],
    ) -> str:
        return "invalid"

    monkeypatch.setattr(models, "_decode_required_literal", mock_decode)

    with pytest.raises(AppError) as exc_info:
        models._decode_language_literal("anything")
    assert exc_info.value.http_status == 400
    assert "Invalid language" in exc_info.value.message


def test_decode_job_create_map_lookup_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test lines 159 and 165 - defensive map lookup failures
    # First test source map failure
    original_source_map = models._SOURCE_MAP

    monkeypatch.setattr(models, "_SOURCE_MAP", {})

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

    # Restore source map and break language map
    monkeypatch.setattr(models, "_SOURCE_MAP", original_source_map)
    monkeypatch.setattr(models, "_LANGUAGE_MAP", {})

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


def test_parse_job_create_with_list_values() -> None:
    # Test the else branch (line 215) - handle list values in payload
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
    # Test line 260 - JSON is not a dict
    with pytest.raises(ValueError, match="Expected JSON object"):
        models.parse_job_response_json('"not a dict"')


def test_parse_job_response_json_invalid_status() -> None:
    # Test line 272 - invalid job status value
    with pytest.raises(ValueError, match="Invalid job status"):
        models.parse_job_response_json(
            '{"status": "invalid", "job_id": "x", "user_id": 42, '
            '"created_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_status_json_not_dict() -> None:
    # Test line 288 - JSON is not a dict
    with pytest.raises(ValueError, match="Expected JSON object"):
        models.parse_job_status_json("123")


def test_parse_job_status_json_invalid_status() -> None:
    # Test line 320 - invalid job status value in result
    with pytest.raises(ValueError, match="Invalid job status"):
        models.parse_job_status_json(
            '{"status": "invalid", "job_id": "x", "user_id": 42, "progress": 0, '
            '"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}'
        )


def test_decode_source_literal_all_values() -> None:
    # Test lines 114, 116, 119 - all source literal branches
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


def test_decode_script_literal_all_values() -> None:
    # Test lines 136-145 - all script literal branches including None
    assert models._decode_script_literal(None) is None
    assert models._decode_script_literal("Latn") == "Latn"
    assert models._decode_script_literal("Cyrl") == "Cyrl"
    assert models._decode_script_literal("Arab") == "Arab"


def test_decode_script_literal_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test line 145 - defensive return None when validation passes but no branches match
    # Mock _decode_optional_literal to not validate, allowing an unexpected value through
    def mock_decode_optional(
        _v: str | int | float | bool | None | list[str] | dict[str, str],
        _f: str,
        _a: frozenset[str],
    ) -> str | None:
        return "unexpected" if isinstance(_v, str) else None

    monkeypatch.setattr(models, "_decode_optional_literal", mock_decode_optional)

    result = models._decode_script_literal("unexpected")
    assert result is None


def test_decode_job_create_from_unknown_user_id_not_int() -> None:
    """Cover line 172: user_id must be an integer in _decode_job_create_from_unknown."""
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
    """Cover line 172: user_id None triggers the error."""
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
    """Cover line 269: user_id must be an integer in parse_job_response_json."""
    with pytest.raises(ValueError, match="user_id must be an integer"):
        models.parse_job_response_json(
            '{"job_id": "x", "user_id": "42", "status": "queued", '
            '"created_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_response_json_user_id_null() -> None:
    """Cover line 269: user_id null is not an integer."""
    with pytest.raises(ValueError, match="user_id must be an integer"):
        models.parse_job_response_json(
            '{"job_id": "x", "user_id": null, "status": "queued", '
            '"created_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_status_json_user_id_not_int() -> None:
    """Cover line 301: user_id must be an integer in parse_job_status_json."""
    with pytest.raises(ValueError, match="user_id must be an integer"):
        models.parse_job_status_json(
            '{"job_id": "x", "user_id": "42", "status": "queued", "progress": 0, '
            '"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}'
        )


def test_parse_job_status_json_user_id_null() -> None:
    """Cover line 301: user_id null is not an integer."""
    with pytest.raises(ValueError, match="user_id must be an integer"):
        models.parse_job_status_json(
            '{"job_id": "x", "user_id": null, "status": "queued", "progress": 0, '
            '"created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}'
        )
