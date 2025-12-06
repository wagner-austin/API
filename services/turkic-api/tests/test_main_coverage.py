"""Tests for main.py to reach 100% coverage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from turkic_api.api.config import Settings
from turkic_api.api.main import _to_json_simple
from turkic_api.api.provider_context import (
    get_queue_from_context as _get_queue_from_context,
)
from turkic_api.api.provider_context import (
    provider_context as _provider_context,
)


def test_to_json_simple_all_value_types() -> None:
    # Test that _to_json_simple handles all primitive types and datetime
    from collections.abc import Mapping

    obj: Mapping[str, str | int | float | bool | datetime | None] = {
        "string": "value",
        "int": 123,
        "float": 45.67,
        "null": None,
        "bool_true": True,
        "bool_false": False,
        "date": datetime(2024, 1, 1, 12, 0, 0),
    }
    result = _to_json_simple(obj)

    # All primitives should be preserved
    assert result["string"] == "value"
    assert result["int"] == 123
    assert result["float"] == 45.67
    assert result["null"] is None
    assert result["bool_true"] is True
    assert result["bool_false"] is False
    # Datetime should be converted to ISO string
    assert result["date"] == "2024-01-01T12:00:00"


def test_get_queue_from_context_no_provider(tmp_path: Path) -> None:
    # Test line 104 - fallback to get_queue when no provider is set
    # Ensure provider_context.queue_provider is None
    original_provider = _provider_context.queue_provider
    _provider_context.queue_provider = None

    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="key",
    )

    # This should call get_queue(settings) on line 104
    queue = _get_queue_from_context(settings)
    if queue is None:
        pytest.fail("expected queue")

    # Restore original
    _provider_context.queue_provider = original_provider
