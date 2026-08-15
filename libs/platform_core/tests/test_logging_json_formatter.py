"""Tests for logging: test_json_formatter_basic."""

from __future__ import annotations

import pytest

from platform_core.json_utils import load_json_str
from platform_core.logging import (
    JsonFormatter,
    stdlib_logging,
)
from platform_core.request_context import request_id_var


def test_json_formatter_basic() -> None:
    """Test JsonFormatter produces valid JSON with required fields."""
    formatter = JsonFormatter(static_fields={}, extra_field_names=[])
    record = stdlib_logging.LogRecord(
        name="test.logger",
        level=stdlib_logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "test message"
    timestamp = parsed["timestamp"]
    assert type(timestamp) is str
    assert timestamp.startswith("20")


def test_json_formatter_with_static_fields() -> None:
    """Test JsonFormatter includes static fields in output."""
    formatter = JsonFormatter(
        static_fields={"service": "test-service", "instance_id": "test-123"},
        extra_field_names=[],
    )
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.WARNING,
        pathname="t.py",
        lineno=1,
        msg="warning",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict

    assert parsed["service"] == "test-service"
    assert parsed["instance_id"] == "test-123"


def test_json_formatter_with_extra_fields() -> None:
    """Test JsonFormatter extracts extra fields from LogRecord."""
    formatter = JsonFormatter(
        static_fields={},
        extra_field_names=["request_id", "latency_ms"],
    )
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-123"
    record.latency_ms = 42

    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict

    assert parsed["request_id"] == "req-123"
    assert parsed["latency_ms"] == 42


def test_json_formatter_includes_request_id_from_context_var() -> None:
    """JsonFormatter should include request_id from context var when present."""
    formatter = JsonFormatter(static_fields={}, extra_field_names=[])
    token = request_id_var.set("ctx-req-1")
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="request with context",
        args=(),
        exc_info=None,
    )

    try:
        output = formatter.format(record)
    finally:
        request_id_var.reset(token)

    parsed = load_json_str(output)
    assert type(parsed) is dict
    assert parsed["request_id"] == "ctx-req-1"


def test_json_formatter_missing_extra_field() -> None:
    """Test JsonFormatter handles missing extra fields gracefully."""
    formatter = JsonFormatter(
        static_fields={},
        extra_field_names=["nonexistent"],
    )
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="msg",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict

    assert "nonexistent" not in parsed


def test_json_formatter_includes_standard_structured_fields() -> None:
    """JsonFormatter should include standard structured fields when present."""
    formatter = JsonFormatter(static_fields={}, extra_field_names=[])
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="structured",
        args=(),
        exc_info=None,
    )
    record.digit = 7
    record.confidence = 0.9

    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict
    assert parsed["digit"] == 7
    assert parsed["confidence"] == 0.9


def test_json_formatter_extra_field_overrides_static() -> None:
    """Test JsonFormatter does not override static fields with extra fields."""
    formatter = JsonFormatter(
        static_fields={"field": "static_value"},
        extra_field_names=["field"],
    )
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="msg",
        args=(),
        exc_info=None,
    )
    record.field = "extra_value"

    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict

    # Static field should NOT be overridden
    assert parsed["field"] == "static_value"


def test_json_formatter_extra_field_invalid_type_skipped() -> None:
    """Test that extra fields with invalid types are skipped."""
    formatter = JsonFormatter(
        static_fields={},
        extra_field_names=["valid_field", "invalid_field"],
    )
    record = stdlib_logging.LogRecord(
        name="test.logger",
        level=stdlib_logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test",
        args=(),
        exc_info=None,
    )
    record.valid_field = "valid_string"
    record.invalid_field = object()

    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict

    # Valid field should be present
    assert parsed["valid_field"] == "valid_string"
    # Invalid field should be skipped
    assert "invalid_field" not in parsed


def test_json_formatter_with_exception() -> None:
    """Test JsonFormatter includes exception info when present."""
    formatter = JsonFormatter(static_fields={}, extra_field_names=[])

    with pytest.raises(ValueError) as raised:
        raise ValueError("test error")
    error = raised.value
    exc_info = (type(error), error, error.__traceback__)

    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.ERROR,
        pathname="t.py",
        lineno=1,
        msg="error occurred",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(record)
    parsed = load_json_str(output)
    assert type(parsed) is dict

    exc_info_value = parsed["exc_info"]
    assert type(exc_info_value) is str
    assert "ValueError: test error" in exc_info_value
    assert "Traceback" in exc_info_value
