"""Tests for logging: test_text_formatter_basic."""

from __future__ import annotations

import pytest

from platform_core.logging import (
    TextFormatter,
    stdlib_logging,
)


def test_text_formatter_basic() -> None:
    """Test TextFormatter produces readable output."""
    formatter = TextFormatter(extra_fields=[])
    record = stdlib_logging.LogRecord(
        name="test.logger",
        level=stdlib_logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)

    assert "[INFO]" in output
    assert "[test.logger]" in output
    assert "test message" in output


def test_text_formatter_with_extra_fields() -> None:
    """Test TextFormatter includes extra fields in output."""
    formatter = TextFormatter(extra_fields=["request_id", "latency_ms"])
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.DEBUG,
        pathname="t.py",
        lineno=1,
        msg="debug",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-456"
    record.latency_ms = 100

    output = formatter.format(record)

    assert "request_id=req-456" in output
    assert "latency_ms=100" in output


def test_text_formatter_missing_extra_field() -> None:
    """Test TextFormatter handles missing extra fields gracefully."""
    formatter = TextFormatter(extra_fields=["nonexistent"])
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

    assert "nonexistent" not in output
    assert "msg" in output


def test_text_formatter_with_exception() -> None:
    """Test TextFormatter includes exception traceback."""
    formatter = TextFormatter(extra_fields=[])

    with pytest.raises(RuntimeError) as raised:
        raise RuntimeError("runtime error")
    error = raised.value
    exc_info = (type(error), error, error.__traceback__)

    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.ERROR,
        pathname="t.py",
        lineno=1,
        msg="error",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(record)

    assert "RuntimeError: runtime error" in output
    assert "Traceback" in output
