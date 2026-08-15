"""Tests for logging: test_setup_logging_json_format."""

from __future__ import annotations

import io

from platform_core.json_utils import load_json_str
from platform_core.logging import (
    JsonFormatter,
    LogEventFields,
    TextFormatter,
    get_logger,
    setup_logging,
    stdlib_logging,
)


def test_setup_logging_json_format() -> None:
    """Test setup_logging configures JSON formatter correctly."""
    logger = setup_logging(
        level="INFO",
        format_mode="json",
        service_name="test-service",
        instance_id="test-instance",
        extra_fields=None,
    )

    assert logger.level == stdlib_logging.INFO
    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert type(handler.formatter) is JsonFormatter


def test_setup_logging_text_format() -> None:
    """Test setup_logging configures text formatter correctly."""
    logger = setup_logging(
        level="DEBUG",
        format_mode="text",
        service_name="test-service",
        instance_id=None,
        extra_fields=["field1"],
    )

    assert logger.level == stdlib_logging.DEBUG
    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert type(handler.formatter) is TextFormatter


def test_setup_logging_clears_handlers() -> None:
    """Test setup_logging clears existing handlers."""
    root = stdlib_logging.getLogger()

    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="svc1",
        instance_id=None,
        extra_fields=None,
    )
    count_after_first = len(root.handlers)

    setup_logging(
        level="WARNING",
        format_mode="text",
        service_name="svc2",
        instance_id=None,
        extra_fields=None,
    )
    count_after_second = len(root.handlers)

    # Should have exactly 1 handler after each setup
    assert count_after_first == 1
    assert count_after_second == 1


def test_setup_logging_debug() -> None:
    """Test setup_logging with DEBUG level."""
    logger = setup_logging(
        level="DEBUG",
        format_mode="json",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )
    assert logger.level == stdlib_logging.DEBUG


def test_setup_logging_info() -> None:
    """Test setup_logging with INFO level."""
    logger = setup_logging(
        level="INFO",
        format_mode="json",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )
    assert logger.level == stdlib_logging.INFO


def test_setup_logging_warning() -> None:
    """Test setup_logging with WARNING level."""
    logger = setup_logging(
        level="WARNING",
        format_mode="json",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )
    assert logger.level == stdlib_logging.WARNING


def test_setup_logging_error() -> None:
    """Test setup_logging with ERROR level."""
    logger = setup_logging(
        level="ERROR",
        format_mode="json",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )
    assert logger.level == stdlib_logging.ERROR


def test_setup_logging_critical() -> None:
    """Test setup_logging with CRITICAL level."""
    logger = setup_logging(
        level="CRITICAL",
        format_mode="json",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )
    assert logger.level == stdlib_logging.CRITICAL


def test_setup_logging_auto_instance_id() -> None:
    """Test setup_logging generates instance_id when None."""
    logger = setup_logging(
        level="INFO",
        format_mode="json",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )

    # Capture output with existing formatter from setup_logging
    buf = io.StringIO()
    handler = stdlib_logging.StreamHandler(buf)
    handler.setFormatter(logger.handlers[0].formatter)

    logger.handlers.clear()
    logger.addHandler(handler)

    logger.info("test")

    output = buf.getvalue()
    parsed = load_json_str(output)
    assert type(parsed) is dict

    # Should have auto-generated instance_id
    inst_id = parsed["instance_id"]
    assert type(inst_id) is str
    assert "-" in inst_id  # Format: hostname-pid


def test_setup_logging_with_extra_fields() -> None:
    """Test setup_logging with extra fields configuration."""
    logger = setup_logging(
        level="INFO",
        format_mode="json",
        service_name="test",
        instance_id="inst-1",
        extra_fields=["custom_field"],
    )

    buf = io.StringIO()
    handler = stdlib_logging.StreamHandler(buf)
    handler.setFormatter(logger.handlers[0].formatter)
    logger.handlers.clear()
    logger.addHandler(handler)

    # Create a custom logger adapter or direct record manipulation
    record = stdlib_logging.LogRecord(
        name="test",
        level=stdlib_logging.INFO,
        pathname="t.py",
        lineno=1,
        msg="msg",
        args=(),
        exc_info=None,
    )
    record.custom_field = "custom_value"
    logger.handle(record)

    output = buf.getvalue()
    parsed = load_json_str(output)
    assert type(parsed) is dict

    assert parsed["custom_field"] == "custom_value"


def test_get_logger() -> None:
    """Test get_logger returns a logger with correct name."""
    logger = get_logger("my.module")

    assert logger.name == "my.module"
    assert type(logger) is stdlib_logging.Logger


def test_get_logger_different_names() -> None:
    """Test get_logger returns different instances for different names."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    assert logger1 is not logger2
    assert logger1.name == "module1"
    assert logger2.name == "module2"


def test_log_event_fields_typeddict() -> None:
    """Test LogEventFields TypedDict has expected structure."""
    # This validates the TypedDict can be constructed with valid fields
    fields: LogEventFields = {
        "latency_ms": 100,
        "request_id": "req-123",
        "digit": 5,
        "confidence": 0.95,
        "model_id": "model-v1",
        "uncertain": False,
    }

    assert fields["latency_ms"] == 100
    assert fields["request_id"] == "req-123"
    assert fields["digit"] == 5
    assert fields["confidence"] == 0.95
    assert fields["model_id"] == "model-v1"
    assert fields["uncertain"] is False


def test_log_event_fields_partial() -> None:
    """Test LogEventFields allows partial construction (total=False)."""
    # Should allow constructing with only some fields
    fields: LogEventFields = {"latency_ms": 50}

    assert fields["latency_ms"] == 50
    assert "request_id" not in fields


def test_setup_logging_silences_third_party() -> None:
    """Test setup_logging sets WARNING level for noisy third-party loggers."""
    setup_logging(
        level="DEBUG",
        format_mode="json",
        service_name="test",
        instance_id=None,
        extra_fields=None,
    )

    assert stdlib_logging.getLogger("urllib3").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("httpx").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("httpcore").level == stdlib_logging.WARNING


def test_load_queue_handler_factory_returns_factory() -> None:
    """Test load_queue_handler_factory returns the QueueHandler class."""
    import logging.handlers

    from platform_core.logging import QueueHandlerFactory, load_queue_handler_factory

    factory = load_queue_handler_factory()
    expected: QueueHandlerFactory = logging.handlers.QueueHandler
    assert factory is expected


def test_load_queue_listener_factory_returns_factory() -> None:
    """Test load_queue_listener_factory returns the QueueListener class."""
    import logging.handlers

    from platform_core.logging import QueueListenerFactory, load_queue_listener_factory

    factory = load_queue_listener_factory()
    expected: QueueListenerFactory = logging.handlers.QueueListener
    assert factory is expected


# =============================================================================
# Rich Logging Tests
# =============================================================================
