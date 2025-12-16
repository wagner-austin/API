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

    try:
        raise ValueError("test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

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

    try:
        raise RuntimeError("runtime error")
    except RuntimeError:
        import sys

        exc_info = sys.exc_info()

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


def test_setup_rich_logging_returns_logger() -> None:
    """Test setup_rich_logging returns configured root logger."""
    from platform_core.logging import setup_rich_logging

    logger = setup_rich_logging(level="INFO")

    assert logger.level == stdlib_logging.INFO
    assert len(logger.handlers) == 1


def test_setup_rich_logging_with_options() -> None:
    """Test setup_rich_logging accepts all options."""
    from platform_core.logging import setup_rich_logging

    logger = setup_rich_logging(
        level="DEBUG",
        show_time=False,
        show_path=True,
    )

    assert logger.level == stdlib_logging.DEBUG


def test_setup_rich_logging_silences_third_party() -> None:
    """Test setup_rich_logging sets WARNING level for noisy loggers."""
    from platform_core.logging import setup_rich_logging

    setup_rich_logging(level="DEBUG")

    assert stdlib_logging.getLogger("urllib3").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("httpx").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("httpcore").level == stdlib_logging.WARNING
    assert stdlib_logging.getLogger("optuna").level == stdlib_logging.WARNING


def test_get_rich_console_after_setup() -> None:
    """Test get_rich_console returns console after setup."""
    from platform_core.logging import (
        RichConsoleProtocol,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()

    # Verify it implements the protocol by calling print
    typed_console: RichConsoleProtocol = console
    typed_console.print("test output")  # Should not raise


def test_get_rich_console_before_setup_raises() -> None:
    """Test get_rich_console raises RuntimeError before setup."""
    import platform_core.logging as log_mod
    from platform_core.logging import get_rich_console

    # Reset module state to simulate no setup
    original = log_mod._rich_console
    log_mod._rich_console = None

    try:
        raised = False
        try:
            get_rich_console()
        except RuntimeError as e:
            raised = True
            assert "setup_rich_logging()" in str(e)

        assert raised, "Expected RuntimeError"
    finally:
        log_mod._rich_console = original


def test_create_rich_table_basic() -> None:
    """Test create_rich_table creates a table with columns and rows."""
    from platform_core.logging import (
        RichTableProtocol,
        create_rich_table,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    table = create_rich_table(title="Test Table")

    # Verify Protocol interface by using methods
    typed_table: RichTableProtocol = table
    typed_table.add_column("Col1", style="cyan")
    typed_table.add_column("Col2", justify="right")
    typed_table.add_row("val1", "val2")

    # Verify table can be printed (tests actual functionality)
    console.print(table)


def test_create_rich_table_no_header() -> None:
    """Test create_rich_table with show_header=False."""
    from platform_core.logging import (
        create_rich_table,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    table = create_rich_table(show_header=False)
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("k1", "v1")

    # Verify table can be printed
    console.print(table)


def test_create_rich_panel_basic() -> None:
    """Test create_rich_panel creates a panel."""
    from platform_core.logging import (
        create_rich_panel,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    panel = create_rich_panel("[bold]Content[/bold]", title="Title")

    # Verify panel can be printed (tests actual functionality)
    console.print(panel)


def test_create_rich_panel_no_title() -> None:
    """Test create_rich_panel without title."""
    from platform_core.logging import (
        create_rich_panel,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    panel = create_rich_panel("Simple content")

    # Verify panel can be printed
    console.print(panel)


def test_create_rich_progress_context_manager() -> None:
    """Test create_rich_progress as context manager."""
    from platform_core.logging import (
        RichProgressProtocol,
        create_rich_progress,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    progress = create_rich_progress(console)

    # Verify Protocol interface by using all methods
    typed_progress: RichProgressProtocol = progress
    task_id_captured = -1

    with typed_progress as p:
        task_id = p.add_task("Testing...", total=10.0)
        task_id_captured = task_id
        p.update(task_id, description="Updated")
        p.advance(task_id, advance=5.0)

    # Task ID should be a valid integer (0 for first task)
    assert task_id_captured == 0


def test_create_rich_spinner_progress_context_manager() -> None:
    """Test create_rich_spinner_progress as context manager."""
    from platform_core.logging import (
        RichProgressProtocol,
        create_rich_spinner_progress,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    progress = create_rich_spinner_progress(console)

    # Verify Protocol interface
    typed_progress: RichProgressProtocol = progress
    task_id_captured = -1

    with typed_progress as p:
        task_id = p.add_task("Spinner test...")
        task_id_captured = task_id
        p.advance(task_id)

    # Task ID should be a valid integer (0 for first task)
    assert task_id_captured == 0


def test_rich_console_print_table() -> None:
    """Test rich console can print a table."""
    from platform_core.logging import (
        create_rich_table,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    table = create_rich_table(title="Print Test")
    table.add_column("A")
    table.add_row("value")

    # Should not raise
    console.print(table)


def test_rich_console_print_panel() -> None:
    """Test rich console can print a panel."""
    from platform_core.logging import (
        create_rich_panel,
        get_rich_console,
        setup_rich_logging,
    )

    setup_rich_logging()
    console = get_rich_console()
    panel = create_rich_panel("Panel content")

    # Should not raise
    console.print(panel)
