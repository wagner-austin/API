from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as _mp
import socket
import sys
import time
from types import TracebackType
from typing import Literal, Protocol, TypedDict

from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.request_context import request_id_var

LogFormat = Literal["json", "text"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_LogRecordValue = (
    JSONValue
    | tuple[
        type[BaseException] | BaseException | None,
        BaseException | None,
        TracebackType | None,
    ]
    | tuple[str | int | float | bool | None, ...]
    | logging.LogRecord
    | logging.Logger
    | logging.Handler
    | logging.Filter
    | logging.Formatter
)


class _OsModule(Protocol):
    """Protocol for os module to avoid Any from __import__."""

    def getpid(self) -> int: ...


class _LogRecordMapping(Protocol):
    """Minimal mapping interface for LogRecord.__dict__ without Any leakage."""

    def __contains__(self, key: str) -> bool: ...

    def __getitem__(self, key: str) -> _LogRecordValue: ...


class _MissingValue:
    """Sentinel for absent or invalid LogRecord attributes."""

    __slots__ = ()


_MISSING = _MissingValue()


def _get_json_record_value(record: logging.LogRecord, field_name: str) -> JSONValue | _MissingValue:
    """Fetch a record attribute and validate it is JSON-compatible."""
    record_mapping: _LogRecordMapping = record.__dict__
    if field_name not in record_mapping:
        return _MISSING
    raw_value = record_mapping[field_name]
    if isinstance(raw_value, (dict, list, str, int, float, bool)) or raw_value is None:
        return raw_value
    return _MISSING


class JsonFormatter(logging.Formatter):
    """Standard JSON formatter for all services.

    Produces consistent structured logs with:
    - ISO8601 timestamp (UTC)
    - level, logger, message
    - Optional static fields (service, instance_id, etc.)
    - Optional extra fields extracted from LogRecord
    - Exception info if present
    """

    def __init__(
        self,
        *,
        static_fields: dict[str, str],
        extra_field_names: list[str],
    ) -> None:
        """Initialize JSON formatter.

        Args:
            static_fields: Fields to include in every log record (e.g., service name, instance_id)
            extra_field_names: Names of extra fields to extract from LogRecord attributes
        """
        super().__init__()
        self._static = static_fields
        self._extra_fields = extra_field_names

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        # Explicit type for payload to accept JSON-compatible values
        payload: dict[str, JSONValue] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add static fields
        for key in self._static:
            payload[key] = self._static[key]

        # Attach request_id when present in context
        rid = request_id_var.get()
        if isinstance(rid, str) and rid != "":
            payload["request_id"] = rid

        # Add dynamic extra fields if they exist on the record
        for field_name in self._extra_fields:
            if field_name in self._static:
                continue
            field_value = _get_json_record_value(record, field_name)
            if isinstance(field_value, _MissingValue):
                continue
            payload[field_name] = field_value

        # Include standard structured fields when present on the record
        # Includes ML training progress fields for visibility
        for field_name in (
            "latency_ms",
            "digit",
            "confidence",
            "model_id",
            "uncertain",
            # Training progress fields
            "round",
            "train_auc",
            "val_auc",
            "best_val_auc",
            "n_samples",
            "n_features",
            "samples_train",
            "samples_val",
            "samples_test",
            "early_stopped",
            "config",
            "test_auc",
        ):
            if field_name in payload:
                continue
            field_value = _get_json_record_value(record, field_name)
            if isinstance(field_value, _MissingValue):
                continue
            payload[field_name] = field_value

        # Add exception info if present
        if record.exc_info is not None:
            payload["exc_info"] = self.formatException(record.exc_info)

        return dump_json_str(payload, compact=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development/debugging.

    Format: [timestamp] [LEVEL] [logger] [extra_fields] message
    """

    def __init__(self, *, extra_fields: list[str]) -> None:
        """Initialize text formatter.

        Args:
            extra_fields: Names of extra fields to show in output
        """
        super().__init__()
        self._extra_fields = extra_fields

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as human-readable text."""
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        parts: list[str] = [
            f"[{timestamp}]",
            f"[{record.levelname}]",
            f"[{record.name}]",
        ]

        # Add extra fields if configured
        for field_name in self._extra_fields:
            if hasattr(record, field_name):
                # Get attr and convert to string for display
                attr_value: str | int | float | bool | None = getattr(record, field_name)
                value_str: str = str(attr_value)
                parts.append(f"{field_name}={value_str}")

        parts.append(record.getMessage())

        line = " ".join(parts)

        # Add exception info if present
        if record.exc_info is not None:
            line = line + "\n" + self.formatException(record.exc_info)

        return line


def _compute_instance_id() -> str:
    """Generate a stable instance ID from hostname and PID."""
    host = socket.gethostname().split(".")[0]
    os_mod = __import__("os")
    # Use Protocol to narrow the type from Any
    os_protocol: _OsModule = os_mod
    pid_value = os_protocol.getpid()
    return f"{host}-{pid_value}"


def _level_to_int(level: LogLevel) -> int:
    """Convert string log level to integer constant."""
    level_map: dict[LogLevel, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map[level]


def setup_logging(
    *,
    level: LogLevel,
    format_mode: LogFormat,
    service_name: str,
    instance_id: str | None,
    extra_fields: list[str] | None,
) -> logging.Logger:
    """Setup standardized logging for a service.

    Configures the root logger with either JSON or text formatting.
    Clears existing handlers to ensure clean state.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_mode: Output format ("json" for production, "text" for dev)
        service_name: Service name to include in all logs
        instance_id: Instance ID (auto-generated if None)
        extra_fields: List of extra field names to extract from records (empty list if None)

    Returns:
        Configured root logger

    Example:
        >>> from platform_core.logging import setup_logging
        >>> logger = setup_logging(
        ...     level="INFO",
        ...     format_mode="json",
        ...     service_name="my-api",
        ...     instance_id=None,
        ...     extra_fields=None
        ... )
        >>> logger.info("Server started")
    """
    log_level = _level_to_int(level)

    # Configure root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    # Build static fields
    computed_instance_id = instance_id if instance_id is not None else _compute_instance_id()
    static_fields: dict[str, str] = {
        "service": service_name,
        "instance_id": computed_instance_id,
    }

    # Normalize extra fields
    extra_field_names = extra_fields if extra_fields is not None else []

    # Create handler with appropriate formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    if format_mode == "json":
        handler.setFormatter(
            JsonFormatter(static_fields=static_fields, extra_field_names=extra_field_names)
        )
    else:
        handler.setFormatter(TextFormatter(extra_fields=extra_field_names))

    root.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return root


# Expose stdlib logging module for typed test utilities.
stdlib_logging = logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance by name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> from platform_core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


class LogEventFields(TypedDict, total=False):
    """Optional structured fields for log events.

    Services can extend this by creating their own TypedDict with additional fields.
    This base TypedDict provides commonly used fields across services.
    """

    latency_ms: int
    request_id: str
    digit: int
    confidence: float
    model_id: str
    uncertain: bool
    # ML training progress fields
    round: int
    train_auc: float
    val_auc: float
    best_val_auc: float
    n_samples: int
    n_features: int
    samples_train: int
    samples_val: int
    samples_test: int
    early_stopped: bool
    config: dict[str, float | int]
    test_auc: float


# =============================================================================
# Multiprocessing Queue Logging Support
# =============================================================================
# These types support multiprocessing IPC logging via QueueHandler/QueueListener.
# Used by calibration runners and other multiprocessing code.


class QueueListenerProtocol(Protocol):
    """Protocol for logging.handlers.QueueListener."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


class LogRecordQueueProtocol(Protocol):
    """Protocol for multiprocessing.Queue[logging.LogRecord].

    This protocol allows typed access to log record queues without
    exposing the generic Queue type.
    """

    def put(
        self,
        obj: logging.LogRecord,
        block: bool = True,
        timeout: float | None = None,
    ) -> None: ...

    def get(self, block: bool = True, timeout: float | None = None) -> logging.LogRecord: ...

    def empty(self) -> bool: ...


class QueueHandlerFactory(Protocol):
    """Protocol for QueueHandler constructor."""

    def __call__(self, queue: _mp.Queue[logging.LogRecord]) -> logging.Handler: ...


class QueueListenerFactory(Protocol):
    """Protocol for QueueListener constructor."""

    def __call__(
        self,
        queue: _mp.Queue[logging.LogRecord],
        *handlers: logging.Handler,
        respect_handler_level: bool = False,
    ) -> QueueListenerProtocol: ...


def load_queue_handler_factory() -> QueueHandlerFactory:
    """Load QueueHandler constructor from logging.handlers.

    Returns a factory that creates QueueHandler instances for sending
    log records to a multiprocessing queue.
    """
    factory: QueueHandlerFactory = logging.handlers.QueueHandler
    return factory


def load_queue_listener_factory() -> QueueListenerFactory:
    """Load QueueListener constructor from logging.handlers.

    Returns a factory that creates QueueListener instances for receiving
    log records from a multiprocessing queue and dispatching to handlers.
    """
    factory: QueueListenerFactory = logging.handlers.QueueListener
    return factory


# =============================================================================
# Rich Console Logging Support
# =============================================================================
# These types support rich console output for CLI scripts.
# Uses RichHandler for styled logging and exposes console for Tables/Panels.


class RichRenderableProtocol(Protocol):
    """Protocol for rich renderable objects (Tables, Panels, etc)."""

    ...


class RichConsoleProtocol(Protocol):
    """Protocol for rich.console.Console.

    Provides typed access to rich console for rendering Tables, Panels,
    and other rich renderables outside of standard logging.
    """

    def print(
        self,
        *args: str | RichRenderableProtocol,
        style: str | None = None,
        **kwargs: str,
    ) -> None:
        """Print to console with optional styling."""
        ...


class RichTableProtocol(Protocol):
    """Protocol for rich.table.Table."""

    def add_column(
        self,
        header: str,
        style: str | None = None,
        justify: str | None = None,
    ) -> None:
        """Add a column to the table."""
        ...

    def add_row(self, *renderables: str) -> None:
        """Add a row to the table."""
        ...


class RichProgressProtocol(Protocol):
    """Protocol for rich.progress.Progress context manager."""

    def __enter__(self) -> RichProgressProtocol:
        """Enter context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context."""
        ...

    def add_task(self, description: str, total: float | None = None) -> int:
        """Add a task to track."""
        ...

    def update(self, task_id: int, description: str | None = None) -> None:
        """Update task description."""
        ...

    def advance(self, task_id: int, advance: float = 1.0) -> None:
        """Advance task progress."""
        ...


class _RichHandlerFactory(Protocol):
    """Protocol for RichHandler constructor."""

    def __call__(
        self,
        *,
        show_time: bool = True,
        show_path: bool = False,
        rich_tracebacks: bool = True,
    ) -> logging.Handler:
        """Create RichHandler instance."""
        ...


class _RichTableFactory(Protocol):
    """Protocol for Table constructor."""

    def __call__(
        self,
        *,
        title: str | None = None,
        show_header: bool = True,
    ) -> RichTableProtocol:
        """Create Table instance."""
        ...


class _RichPanelFactory(Protocol):
    """Protocol for Panel constructor."""

    def __call__(self, content: str, *, title: str | None = None) -> str:
        """Create Panel instance."""
        ...


class _RichProgressColumn(Protocol):
    """Protocol for progress column objects."""

    ...


class _RichProgressColumnFactory(Protocol):
    """Protocol for progress column constructors."""

    def __call__(self) -> _RichProgressColumn:
        """Create column instance."""
        ...


class _RichTextColumnFactory(Protocol):
    """Protocol for TextColumn constructor."""

    def __call__(self, text: str) -> _RichProgressColumn:
        """Create TextColumn instance."""
        ...


class _RichProgressFactory(Protocol):
    """Protocol for Progress constructor."""

    def __call__(
        self,
        *columns: _RichProgressColumn,
        console: RichConsoleProtocol,
    ) -> RichProgressProtocol:
        """Create Progress instance."""
        ...


# Module-level console reference for rich output
_rich_console: RichConsoleProtocol | None = None


class _RichHandlerResult(TypedDict):
    """Result from creating a RichHandler."""

    handler: logging.Handler
    console: RichConsoleProtocol


def _create_rich_handler(
    *,
    show_time: bool,
    show_path: bool,
    rich_tracebacks: bool,
) -> _RichHandlerResult:
    """Create RichHandler and extract its console with proper typing."""
    rich_logging_mod = __import__("rich.logging", fromlist=["RichHandler"])
    rich_handler_cls: _RichHandlerFactory = rich_logging_mod.RichHandler

    handler = rich_handler_cls(
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=rich_tracebacks,
    )

    # Extract console at creation time with Protocol typing
    # RichHandler creates its own Console in __init__ and stores it as instance attribute
    # Access via __dict__ with typed dict to get the console with proper Protocol type
    handler_dict: dict[str, RichConsoleProtocol] = handler.__dict__
    console: RichConsoleProtocol = handler_dict["console"]

    return {"handler": handler, "console": console}


def _load_rich_table_factory() -> _RichTableFactory:
    """Load Table factory from rich.table."""
    rich_table_mod = __import__("rich.table", fromlist=["Table"])
    factory: _RichTableFactory = rich_table_mod.Table
    return factory


def _load_rich_panel_factory() -> _RichPanelFactory:
    """Load Panel factory from rich.panel."""
    rich_panel_mod = __import__("rich.panel", fromlist=["Panel"])
    factory: _RichPanelFactory = rich_panel_mod.Panel
    return factory


class _RichProgressComponents:
    """Container for rich.progress component factories."""

    progress: _RichProgressFactory
    spinner: _RichProgressColumnFactory
    text: _RichTextColumnFactory
    bar: _RichProgressColumnFactory
    task_pct: _RichProgressColumnFactory


def _load_rich_progress_components() -> _RichProgressComponents:
    """Load Progress component factories from rich.progress."""
    rich_progress_mod = __import__(
        "rich.progress",
        fromlist=["Progress", "SpinnerColumn", "TextColumn", "BarColumn", "TaskProgressColumn"],
    )
    components = _RichProgressComponents()
    components.progress = rich_progress_mod.Progress
    components.spinner = rich_progress_mod.SpinnerColumn
    components.text = rich_progress_mod.TextColumn
    components.bar = rich_progress_mod.BarColumn
    components.task_pct = rich_progress_mod.TaskProgressColumn
    return components


def setup_rich_logging(
    *,
    level: LogLevel = "INFO",
    show_time: bool = True,
    show_path: bool = False,
) -> logging.Logger:
    """Setup logging with rich.logging.RichHandler for CLI scripts.

    Configures the root logger with RichHandler for beautiful console output.
    The handler's console can be accessed via get_rich_console() for rendering
    Tables, Panels, and Progress bars.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        show_time: Whether to show timestamps in log output
        show_path: Whether to show file path in log output

    Returns:
        Configured root logger

    Example:
        >>> from platform_core.logging import setup_rich_logging, get_rich_console
        >>> logger = setup_rich_logging(level="INFO")
        >>> logger.info("Starting optimization...")
        >>> console = get_rich_console()
        >>> # Use console for Tables, Panels, Progress
    """
    global _rich_console

    log_level = _level_to_int(level)

    # Configure root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    # Create RichHandler and extract console with proper typing
    result = _create_rich_handler(
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=True,
    )
    handler = result["handler"]
    handler.setLevel(logging.DEBUG)

    # Add to root logger
    root.addHandler(handler)

    # Store console reference for later access
    _rich_console = result["console"]

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    return root


def get_rich_console() -> RichConsoleProtocol:
    """Get the rich Console for rendering Tables, Panels, Progress bars.

    Must be called after setup_rich_logging(). Returns the console instance
    that can be used for rendering rich output.

    Returns:
        Rich Console instance

    Raises:
        RuntimeError: If setup_rich_logging() was not called first

    Example:
        >>> from platform_core.logging import setup_rich_logging, get_rich_console
        >>> setup_rich_logging()
        >>> console = get_rich_console()
        >>> table = create_rich_table(title="Results")
        >>> console.print(table)
    """
    if _rich_console is None:
        msg = "setup_rich_logging() must be called before get_rich_console()"
        raise RuntimeError(msg)

    return _rich_console


def create_rich_table(
    title: str | None = None,
    show_header: bool = True,
) -> RichTableProtocol:
    """Create a rich Table for console output.

    Args:
        title: Optional table title
        show_header: Whether to show column headers

    Returns:
        Rich Table instance
    """
    table_factory = _load_rich_table_factory()
    return table_factory(title=title, show_header=show_header)


def create_rich_panel(content: str, title: str | None = None) -> str:
    """Create a rich Panel for console output.

    Args:
        content: Panel content (can include rich markup)
        title: Optional panel title

    Returns:
        Rich Panel instance (typed as str for Protocol compatibility)
    """
    panel_factory = _load_rich_panel_factory()
    return panel_factory(content, title=title)


def create_rich_progress(console: RichConsoleProtocol) -> RichProgressProtocol:
    """Create a rich Progress bar for console output.

    Args:
        console: Rich Console instance from get_rich_console()

    Returns:
        Rich Progress context manager
    """
    components = _load_rich_progress_components()
    return components.progress(
        components.spinner(),
        components.text("[progress.description]{task.description}"),
        components.bar(),
        components.task_pct(),
        console=console,
    )


def create_rich_spinner_progress(console: RichConsoleProtocol) -> RichProgressProtocol:
    """Create a simple rich Progress with just spinner and text.

    Args:
        console: Rich Console instance from get_rich_console()

    Returns:
        Rich Progress context manager with spinner only
    """
    components = _load_rich_progress_components()
    return components.progress(
        components.spinner(),
        components.text("[progress.description]{task.description}"),
        console=console,
    )


__all__ = [
    "JsonFormatter",
    "LogEventFields",
    "LogFormat",
    "LogLevel",
    "LogRecordQueueProtocol",
    "QueueHandlerFactory",
    "QueueListenerFactory",
    "QueueListenerProtocol",
    "RichConsoleProtocol",
    "RichProgressProtocol",
    "RichRenderableProtocol",
    "RichTableProtocol",
    "TextFormatter",
    "create_rich_panel",
    "create_rich_progress",
    "create_rich_spinner_progress",
    "create_rich_table",
    "get_logger",
    "get_rich_console",
    "load_queue_handler_factory",
    "load_queue_listener_factory",
    "setup_logging",
    "setup_rich_logging",
    "stdlib_logging",
]
