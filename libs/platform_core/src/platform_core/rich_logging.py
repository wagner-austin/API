"""Rich console logging for CLI scripts.

The RichHandler setup, the console accessor, and the Table / Panel /
Progress helpers, typed through Protocols so ``rich`` stays an optional
runtime import. The core structured logging (formatters,
``setup_logging``, ``get_logger``) lives in
:mod:`platform_core.logging`.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Protocol, TypedDict

from platform_core.logging import LogLevel, _level_to_int


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
        >>> from platform_core.rich_logging import setup_rich_logging, get_rich_console
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
        >>> from platform_core.rich_logging import setup_rich_logging, get_rich_console
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
    "RichConsoleProtocol",
    "RichProgressProtocol",
    "RichRenderableProtocol",
    "RichTableProtocol",
    "create_rich_panel",
    "create_rich_progress",
    "create_rich_spinner_progress",
    "create_rich_table",
    "get_rich_console",
    "setup_rich_logging",
]
