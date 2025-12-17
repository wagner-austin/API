"""State management for optimization script lifecycle."""

from __future__ import annotations

import signal
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import FrameType

from platform_core.logging import create_rich_panel, get_rich_console


class OptimizationState:
    """Manages optimization script lifecycle state.

    Handles proper startup and shutdown, including signal handling
    for graceful interruption.
    """

    def __init__(self) -> None:
        """Initialize state manager."""
        self._interrupted = False
        self._original_sigint: Callable[[int, FrameType | None], None] | int | None = None
        self._shutdown_callbacks: list[Callable[[], None]] = []

    @property
    def interrupted(self) -> bool:
        """Check if optimization was interrupted."""
        return self._interrupted

    def set_interrupted(self) -> None:
        """Mark the state as interrupted."""
        self._interrupted = True

    def register_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to run on shutdown.

        Args:
            callback (Callable[[], None]): Function to call during shutdown.
        """
        self._shutdown_callbacks.append(callback)

    def _handle_sigint(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        self._interrupted = True

    def start(self) -> None:
        """Start state management and install signal handlers."""
        self._interrupted = False
        # Install custom SIGINT handler
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_sigint)

    def stop(self) -> None:
        """Stop state management and cleanup.

        Restores original signal handlers and runs shutdown callbacks.
        """
        # Restore original handler
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None

        # Run shutdown callbacks in reverse order
        for callback in reversed(self._shutdown_callbacks):
            callback()
        self._shutdown_callbacks.clear()

    def print_interrupted_message(self) -> None:
        """Print interruption message to console."""
        console = get_rich_console()
        console.print()
        console.print(create_rich_panel("[bold red]Process Interrupted by User[/bold red]"))


# Module-level singleton for simple access
_state: OptimizationState | None = None


def get_state() -> OptimizationState:
    """Get the global optimization state instance.

    Returns:
        OptimizationState: The singleton state instance.
    """
    global _state
    if _state is None:
        _state = OptimizationState()
    return _state


@contextmanager
def managed_execution() -> Generator[OptimizationState, None, None]:
    """Context manager for managed script execution.

    Handles proper startup and shutdown with signal handling.
    KeyboardInterrupt is caught, marked, and re-raised for the caller to handle.

    Yields:
        OptimizationState: The state instance for checking interruption status.

    Raises:
        KeyboardInterrupt: Re-raised after marking state as interrupted.
    """
    state = get_state()
    state.start()
    try:
        yield state
    except KeyboardInterrupt:
        state.set_interrupted()
        raise
    finally:
        state.stop()


def is_interrupted() -> bool:
    """Check if the current execution has been interrupted.

    Returns:
        bool: True if interrupted, False otherwise.
    """
    state = get_state()
    return state.interrupted
