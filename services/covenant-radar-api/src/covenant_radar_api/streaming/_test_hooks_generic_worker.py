"""Test hooks for GenericStreamingWorker.

Provides hookable time/UUID/timestamp functions and text generation protocol
for the generic streaming worker. Production code uses real implementations;
tests override with fakes.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

# =============================================================================
# Text Generator Protocol
# =============================================================================


class TextGeneratorProtocol(Protocol):
    """Protocol for generating alert text from prompts.

    The generic streaming worker uses this to generate human-readable
    alert summaries. GeminiClient.generate_text() satisfies this
    structurally.
    """

    def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Text prompt describing the alert context.

        Returns:
            Generated text response.
        """
        ...


# =============================================================================
# Real Implementations
# =============================================================================


def _real_perf_counter() -> float:
    """Return performance counter value.

    Returns:
        Current performance counter in seconds.
    """
    return time.perf_counter()


def _real_generate_uuid() -> str:
    """Generate a random UUID string.

    Returns:
        UUID4 string in standard hyphenated format.
    """
    return str(uuid.uuid4())


def _real_current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO format.

    Returns:
        ISO 8601 timestamp with Z suffix (e.g., "2026-03-01T12:00:00Z").
    """
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Module-Level Injectable Hooks
# =============================================================================

# Production code calls these; tests override before calling.
perf_counter: Callable[[], float] = _real_perf_counter
generate_uuid: Callable[[], str] = _real_generate_uuid
current_iso_timestamp: Callable[[], str] = _real_current_iso_timestamp


# =============================================================================
# Fake Text Generator
# =============================================================================


class FakeTextGenerator:
    """Fake text generator for testing.

    Records all calls and returns a configurable response string.
    """

    def __init__(self) -> None:
        """Initialize with empty call history and default response."""
        self.calls: list[str] = []
        self.next_response: str = "Fake alert summary"

    def generate_text(self, prompt: str) -> str:
        """Record call and return configured response.

        Args:
            prompt: Text prompt (recorded for verification).

        Returns:
            The configured next_response string.
        """
        self.calls.append(prompt)
        return self.next_response


# =============================================================================
# Hook Management
# =============================================================================


def use_real_hooks() -> None:
    """Restore all hooks to real production implementations.

    Call this in test teardown to prevent test pollution.
    """
    global perf_counter, generate_uuid, current_iso_timestamp
    perf_counter = _real_perf_counter
    generate_uuid = _real_generate_uuid
    current_iso_timestamp = _real_current_iso_timestamp


__all__ = [
    "FakeTextGenerator",
    "TextGeneratorProtocol",
    "current_iso_timestamp",
    "generate_uuid",
    "perf_counter",
    "use_real_hooks",
]
