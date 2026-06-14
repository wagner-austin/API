"""No-op ``KeyboardProtocol`` stub for tests that do not assert keyboard input.

Real production code dispatches synthetic keyboard events through CDP, so
the ``page.keyboard`` slot is never the test surface. This stub exists
only to satisfy the structural type check; tests that need to assert
keyboard behavior should use a richer substitute.
"""

from __future__ import annotations


class NoOpKeyboard:
    """No-op ``KeyboardProtocol`` stub.

    Press/type calls are intentionally discarded -- production keyboard
    paths are exercised by CDP injection tests, not by direct ``page.keyboard``
    calls.
    """

    def press(self, key: str, *, delay: float | None = None) -> None:
        """Press a keyboard key (no-op)."""
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        """Type text (no-op)."""
        _ = (text, delay)


__all__ = ["NoOpKeyboard"]
