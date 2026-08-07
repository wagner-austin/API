"""Per-tick runtime context carried onto every emitted event.

The tick number, bot state, and in-flight action kind that
:mod:`tankpit_bot.runtime_logging` merges into each record so a log
line can be read without its surrounding lines.
"""

from __future__ import annotations

from typing_extensions import TypedDict

# Internal storage for the active context, split into one typed slot per
# field so each value is mypy-narrowed at its source. The public
# :class:`RuntimeContextDict` view is assembled by :func:`get_runtime_context`.
RUNTIME_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "tick_n",
        "bot_state",
        "in_flight_action_kind",
    }
)
"""Every field :func:`set_runtime_context` can attach to an event."""


_RUNTIME_CONTEXT_TICK_N: int | None = None

_RUNTIME_CONTEXT_BOT_STATE: str | None = None

_RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND: str | None = None


class RuntimeContextDict(TypedDict, total=False):
    """Per-tick context auto-attached to every emit_* event.

    Each field is optional; absent fields are omitted from the JSONL
    record. The tick loop sets these once per tick so every event
    emitted that tick carries the same context. Explicit fields passed
    to an emit_* call override the context (last-write-wins).

    Attributes:
        tick_n: 1-based index of the currently-executing tick. Use 0
            when no tick is active (boot, login, shutdown). Always
            attached when set, even if the value is 0.
        bot_state: ``"<mode>/<mode_state>"`` snapshot of the durable
            AI mode and its inner state. Empty string when none.
        in_flight_action_kind: ``ActionKind`` literal of the bot's
            current in-flight action, or ``"none"`` when idle.
    """

    tick_n: int
    bot_state: str
    in_flight_action_kind: str


def set_runtime_context(
    *,
    tick_n: int | None = None,
    bot_state: str | None = None,
    in_flight_action_kind: str | None = None,
) -> None:
    """Set or update the active per-tick runtime context.

    Each subsequent ``emit_*`` call attaches the present context fields
    to its structured payload (under the field names ``tick_n``,
    ``bot_state``, ``in_flight_action_kind``). Pass ``None`` to leave a
    previous value unchanged; use :func:`clear_runtime_context` to
    remove every value.

    Args:
        tick_n: 1-based current tick index, or ``None`` to keep the
            previous value.
        bot_state: ``"<mode>/<mode_state>"`` snapshot, or ``None`` to
            keep the previous value.
        in_flight_action_kind: ``ActionKind`` string, or ``None`` to
            keep the previous value.
    """
    global _RUNTIME_CONTEXT_TICK_N
    global _RUNTIME_CONTEXT_BOT_STATE
    global _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND
    if tick_n is not None:
        _RUNTIME_CONTEXT_TICK_N = tick_n
    if bot_state is not None:
        _RUNTIME_CONTEXT_BOT_STATE = bot_state
    if in_flight_action_kind is not None:
        _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND = in_flight_action_kind


def clear_runtime_context() -> None:
    """Remove every field from the active runtime context.

    Subsequent ``emit_*`` calls emit without context until
    :func:`set_runtime_context` is called again. The tick loop's
    teardown path calls this so test/probe sessions start clean.
    """
    global _RUNTIME_CONTEXT_TICK_N
    global _RUNTIME_CONTEXT_BOT_STATE
    global _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND
    _RUNTIME_CONTEXT_TICK_N = None
    _RUNTIME_CONTEXT_BOT_STATE = None
    _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND = None


def get_runtime_context() -> RuntimeContextDict:
    """Return a typed defensive copy of the current runtime context.

    Returns:
        A typed snapshot of the active context. Callers may mutate the
        returned dict without affecting the module-level state.
    """
    snapshot: RuntimeContextDict = {}
    if _RUNTIME_CONTEXT_TICK_N is not None:
        snapshot["tick_n"] = _RUNTIME_CONTEXT_TICK_N
    if _RUNTIME_CONTEXT_BOT_STATE is not None:
        snapshot["bot_state"] = _RUNTIME_CONTEXT_BOT_STATE
    if _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND is not None:
        snapshot["in_flight_action_kind"] = _RUNTIME_CONTEXT_IN_FLIGHT_ACTION_KIND
    return snapshot


__all__ = [
    "RuntimeContextDict",
    "clear_runtime_context",
    "get_runtime_context",
    "set_runtime_context",
]
