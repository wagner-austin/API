"""Per-tick runtime context carried onto every emitted event.

The tick number, bot state, and in-flight action kind that
:mod:`tankpit_bot.runtime_logging` merges into each record so a log
line can be read without its surrounding lines.

Held in :class:`contextvars.ContextVar` slots rather than module
globals ([[session-state-deglobalisation]] step 10). This is the one
place in the refactor where threading a parameter would have been the
wrong answer: ``emit_ai`` and ``emit_diagnostic`` are called from 256
sites inside pure planner logic, and giving every scoring function a
logging argument to carry would cost far more than the globals did.
A context variable is ambient by design and still isolates per
thread and per async task, which a module global does not.
"""

from __future__ import annotations

from contextvars import ContextVar

from typing_extensions import TypedDict

# One typed slot per field so each value is mypy-narrowed at its source.
# The public :class:`RuntimeContextDict` view is assembled by
# :func:`get_runtime_context`.
RUNTIME_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "tick_n",
        "bot_state",
        "in_flight_action_kind",
    }
)
"""Every field :func:`set_runtime_context` can attach to an event."""


_TICK_N: ContextVar[int | None] = ContextVar("tankpit_tick_n", default=None)

_BOT_STATE: ContextVar[str | None] = ContextVar("tankpit_bot_state", default=None)

_IN_FLIGHT_ACTION_KIND: ContextVar[str | None] = ContextVar(
    "tankpit_in_flight_action_kind", default=None
)


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
    if tick_n is not None:
        _TICK_N.set(tick_n)
    if bot_state is not None:
        _BOT_STATE.set(bot_state)
    if in_flight_action_kind is not None:
        _IN_FLIGHT_ACTION_KIND.set(in_flight_action_kind)


def clear_runtime_context() -> None:
    """Remove every field from the active runtime context.

    Subsequent ``emit_*`` calls emit without context until
    :func:`set_runtime_context` is called again.

    This docstring used to claim "the tick loop's teardown path calls
    this". It does not, and never did — measured 2026-08-07: the only
    callers anywhere are ``tests/conftest.py`` and this function's own
    test. In production the context is set once per tick
    (``bot/tick_loop.py``) and never cleared, so the end-of-run
    scorecard carries the final tick's ``tick_n`` and ``bot_state``.
    That is defensible — the scorecard does belong to that tick — but
    it was not what the docstring said.
    """
    _TICK_N.set(None)
    _BOT_STATE.set(None)
    _IN_FLIGHT_ACTION_KIND.set(None)


def get_runtime_context() -> RuntimeContextDict:
    """Return a typed defensive copy of the current runtime context.

    Returns:
        A typed snapshot of the active context. Callers may mutate the
        returned dict without affecting the module-level state.
    """
    snapshot: RuntimeContextDict = {}
    tick_n = _TICK_N.get()
    if tick_n is not None:
        snapshot["tick_n"] = tick_n
    bot_state = _BOT_STATE.get()
    if bot_state is not None:
        snapshot["bot_state"] = bot_state
    in_flight_action_kind = _IN_FLIGHT_ACTION_KIND.get()
    if in_flight_action_kind is not None:
        snapshot["in_flight_action_kind"] = in_flight_action_kind
    return snapshot


__all__ = [
    "RuntimeContextDict",
    "clear_runtime_context",
    "get_runtime_context",
    "set_runtime_context",
]
