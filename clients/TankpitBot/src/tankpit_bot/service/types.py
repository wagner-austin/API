"""TypedDicts and small translators for the bot service wire surface.

Every dict here crosses either the HTTP boundary (client SPA ↔ bot) or
the cross-thread boundary (aiohttp server ↔ sync tick loop). They are
paired with encode/decode in :mod:`tankpit_bot.service.types_codecs`.

The wire vocabulary adds one value on top of :data:`AIMode`: ``"AUTO"``.
On the wire, ``"AUTO"`` means "restore the durable HFSM auto-arbitrator";
internally it maps to ``manual_mode = None`` in
:class:`tankpit_bot.bot.ai.types.AIStateDict`. The three ``AIMode``
literals (``"UNSET"``, ``"HUNT"``, ``"COLLECT"``) pass through
unchanged and pin the arbitrator to that mode.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from tankpit_bot.bot.ai.modes import AIMode, AIModeState

WireMode = Literal["UNSET", "HUNT", "COLLECT", "AUTO"]

WIRE_MODES: tuple[WireMode, ...] = (
    "UNSET",
    "HUNT",
    "COLLECT",
    "AUTO",
)


class ModeCommandDict(TypedDict):
    """Wire payload for ``POST /api/tankbot/mode``.

    Attributes:
        manual_mode: The mode the SPA is asking the bot to hold. Use
            ``"AUTO"`` to restore auto-arbitration; the three
            :data:`AIMode` literals pin the durable HFSM to that mode.
    """

    manual_mode: WireMode


def make_mode_command(manual_mode: WireMode) -> ModeCommandDict:
    """Create a :class:`ModeCommandDict`.

    Args:
        manual_mode: Wire-level mode literal.

    Returns:
        Populated :class:`ModeCommandDict`.
    """
    return ModeCommandDict(manual_mode=manual_mode)


class LiveStatsDict(TypedDict):
    """Per-session live counters shown in the SPA stats panel.

    Attributes:
        kills: Distinct victim tanks the bot has confirmed dead this
            session. Sourced from ``session_kill_count`` in
            :class:`tankpit_bot.bot.ai.types.AIStateDict`.
        hits: Bot shots that resolved on a live enemy this session.
            Sourced from ``session_hit_count``.
        misses: Bot shots that resolved without a hit this session.
            Sourced from ``session_miss_count``.
        radars_used: Radar-scan commands the executor has dispatched
            this session. Sourced from ``live_radars_used``.
        teleports: Teleport commands the executor has dispatched this
            session. Sourced from ``live_teleports``.
    """

    kills: int
    hits: int
    misses: int
    radars_used: int
    teleports: int


def make_live_stats(
    kills: int,
    hits: int,
    misses: int,
    radars_used: int,
    teleports: int,
) -> LiveStatsDict:
    """Create a :class:`LiveStatsDict`.

    Args:
        kills: Distinct confirmed kills this session.
        hits: Successful shots this session.
        misses: Missed shots this session.
        radars_used: Radar dispatches this session.
        teleports: Teleport dispatches this session.

    Returns:
        Populated :class:`LiveStatsDict`.
    """
    return LiveStatsDict(
        kills=kills,
        hits=hits,
        misses=misses,
        radars_used=radars_used,
        teleports=teleports,
    )


def zero_live_stats() -> LiveStatsDict:
    """Return a :class:`LiveStatsDict` with every counter at zero.

    Returns:
        :class:`LiveStatsDict` initialised to zero.
    """
    return LiveStatsDict(
        kills=0,
        hits=0,
        misses=0,
        radars_used=0,
        teleports=0,
    )


class SessionStatusDict(TypedDict):
    """Snapshot of the bot service pushed to SPA subscribers over SSE.

    Emitted from the sync tick loop each tick and consumed by the SPA
    status subscriber to paint the button state + stats panel.

    Attributes:
        running: True while a session is active (post-``wait_for_game_ready``,
            pre-teardown). False before the first start and between
            sessions.
        manual_mode: The SPA-selected mode literal. ``"AUTO"`` when
            auto-arbitration is restored.
        active_mode: The durable HFSM mode the arbitrator resolved to
            this tick. Equals ``manual_mode`` when the latter pins to
            one of the three :data:`AIMode` literals; may differ when
            ``manual_mode == "AUTO"``.
        active_mode_state: Durable substate within :attr:`active_mode`.
        session_started_ms: Wall-clock timestamp when the current
            session began, or ``0`` when ``running`` is False.
        tick_timestamp_ms: Wall-clock timestamp when this snapshot was
            captured. Consumers use it for staleness gating.
        stats: Live counters for the SPA stats panel.
    """

    running: bool
    manual_mode: WireMode
    active_mode: AIMode
    active_mode_state: AIModeState
    session_started_ms: int
    tick_timestamp_ms: int
    stats: LiveStatsDict


def make_session_status(
    running: bool,
    manual_mode: WireMode,
    active_mode: AIMode,
    active_mode_state: AIModeState,
    session_started_ms: int,
    tick_timestamp_ms: int,
    stats: LiveStatsDict,
) -> SessionStatusDict:
    """Create a :class:`SessionStatusDict`.

    Args:
        running: Whether a session is currently active.
        manual_mode: Wire-level mode literal selected by the SPA.
        active_mode: Durable HFSM mode resolved this tick.
        active_mode_state: Durable substate within ``active_mode``.
        session_started_ms: Session start wall-clock, or ``0`` when
            ``running`` is False.
        tick_timestamp_ms: Snapshot capture wall-clock.
        stats: Live counters.

    Returns:
        Populated :class:`SessionStatusDict`.
    """
    return SessionStatusDict(
        running=running,
        manual_mode=manual_mode,
        active_mode=active_mode,
        active_mode_state=active_mode_state,
        session_started_ms=session_started_ms,
        tick_timestamp_ms=tick_timestamp_ms,
        stats=stats,
    )


def idle_session_status(tick_timestamp_ms: int) -> SessionStatusDict:
    """Return a :class:`SessionStatusDict` representing an idle service.

    Used when the service is up but no session is running yet — the
    SPA still needs a status frame to paint "idle" buttons.

    Args:
        tick_timestamp_ms: Snapshot capture wall-clock.

    Returns:
        :class:`SessionStatusDict` with ``running=False`` and every
        counter zeroed.
    """
    return make_session_status(
        running=False,
        manual_mode="AUTO",
        active_mode="UNSET",
        active_mode_state="",
        session_started_ms=0,
        tick_timestamp_ms=tick_timestamp_ms,
        stats=zero_live_stats(),
    )


def wire_mode_to_manual(wire: WireMode) -> AIMode | None:
    """Translate a wire mode literal to an :data:`AIMode` override.

    ``"AUTO"`` means "restore auto-arbitration" and maps to ``None``;
    the three :data:`AIMode` literals pass through and pin the durable
    HFSM to that mode.

    Args:
        wire: The wire-level mode literal.

    Returns:
        The :data:`AIMode` value to pin the arbitrator to, or ``None``
        when auto-arbitration should run.
    """
    if wire == "AUTO":
        return None
    if wire == "UNSET":
        return "UNSET"
    if wire == "HUNT":
        return "HUNT"
    return "COLLECT"


def manual_to_wire_mode(manual: AIMode | None) -> WireMode:
    """Translate an :data:`AIMode` override to a wire mode literal.

    Inverse of :func:`wire_mode_to_manual`.

    Args:
        manual: The :data:`AIMode` the arbitrator is pinned to, or
            ``None`` when auto-arbitration is active.

    Returns:
        The wire-level mode literal equivalent.
    """
    if manual is None:
        return "AUTO"
    if manual == "UNSET":
        return "UNSET"
    if manual == "HUNT":
        return "HUNT"
    return "COLLECT"


__all__ = [
    "WIRE_MODES",
    "LiveStatsDict",
    "ModeCommandDict",
    "SessionStatusDict",
    "WireMode",
    "idle_session_status",
    "make_live_stats",
    "make_mode_command",
    "make_session_status",
    "manual_to_wire_mode",
    "wire_mode_to_manual",
    "zero_live_stats",
]
