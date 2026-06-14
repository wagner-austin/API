"""Time-aware replay regression for the ghost-tank wire-presence gate.

The capture ``ghost_map_refresh_wire_silent`` is a trimmed slice of live
run ``bot-20260613-104415`` (account Artax, Practice room). In it the
purple-9 enemy tank ``517`` is genuinely in view early -- it emits
per-tank wire traffic (one tank-info, two movement responses) ending at
the slice's "last wire" instant -- then leaves. The server map blob keeps
re-listing it at its stale cached tile ``(34, 96)`` for the rest of the
slice, so ``timestamp_ms`` stays fresh while no wire source vouches for
it again.

The pre-fix bot gated the kill shot on ``timestamp_ms`` and therefore
fired at this afterimage, hitting empty ground. The fix gates on
``last_wire_seen_ms``, which the map path deliberately never advances.
This test replays the real bytes with a per-message clock (the standard
``replay_session`` collapses timing to one wall-clock instant and so
cannot exercise a time-based gate) and asserts the divergence that makes
the gate work: ``517`` is acquisition-fresh yet wire-silent at the final
instant, so the kill gate rejects it while acquisition still keeps it.
"""

from __future__ import annotations

from tankpit_bot import browser
from tankpit_bot.bot.ai.threats import (
    WIRE_PRESENCE_TTL_MS,
    analyze_threats,
    is_wire_present,
)
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.trackers import init_trackers_with_magic
from tankpit_bot.sniffer.viewport import reset_viewport_tracking
from tankpit_bot.sniffer.world_state import (
    get_world_state,
    reset_world_state,
)
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state
from tankpit_bot.types import CaptureSession
from tests.replay.fixture_loader import load_capture_fixture

_GHOST_FIXTURE = "ghost_map_refresh_wire_silent.capture_session.json"
_GHOST_TANK_ID = 517


class _Clock:
    """A settable millisecond clock for time-aware replay.

    The tank-update decoders read wall-clock time through the injectable
    ``browser.get_current_time_ms`` symbol. Rebinding it to this clock
    advances time to each message's real capture timestamp instead of the
    wall clock, which is what the wire-presence TTL is measured against.
    """

    def __init__(self) -> None:
        """Initialise the clock at zero."""
        self.now_ms = 0

    def __call__(self) -> int:
        """Return the current replay time in milliseconds.

        Returns:
            The most recently set message timestamp.
        """
        return self.now_ms


def _time_aware_replay(session: CaptureSession) -> int:
    """Replay received messages with a per-message clock.

    Resets the decoder globals, builds the XOR table and trackers from the
    session magic, then feeds received messages in timestamp order while
    advancing the tank-update clock to each message's capture time.

    Args:
        session: Loaded capture session to replay.

    Returns:
        The final (largest) received-message timestamp, i.e. the instant
        the gate is evaluated at.

    Raises:
        ValueError: If the session has no magic key to XOR-decode with.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("fixture must carry a magic key")

    clock = _Clock()
    original_clock = browser.get_current_time_ms
    browser.get_current_time_ms = clock
    try:
        reset_world_state()
        reset_xor_state()
        reset_viewport_tracking()
        build_global_xor_table(magic)
        init_trackers_with_magic(magic)

        final_ms = 0
        for timestamp_ms, payload in _received_in_order(session):
            clock.now_ms = timestamp_ms
            final_ms = timestamp_ms
            process_received_message(payload)
        return final_ms
    finally:
        browser.get_current_time_ms = original_clock


def _received_in_order(session: CaptureSession) -> list[tuple[int, str]]:
    """Return (timestamp_ms, payload) for received messages in time order.

    Args:
        session: Loaded capture session.

    Returns:
        Each received message's timestamp and payload, ascending by time.
    """
    received = [
        (message["timestamp_ms"], message["payload"])
        for message in session["messages"]
        if message["direction"] == "received"
    ]
    received.sort(key=_timestamp_of)
    return received


def _timestamp_of(pair: tuple[int, str]) -> int:
    """Return the timestamp from a (timestamp_ms, payload) pair.

    Args:
        pair: A (timestamp_ms, payload) tuple.

    Returns:
        The ``timestamp_ms`` element for sorting.
    """
    return pair[0]


def test_map_refresh_keeps_ghost_acquirable_but_wire_silent() -> None:
    """The departed ghost stays acquisition-fresh yet fails the kill gate.

    This is the exact divergence the fix introduces: the map blob keeps
    ``timestamp_ms`` within the freshness window (so the bot may still
    teleport toward the tank) while ``last_wire_seen_ms`` is frozen far in
    the past (so the bot must not fire at it).
    """
    session = load_capture_fixture(_GHOST_FIXTURE)

    final_ms = _time_aware_replay(session)

    tank = get_world_state()["tanks"][str(_GHOST_TANK_ID)]
    # The map kept the position alive: acquisition-fresh at the final instant.
    assert final_ms - tank["timestamp_ms"] <= WIRE_PRESENCE_TTL_MS
    # The tank was genuinely wire-confirmed once (the stamp is non-zero)...
    assert tank["last_wire_seen_ms"] > 0
    # ...but went wire-silent long before the end (well past the TTL).
    assert final_ms - tank["last_wire_seen_ms"] > WIRE_PRESENCE_TTL_MS
    # Therefore the kill gate rejects it.
    assert is_wire_present(tank["last_wire_seen_ms"], final_ms) is False


def test_ghost_is_still_an_acquisition_threat() -> None:
    """The wire-silent ghost still appears as an acquirable threat.

    Acquisition must keep finding the tank (the fix only forbids the
    shot, not the teleport-toward), and the threat must carry the frozen
    wire stamp so the kill gate can reject it downstream.
    """
    session = load_capture_fixture(_GHOST_FIXTURE)

    final_ms = _time_aware_replay(session)

    world = get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        raise AssertionError("replay must establish the bot's own self_state")
    threats = analyze_threats(world, self_state, final_ms)
    ghost_threats = [t for t in threats if t["tank_id"] == _GHOST_TANK_ID]
    assert len(ghost_threats) == 1
    assert is_wire_present(ghost_threats[0]["last_wire_seen_ms"], final_ms) is False


def test_ghost_fixture_is_present() -> None:
    """The trimmed real-capture ghost fixture is checked in."""
    session = load_capture_fixture(_GHOST_FIXTURE)
    assert session["session_id"] == "ghost-map-refresh-wire-silent"
    assert any(message["direction"] == "received" for message in session["messages"])
