"""The Rb scope-extend law — decode, anchor shift, and server routing.

Wire-measured 2026-08-01 against the 2026-07-10 human capture
(sniff-20260710-202821): all 8 sent scope frames decode as
``[3,'Z',dir]`` with the compass CLOCKWISE FROM NORTH, and every
answering 0x5A fits the ANCHOR law exactly — the tank pins to the
window edge trailing the requested direction ([[viewport-shift-
protocol]]). The measured rows below are the capture's, verbatim.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol.commands import (
    CMD_SCOPE,
    SCOPE_CENTER,
    SCOPE_EAST,
    SCOPE_NORTH,
    SCOPE_NORTHEAST,
    SCOPE_SOUTHEAST,
    SCOPE_WEST,
    build_scope_command,
)
from tankpit_bot.protocol.helpers import DecodeError
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.viewport_window import ViewportTracker
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _scope_payload(direction: int) -> bytes:
    """The plaintext scope payload after the ``!`` prefix and XOR."""
    framed = build_scope_command(direction)
    # [len_lo, len_hi, '!', type, cmd, direction] -> strip length + '!'
    return framed[3:]


def test_scope_frame_decodes_kind_and_direction() -> None:
    """The 3-byte Rb frame carries the compass byte through."""
    decoded = decode_client_command(_scope_payload(SCOPE_SOUTHEAST))
    assert decoded["kind"] == "scope"
    assert decoded["command"] == CMD_SCOPE
    assert decoded["direction"] == SCOPE_SOUTHEAST


def test_truncated_scope_frame_raises() -> None:
    """A scope frame without its direction byte is a decode failure."""
    with pytest.raises(DecodeError):
        decode_client_command(bytes([0x23, CMD_SCOPE]))


def _tracker(tank_x: int, tank_y: int, window: tuple[int, int]) -> ViewportTracker:
    """A tracker for tank 9 at the given tile with a forced stored window."""
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, tank_x, tank_y, 1000)
    tracker = ViewportTracker(world, InMemoryTerrainMap(), client_id=9)
    tracker.window = window
    return tracker


# Every scope-extend the 2026-07-10 human session sent, paired with
# its walk-corrected self position and the served 0x5A windows —
# the anchor law fits all 8 with zero free parameters.
_MEASURED_SHIFTS: tuple[tuple[str, int, tuple[int, int], tuple[int, int], tuple[int, int]], ...] = (
    ("NE", SCOPE_NORTHEAST, (129, 172), (114, 163), (129, 157)),
    ("E", SCOPE_EAST, (144, 164), (129, 157), (144, 157)),
    ("E", SCOPE_EAST, (160, 168), (145, 159), (160, 159)),
    ("SE", SCOPE_SOUTHEAST, (170, 174), (160, 159), (170, 174)),
    ("W", SCOPE_WEST, (188, 233), (188, 221), (173, 221)),
    ("SE", SCOPE_SOUTHEAST, (188, 233), (173, 221), (188, 233)),
    ("E", SCOPE_EAST, (96, 207), (81, 201), (96, 201)),
    ("N", SCOPE_NORTH, (113, 34), (110, 33), (110, 19)),
)


def test_anchor_law_reproduces_every_measured_shift() -> None:
    """All 8 capture rows: shifted window == the served 0x5A origin."""
    for label, direction, (sx, sy), before, after in _MEASURED_SHIFTS:
        tracker = _tracker(sx, sy, before)
        tracker.apply_scope_shift(direction)
        assert tracker.window == after, f"Extend view {label} from {before} at ({sx},{sy})"


def test_scope_center_recenters_like_a_teleport_landing() -> None:
    """Direction 8 restores the rest-state window at (x-8, y-8)."""
    tracker = _tracker(100, 100, (100, 100))
    tracker.apply_scope_shift(SCOPE_CENTER)
    assert tracker.window == (92, 92)


def test_scope_shift_clamps_to_map_bounds() -> None:
    """A westward anchor from the map's edge column clamps at 0."""
    tracker = _tracker(4, 250, (0, 240))
    tracker.apply_scope_shift(SCOPE_WEST)
    assert tracker.window == (0, 240)


def test_unknown_direction_byte_shifts_nothing() -> None:
    """An out-of-table byte leaves the stored window untouched."""
    tracker = _tracker(100, 100, (92, 92))
    tracker.apply_scope_shift(11)
    assert tracker.window == (92, 92)


def _server() -> SimServer:
    """Client tank 9 at (100, 100), enemy 11 far outside the window."""
    world: SimWorldDict = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 100, 100, 1000)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 113, 100, 500)
    return SimServer(world, InMemoryTerrainMap(), client_id=9)


def test_client_scope_command_answers_with_the_shifted_0x5a() -> None:
    """Every Rb is answered by a 0x5A carrying the anchored origin.

    Measured: 8-for-8 in the capture, 50 ms-1.5 s lag — the confirm
    always comes, even when the patch has nothing to enumerate.
    """
    server = _server()
    server.queue_command(9, decode_client_command(_scope_payload(SCOPE_EAST)))
    batch = server.advance_tick()
    updates = [m for m in batch if m["msg_type"] == 0x5A]
    assert len(updates) == 1
    assert (updates[0]["viewport_left"], updates[0]["viewport_top"]) == (100, 92)


def test_scope_shift_reveals_tanks_entering_the_window() -> None:
    """The pan's membership diff announces newly visible tanks (0x3D)."""
    server = _server()
    server.queue_command(9, decode_client_command(_scope_payload(SCOPE_EAST)))
    batch = server.advance_tick()
    positions = [m for m in batch if m["msg_type"] == 0x3D and m.get("tank_id") == 11]
    assert len(positions) == 1


def test_non_client_scope_command_moves_no_window() -> None:
    """Another tank's scope press never touches the client's window."""
    server = _server()
    before = server._viewport.window
    server.queue_command(11, decode_client_command(_scope_payload(SCOPE_EAST)))
    batch = server.advance_tick()
    assert server._viewport.window == before
    assert [m for m in batch if m["msg_type"] == 0x5A] == []
