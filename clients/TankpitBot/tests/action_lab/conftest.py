"""Shared fixtures for action_lab tests.

Centralizes the "stop mocking the inventory tracker" pattern: tests should
mutate the real ``_ws._inventory_state`` via the real ``update_inventory_*``
codepaths (or replay real captured frames through ``process_received_message``)
instead of patching ``get_inventory_state`` at one of the many module-level
import bindings.

The previous mock-everywhere pattern is the root cause of the hang in
``test_run_pickup_attempt_for_probe_completes_immediately_when_inventory_grew``
that originally surfaced this work — ``ops_module.get_inventory_state`` was
patched but the inner waiter in ``equipment_pickup.py`` reads its own
unpatched import, so the wait loop never saw the mock value and spun until
the test was killed.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.world_state_inventory import (
    update_inventory_from_protocol,
)
from tankpit_bot.sniffer.xor import build_global_xor_table
from tankpit_bot.types import CapturedMessage, decode_capture_session

REPO_ROOT = Path(__file__).resolve().parents[2]

FUEL_PROBE_CAPTURE_PATH = REPO_ROOT / "fuel_probe.capture_session.json"

INVENTORY_GROWTH_FRAME_INDEX = 51
"""First received frame in fuel_probe.capture_session.json whose decode
causes the real inventory tracker total to grow (0 -> 112 via a 0x49 sync).
Discovered by replaying the capture through process_received_message and
watching get_inventory_state()."""

INVENTORY_TOTAL_AFTER_GROWTH = 112


class FailIfWaitedPage:
    """Page that fails the test if wait_for_timeout is ever called.

    Guards against regression of the hang where a fast path quietly slipped
    into the wait loop. If wait_for_timeout is called, the fast path did not
    trigger and the test fails fast instead of hanging.
    """

    def wait_for_timeout(self, timeout: float) -> None:
        raise AssertionError(
            f"wait_for_timeout({timeout}) called — fast path did not trigger, test would have hung"
        )


@pytest.fixture()
def replay_pipeline() -> Generator[list[CapturedMessage], None, None]:
    """Reset world state, rebuild XOR table from real captured session magic.

    Yields the typed captured-message list so tests can replay specific frames
    through the real decoder pipeline via ``process_received_message``.
    Restores world state after.
    """
    session_text = core_hooks.read_text(FUEL_PROBE_CAPTURE_PATH)
    session = decode_capture_session(narrow_json_to_dict(load_json_str(session_text)))
    magic = session["magic"]
    if magic is None:
        raise RuntimeError(
            f"Capture {FUEL_PROBE_CAPTURE_PATH.name} has no magic key — cannot replay binary frames"
        )
    reset_world_state()
    build_global_xor_table(magic)
    yield session["messages"]
    reset_world_state()


@pytest.fixture()
def real_inventory() -> Generator[None, None, None]:
    """Reset world state before and after the test.

    Tests use ``update_inventory_from_protocol([counts], [enabled])`` directly
    to set a known inventory baseline via the real codepath, so every module
    binding of ``get_inventory_state`` returns the same real value.
    """
    reset_world_state()
    yield
    reset_world_state()


@pytest.fixture(autouse=True)
def restore_action_hooks() -> Generator[None, None, None]:
    """Restore canonical action_lab hooks after each test.

    Tests that override ``action_hooks.drain_buffered_messages``,
    ``action_hooks.get_current_time_ms``, ``action_hooks.check_and_clear_*``
    don't have to remember to restore — this autouse fixture does it.
    """
    from tankpit_bot.bot.world_sync import drain_messages as _real_drain
    from tankpit_bot.browser import get_current_time_ms as _real_clock
    from tankpit_bot.sniffer.world_state import (
        check_and_clear_radar_scan_complete as _real_clear_radar,
    )
    from tankpit_bot.sniffer.world_state_combat import (
        check_and_clear_teleport_landed as _real_clear_teleport,
    )

    yield

    action_hooks.drain_buffered_messages = _real_drain
    action_hooks.get_current_time_ms = _real_clock
    action_hooks.check_and_clear_radar_scan_complete = _real_clear_radar
    action_hooks.check_and_clear_teleport_landed = _real_clear_teleport


class Terrain:
    """Terrain fake mirroring the real ``TerrainMapProtocol`` semantics.

    Real terrain (see ``src/tankpit_bot/terrain.py``) has three tile types:
    ``ROCK``/``GROUND``/``WATER``, and ``is_passable`` is exactly
    ``get_terrain(x, y) == GROUND``. Tests express terrain by stating the
    default tile type and a sparse override map of obstacles. Matches the
    real semantics so reachability logic exercised against this fake
    behaves identically to a live session.
    """

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def __init__(
        self,
        *,
        default: str = GROUND,
        overrides: dict[tuple[int, int], str] | None = None,
    ) -> None:
        self._default = default
        self._overrides = overrides or {}

    def get_terrain(self, x: int, y: int) -> str:
        return self._overrides.get((x, y), self._default)

    def is_passable(self, x: int, y: int) -> bool:
        return self.get_terrain(x, y) == self.GROUND

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        rows: list[list[str]] = []
        left = center_x - (width // 2)
        top = center_y - (height // 2)
        for y in range(top, top + height):
            row: list[str] = []
            for x in range(left, left + width):
                row.append(self.get_terrain(x, y))
            rows.append(row)
        return rows


def ground_terrain(
    obstacles: dict[tuple[int, int], str] | None = None,
) -> TerrainMapProtocol:
    """Default-GROUND terrain with optional sparse obstacles.

    Use for chain tests where most of the map is walkable and only a few
    rocks or water tiles block specific paths.
    """
    return Terrain(default=Terrain.GROUND, overrides=obstacles)


def rock_wall(x: int, y_range: range) -> dict[tuple[int, int], str]:
    """Build a vertical rock wall — one column, full y range.

    Used to force ``requires_reposition=True`` in chain tests: a wall
    spanning the full viewport height blocks every BFS detour inside
    viewport bounds, so the real reachability check returns False.
    """
    return {(x, y): Terrain.ROCK for y in y_range}


def set_inventory_total(total: int) -> None:
    """Set the real inventory tracker so its total equals ``total``.

    Uses the real ``update_inventory_from_protocol`` codepath, so every
    module-level binding of ``get_inventory_state`` sees the same value.
    All counts are loaded into ``dual_shots`` for simplicity; callers that
    need a specific distribution should call ``update_inventory_from_protocol``
    directly.
    """
    update_inventory_from_protocol(
        counts=[0, total, 0, 0, 0],
        enabled=[True, True, True, True, True],
    )
