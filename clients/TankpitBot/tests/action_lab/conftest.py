"""Shared fixtures for action_lab tests.

Centralizes the "stop mocking the inventory tracker" pattern: tests should
mutate the real ``_get_world_service().inventory_state`` via the real ``update_inventory_*``
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
from typing import NamedTuple

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._enemy_teleport_harness import (
    enemy_module,
    enemy_probe_module,
)
from tests.action_lab._fuel_probe_harness import (
    fuel_probe_module,
    fuel_targeting_module,
)
from tests.action_lab._viewport_probe_harness import viewport_module
from tests.fakes.terrain import InMemoryTerrainMap

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import movement_probe as movement_probe_module
from tankpit_bot.action_lab import queue_probe as queue_probe_module
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_inventory import (
    update_inventory_from_protocol,
)
from tankpit_bot.types import CapturedMessage, decode_capture_session

REPO_ROOT = Path(__file__).resolve().parents[2]

FUEL_PROBE_CAPTURE_PATH = REPO_ROOT / "fuel_probe.capture_session.json"

INVENTORY_GROWTH_FRAME_INDEX = 51
"""First received frame in fuel_probe.capture_session.json whose decode
causes the real inventory tracker total to grow (0 -> 112 via a 0x49 sync).
Discovered by replaying the capture through process_received_message and
watching get_inventory_state(get_world_service())."""

INVENTORY_TOTAL_AFTER_GROWTH = 112


class ReplayPipeline(NamedTuple):
    """A capture's frames paired with the table they were encoded under.

    The table travels WITH the frames because it is session state, not
    process state ([[session-state-deglobalisation]]) — a test holding
    the frames without the table cannot decode them.

    Attributes:
        messages: Every captured message, both directions, in capture
            order.
        xor_table: The table built from this capture's own magic.
    """

    messages: list[CapturedMessage]
    xor_table: bytes


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

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


@pytest.fixture()
def replay_pipeline() -> Generator[ReplayPipeline, None, None]:
    """Reset world state and build the capture's own XOR table.

    Yields the typed captured-message list together with the table
    those frames were encoded under, so tests can replay specific
    frames through the real decoder pipeline via
    ``process_received_message``. Restores world state after.
    """
    session_text = core_hooks.read_text(FUEL_PROBE_CAPTURE_PATH)
    session = decode_capture_session(narrow_json_to_dict(load_json_str(session_text)))
    magic = session["magic"]
    if magic is None:
        raise RuntimeError(
            f"Capture {FUEL_PROBE_CAPTURE_PATH.name} has no magic key — cannot replay binary frames"
        )
    reset_world_state()
    yield ReplayPipeline(
        messages=session["messages"],
        xor_table=build_session_xor_table(magic),
    )
    reset_world_state()


@pytest.fixture()
def real_inventory() -> Generator[None, None, None]:
    """Reset world state before and after the test.

    Tests use ``update_inventory_from_protocol(get_world_service(), [counts], [enabled])`` directly
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
    ``action_hooks.get_current_time_ms``, ``action_hooks.check_and_clear_*``,
    or ``action_hooks.wait_for_*_sync`` don't have to remember to restore —
    this autouse fixture does it. The sync-wait hooks matter for xdist:
    a leaked ``wait_for_radar_sync`` fake survives into whatever test
    module the same worker runs next.
    """
    from tankpit_bot.action_lab._test_hooks import (
        _default_check_and_clear_teleport_landed,
    )
    from tankpit_bot.action_lab.session import (
        wait_for_radar_sync as _real_wait_radar,
    )
    from tankpit_bot.action_lab.session import (
        wait_for_world_sync as _real_wait_world,
    )
    from tankpit_bot.bot.world_sync import drain_messages as _real_drain
    from tankpit_bot.browser import get_current_time_ms as _real_clock
    from tankpit_bot.sniffer.world_state import (
        check_and_clear_radar_scan_complete as _real_clear_radar,
    )

    yield

    action_hooks.drain_buffered_messages = _real_drain
    action_hooks.get_current_time_ms = _real_clock
    action_hooks.check_and_clear_radar_scan_complete = _real_clear_radar
    action_hooks.check_and_clear_teleport_landed = _default_check_and_clear_teleport_landed
    action_hooks.wait_for_world_sync = _real_wait_world
    action_hooks.wait_for_radar_sync = _real_wait_radar


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

    def is_landing_legal(self, x: int, y: int) -> bool:
        return self.is_passable(x, y)

    def is_landing_attainable(self, x: int, y: int) -> bool:
        return self.is_landing_legal(x, y)

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
        get_world_service(),
        counts=[0, total, 0, 0, 0],
        enabled=[True, True, True, True, True],
    )


@pytest.fixture(autouse=True)
def restore_fuel_probe_hooks() -> Generator[None, None, None]:
    """Restore patched fuel-probe module attributes after each test.

    The six ``test_fuel_probe_*`` modules each reach into
    ``fuel_probe_module`` / ``fuel_targeting_module`` to swap internals.
    This lives here rather than in
    :mod:`tests.action_lab._fuel_probe_harness` because a fixture cannot
    travel by import without becoming an unused-name violation at every
    call site. Restoring an attribute a given test never touched is a
    no-op, so the broader autouse scope costs nothing.
    """
    original_get_time = action_hooks.get_current_time_ms
    original_check_radar = action_hooks.check_and_clear_radar_scan_complete
    original_drain = action_hooks.drain_buffered_messages
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_radar_sync = action_hooks.wait_for_radar_sync
    original_get_terrain_map = fuel_probe_module.get_terrain_map
    original_targeting_terrain = fuel_targeting_module.get_terrain_map
    original_wait_outcome = fuel_probe_module._wait_for_teleport_outcome
    original_find_visible = fuel_probe_module._find_visible_fuel_target
    original_requires_reposition = fuel_probe_module._visible_fuel_requires_reposition
    original_find_landing = fuel_probe_module._find_visible_fuel_landing_tile
    original_wait_pickup = fuel_probe_module._wait_for_pickup_outcome
    original_public_requires = fuel_probe_module.visible_fuel_requires_reposition
    original_public_landing = fuel_probe_module.find_visible_fuel_landing_tile
    original_probe_class = fuel_probe_module.FuelProbe
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.check_and_clear_radar_scan_complete = original_check_radar
    action_hooks.drain_buffered_messages = original_drain
    action_hooks.wait_for_world_sync = original_wait_sync
    action_hooks.wait_for_radar_sync = original_wait_radar_sync
    fuel_probe_module.get_terrain_map = original_get_terrain_map
    fuel_targeting_module.get_terrain_map = original_targeting_terrain
    fuel_probe_module._wait_for_teleport_outcome = original_wait_outcome
    fuel_probe_module._find_visible_fuel_target = original_find_visible
    fuel_probe_module._visible_fuel_requires_reposition = original_requires_reposition
    fuel_probe_module._find_visible_fuel_landing_tile = original_find_landing
    fuel_probe_module._wait_for_pickup_outcome = original_wait_pickup
    fuel_probe_module.visible_fuel_requires_reposition = original_public_requires
    fuel_probe_module.find_visible_fuel_landing_tile = original_public_landing
    fuel_probe_module.FuelProbe = original_probe_class


@pytest.fixture(autouse=True)
def restore_enemy_teleport_hooks() -> Generator[None, None, None]:
    """Restore patched enemy-teleport module attributes after each test.

    The four ``test_enemy_teleport_*`` modules reach into
    ``enemy_module`` / ``enemy_probe_module`` to swap internals. It lives
    here rather than in the shared harness because a fixture cannot
    travel by import without becoming an unused-name violation at every
    call site. Restoring an attribute a given test never touched is a
    no-op, so the directory-wide scope costs nothing.
    """
    original_get_time = action_hooks.get_current_time_ms
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_initial = action_hooks.wait_for_initial_self_state
    original_require_enemy = enemy_module._require_fresh_enemy_threat
    original_enemy_by_id = enemy_module._enemy_by_id
    original_choose_landing = enemy_module.choose_combat_landing_tile
    original_wait_outcome = enemy_module._wait_for_teleport_outcome
    original_probe_class = enemy_probe_module.EnemyTeleportProbe
    original_sync_playwright = core_hooks.sync_playwright
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.wait_for_world_sync = original_wait_sync
    action_hooks.wait_for_initial_self_state = original_wait_initial
    enemy_module._require_fresh_enemy_threat = original_require_enemy
    enemy_module._enemy_by_id = original_enemy_by_id
    enemy_module.choose_combat_landing_tile = original_choose_landing
    enemy_module._wait_for_teleport_outcome = original_wait_outcome
    enemy_probe_module.EnemyTeleportProbe = original_probe_class
    core_hooks.sync_playwright = original_sync_playwright


@pytest.fixture(autouse=True)
def restore_movement_probe_hooks() -> Generator[None, None, None]:
    """Restore patched movement-probe hooks after each test.

    Covers what the directory-wide ``restore_action_hooks`` does not:
    ``sync_playwright``, the startup-state hooks, and the three
    ``movement_probe`` module attributes the four
    ``test_movement_probe_*`` modules swap. Autouse here rather than in
    the harness because a fixture cannot travel by import without
    becoming an unused-name violation at every call site.
    """
    original_get_time = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    original_sync_playwright = core_hooks.sync_playwright
    original_wait_for_initial_self_state = action_hooks.wait_for_initial_self_state
    original_advance_startup_state = action_hooks.advance_startup_state
    original_get_terrain_map = movement_probe_module._get_probe_terrain_map
    original_build_targets = movement_probe_module._build_probe_targets
    original_wait_for_move_outcome = movement_probe_module._wait_for_move_outcome
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.drain_buffered_messages = original_drain
    core_hooks.sync_playwright = original_sync_playwright
    action_hooks.wait_for_initial_self_state = original_wait_for_initial_self_state
    action_hooks.advance_startup_state = original_advance_startup_state
    movement_probe_module._get_probe_terrain_map = original_get_terrain_map
    movement_probe_module._build_probe_targets = original_build_targets
    movement_probe_module._wait_for_move_outcome = original_wait_for_move_outcome


@pytest.fixture(autouse=True)
def restore_queue_probe_hooks() -> Generator[None, None, None]:
    """Restore patched queue-probe hooks after each test.

    Covers what the directory-wide ``restore_action_hooks`` does not:
    ``sync_playwright``, the startup-state hooks, and
    ``queue_probe.run_single_experiment``. Autouse here rather than in
    the harness because a fixture cannot travel by import without
    becoming an unused-name violation at every call site.
    """
    orig_time = action_hooks.get_current_time_ms
    orig_drain = action_hooks.drain_buffered_messages
    orig_playwright = core_hooks.sync_playwright
    orig_wait = action_hooks.wait_for_initial_self_state
    orig_advance = action_hooks.advance_startup_state
    orig_run_single = queue_probe_module.run_single_experiment
    yield
    action_hooks.get_current_time_ms = orig_time
    action_hooks.drain_buffered_messages = orig_drain
    core_hooks.sync_playwright = orig_playwright
    action_hooks.wait_for_initial_self_state = orig_wait
    action_hooks.advance_startup_state = orig_advance
    queue_probe_module.run_single_experiment = orig_run_single


@pytest.fixture()
def _all_ground_terrain() -> Generator[None, None, None]:
    original = viewport_module.get_terrain_map
    viewport_module.get_terrain_map = lambda: InMemoryTerrainMap()
    yield
    viewport_module.get_terrain_map = original
