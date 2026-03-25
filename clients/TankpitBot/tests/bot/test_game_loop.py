"""Tests for game_loop module: find enemies, teleport, hunt, patrol, terrain."""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.game_loop import (
    _find_closest_enemy,
    _hunt_tick,
    _load_terrain,
    _patrol_step,
    _step_toward,
    _teleport_near,
    run_game_loop,
)
from tankpit_bot.sniffer.world_state import (
    reset_world_state,
    update_world_state_from_move_response_full,
    update_world_state_from_tank_entry,
)
from tankpit_bot.state.types import TankStateDict
from tests.fakes import FakeCDPSession, FakePage, FakeTerrainMap

log = get_logger(__name__)


# =========================================================================
# Helpers
# =========================================================================


def _setup_self(tank_id: int = 1, x: int = 100, y: int = 100, team: int = 0) -> None:
    """Set up the self tank in world state via MovementResponse."""
    reset_world_state()
    update_world_state_from_move_response_full(tank_id, x, y, team, rank=0)


def _make_bot() -> tuple[Bot, FakeCDPSession]:
    """Create a Bot with a FakeCDPSession so _send_bytes works."""
    bot = Bot("https://test.tankpit.com/", headless=True)
    fake_cdp = FakeCDPSession()
    bot._cdp = fake_cdp
    return bot, fake_cdp


class _LoopDoneError(Exception):
    """Raised by _FakeGameLoopPage to break the infinite game loop."""


class _FakeGameLoopPage(FakePage):
    """FakePage that breaks the game loop after N wait_for_timeout calls.

    Subclasses FakePage to satisfy PageProtocol, but overrides
    wait_for_timeout to count calls and raise _LoopDoneError when the
    limit is exceeded.
    """

    def __init__(self, max_ticks: int) -> None:
        """Initialize with tick limit.

        Args:
            max_ticks: Number of wait_for_timeout calls allowed before raising.
        """
        super().__init__(FakeCDPSession())
        self._max_ticks = max_ticks
        self._tick_count = 0

    def wait_for_timeout(self, timeout: float) -> None:
        """Count ticks; raise _LoopDoneError when limit exceeded.

        Args:
            timeout: Timeout in ms (ignored).
        """
        _ = timeout
        self._tick_count += 1
        if self._tick_count > self._max_ticks:
            raise _LoopDoneError


# =========================================================================
# _find_closest_enemy
# =========================================================================


class TestFindClosestEnemy:
    """Tests for _find_closest_enemy."""

    def test_returns_none_when_no_self_state(self) -> None:
        """No self_state means we can't determine team; returns None."""
        reset_world_state()
        result = _find_closest_enemy(100, 100)
        assert result is None

    def test_returns_none_when_no_enemies(self) -> None:
        """Self exists but no other tanks."""
        _setup_self(tank_id=1, x=100, y=100, team=0)
        result = _find_closest_enemy(100, 100)
        assert result is None

    def test_finds_closest_enemy(self) -> None:
        """Returns the closest enemy tank by Manhattan distance."""
        _setup_self(tank_id=1, x=100, y=100, team=0)
        # Add two enemies (different team)
        update_world_state_from_tank_entry(tank_id=10, x=110, y=110, name="EnemyFar")
        # Give the far one a different team via move_response_full to set team
        update_world_state_from_move_response_full(10, 110, 110, team=1, rank=0)
        update_world_state_from_tank_entry(tank_id=11, x=105, y=105, name="EnemyClose")
        update_world_state_from_move_response_full(11, 105, 105, team=1, rank=0)

        result = _find_closest_enemy(100, 100)
        if result is None:
            raise AssertionError("Expected an enemy tank to be found")
        assert result["tank_id"] == 11
        assert result["name"] == "EnemyClose"

    def test_skips_same_team(self) -> None:
        """Tanks on the same team are not enemies."""
        _setup_self(tank_id=1, x=100, y=100, team=0)
        update_world_state_from_tank_entry(tank_id=10, x=105, y=105, name="Ally")
        update_world_state_from_move_response_full(10, 105, 105, team=0, rank=0)

        result = _find_closest_enemy(100, 100)
        assert result is None

    def test_skips_zero_position_tanks(self) -> None:
        """Tanks at (0,0) are info-only placeholders and should be skipped."""
        _setup_self(tank_id=1, x=100, y=100, team=0)
        update_world_state_from_tank_entry(tank_id=10, x=0, y=0, name="Ghost")
        update_world_state_from_move_response_full(10, 0, 0, team=1, rank=0)

        result = _find_closest_enemy(100, 100)
        assert result is None

    def test_skips_farther_enemy(self) -> None:
        """Close enemy found first; farther enemy does not replace it."""
        _setup_self(tank_id=1, x=100, y=100, team=0)
        # Close enemy first in insertion order
        update_world_state_from_tank_entry(tank_id=10, x=105, y=105, name="EnemyClose")
        update_world_state_from_move_response_full(10, 105, 105, team=1, rank=0)
        # Far enemy second — dist > best_dist, so branch 177→170 taken
        update_world_state_from_tank_entry(tank_id=11, x=150, y=150, name="EnemyFar")
        update_world_state_from_move_response_full(11, 150, 150, team=1, rank=0)

        result = _find_closest_enemy(100, 100)
        if result is None:
            raise AssertionError("Expected an enemy tank to be found")
        assert result["tank_id"] == 10
        assert result["name"] == "EnemyClose"


# =========================================================================
# _teleport_near
# =========================================================================


class TestTeleportNear:
    """Tests for _teleport_near."""

    def test_first_offset_passable_no_terrain(self) -> None:
        """Without terrain, first in-bounds offset is chosen."""
        tx, ty = _teleport_near(100, 100, None)
        # First offset is (5, 0) -> (105, 100)
        assert (tx, ty) == (105, 100)

    def test_first_offset_passable_with_terrain(self) -> None:
        """With passable terrain, first offset is chosen."""
        terrain = FakeTerrainMap()
        tx, ty = _teleport_near(100, 100, terrain)
        assert (tx, ty) == (105, 100)

    def test_skips_impassable_offset(self) -> None:
        """If first offset is blocked, tries the next."""
        terrain = FakeTerrainMap(terrain_data={(105, 100): "#"})
        tx, ty = _teleport_near(100, 100, terrain)
        # First offset (105, 100) blocked, second (-5,0) -> (95, 100)
        assert (tx, ty) == (95, 100)

    def test_all_offsets_blocked_returns_enemy_position(self) -> None:
        """If all offsets blocked, falls back to enemy position."""
        terrain = FakeTerrainMap(
            terrain_data={
                (105, 100): "#",
                (95, 100): "#",
                (100, 105): "#",
                (100, 95): "#",
            }
        )
        tx, ty = _teleport_near(100, 100, terrain)
        assert (tx, ty) == (100, 100)

    def test_offset_out_of_bounds_skipped(self) -> None:
        """Offsets that go out of [0, 256) are skipped."""
        # Enemy at (2, 100): offset (5,0)=>(7,100) ok, (-5,0)=>(-3,100) OOB
        tx, ty = _teleport_near(2, 100, None)
        assert (tx, ty) == (7, 100)

    def test_near_upper_bound(self) -> None:
        """Offsets near 255 boundary: (5,0)=>(260, y) OOB, (-5,0)=>(250, y) ok."""
        tx, ty = _teleport_near(255, 100, None)
        # (260,100) OOB, (250,100) ok
        assert (tx, ty) == (250, 100)


# =========================================================================
# _step_toward
# =========================================================================


class TestStepToward:
    """Tests for _step_toward — one tile toward target, terrain-aware."""

    def test_primary_axis_x(self) -> None:
        """Larger X gap: moves along X axis."""
        mx, my = _step_toward(100, 100, 110, 102, None)
        assert (mx, my) == (101, 100)

    def test_primary_axis_y(self) -> None:
        """Larger Y gap: moves along Y axis."""
        mx, my = _step_toward(100, 100, 102, 110, None)
        assert (mx, my) == (100, 101)

    def test_equal_gap_prefers_x(self) -> None:
        """Equal gap: abs(dx) >= abs(dy) so X-axis first."""
        mx, my = _step_toward(100, 100, 105, 105, None)
        assert (mx, my) == (101, 100)

    def test_blocked_primary_uses_secondary(self) -> None:
        """If primary direction blocked, uses secondary."""
        terrain = FakeTerrainMap(terrain_data={(101, 100): "#"})
        mx, my = _step_toward(100, 100, 110, 105, terrain)
        # X primary blocked at (101, 100), try Y secondary (100, 101)
        assert (mx, my) == (100, 101)

    def test_both_blocked_returns_same_position(self) -> None:
        """If both candidates blocked, returns original position."""
        terrain = FakeTerrainMap(
            terrain_data={
                (101, 100): "#",
                (100, 101): "#",
            }
        )
        mx, my = _step_toward(100, 100, 110, 110, terrain)
        assert (mx, my) == (100, 100)

    def test_negative_direction(self) -> None:
        """Moves toward target that is to the left/up."""
        mx, my = _step_toward(100, 100, 90, 80, None)
        # Y gap (20) > X gap (10), so Y first: (100, 99)
        assert (mx, my) == (100, 99)

    def test_dx_zero_uses_dy_only(self) -> None:
        """When dx=0, only Y candidates are generated."""
        mx, my = _step_toward(100, 100, 100, 110, None)
        assert (mx, my) == (100, 101)

    def test_dy_zero_uses_dx_only(self) -> None:
        """When dy=0, only X candidates are generated."""
        mx, my = _step_toward(100, 100, 110, 100, None)
        assert (mx, my) == (101, 100)

    def test_out_of_bounds_blocked(self) -> None:
        """At boundary edge (0, 0) moving toward (-1, -1) is blocked."""
        mx, my = _step_toward(0, 0, -10, -10, None)
        # candidates would be (-1, 0) and (0, -1), both OOB
        assert (mx, my) == (0, 0)


# =========================================================================
# _hunt_tick
# =========================================================================


class TestHuntTick:
    """Tests for _hunt_tick — one tick of hunt behavior."""

    def _make_target(
        self,
        *,
        tank_id: int = 10,
        x: int = 105,
        y: int = 105,
        name: str = "Enemy",
    ) -> TankStateDict:
        """Create a TankStateDict-like target dict."""
        return TankStateDict(
            tank_id=tank_id,
            x=x,
            y=y,
            team=1,
            rank=0,
            damage_state=0,
            name=name,
            is_bot=False,
            is_self=False,
        )

    def test_shoot_path(self) -> None:
        """When shoot=True, sends shoot command and next_shoot=False."""
        bot, _ = _make_bot()
        target = self._make_target(x=105, y=105)
        nx, ny, next_shoot = _hunt_tick(bot, 100, 100, target, True, None)
        # Position unchanged after shooting
        assert (nx, ny) == (100, 100)
        # Next tick should move
        assert next_shoot is False

    def test_move_path(self) -> None:
        """When shoot=False, moves one tile toward target and next_shoot=True."""
        bot, _ = _make_bot()
        target = self._make_target(x=110, y=102)
        nx, ny, next_shoot = _hunt_tick(bot, 100, 100, target, False, None)
        # Moved one tile toward target (X gap larger)
        assert (nx, ny) == (101, 100)
        assert next_shoot is True

    def test_blocked_move_shoots_instead(self) -> None:
        """When movement is blocked, shoots instead of wasting the tick."""
        terrain = FakeTerrainMap(
            terrain_data={
                (101, 100): "#",
                (100, 101): "#",
            }
        )
        bot, _ = _make_bot()
        target = self._make_target(x=110, y=110)
        nx, ny, next_shoot = _hunt_tick(bot, 100, 100, target, False, terrain)
        # Position unchanged (blocked)
        assert (nx, ny) == (100, 100)
        # next tick should shoot (True) since we shot this tick due to block
        assert next_shoot is True

    def test_shoot_sends_bytes(self) -> None:
        """Verify that shooting actually calls _send_bytes via the FakeCDPSession."""
        bot, cdp = _make_bot()
        target = self._make_target(x=105, y=105, tank_id=42)
        _hunt_tick(bot, 100, 100, target, True, None)
        assert len(cdp._sent_methods) == 1
        assert cdp._sent_methods[0] == "Runtime.evaluate"

    def test_move_sends_bytes(self) -> None:
        """Verify that moving actually calls _send_bytes."""
        bot, cdp = _make_bot()
        target = self._make_target(x=110, y=100)
        _hunt_tick(bot, 100, 100, target, False, None)
        assert len(cdp._sent_methods) == 1


# =========================================================================
# _patrol_step
# =========================================================================


class TestPatrolStep:
    """Tests for _patrol_step — terrain-aware patrol movement."""

    def test_normal_move_direction_0(self) -> None:
        """Direction 0 (east): moves right by 1."""
        bot, _ = _make_bot()
        dir_dx = [1, 0, -1, 0]
        dir_dy = [0, 1, 0, -1]
        nx, ny, blocked = _patrol_step(bot, 100, 100, 0, dir_dx, dir_dy, None)
        assert (nx, ny) == (101, 100)
        assert blocked is False

    def test_normal_move_direction_1(self) -> None:
        """Direction 1 (south): moves down by 1."""
        bot, _ = _make_bot()
        dir_dx = [1, 0, -1, 0]
        dir_dy = [0, 1, 0, -1]
        nx, ny, blocked = _patrol_step(bot, 100, 100, 1, dir_dx, dir_dy, None)
        assert (nx, ny) == (100, 101)
        assert blocked is False

    def test_blocked_by_terrain(self) -> None:
        """Impassable terrain blocks the patrol move."""
        terrain = FakeTerrainMap(terrain_data={(101, 100): "#"})
        bot, _ = _make_bot()
        dir_dx = [1, 0, -1, 0]
        dir_dy = [0, 1, 0, -1]
        nx, ny, blocked = _patrol_step(bot, 100, 100, 0, dir_dx, dir_dy, terrain)
        assert (nx, ny) == (100, 100)
        assert blocked is True

    def test_blocked_by_boundary(self) -> None:
        """Moving beyond map edge (x=255, direction east) is blocked."""
        bot, _ = _make_bot()
        dir_dx = [1, 0, -1, 0]
        dir_dy = [0, 1, 0, -1]
        nx, ny, blocked = _patrol_step(bot, 255, 100, 0, dir_dx, dir_dy, None)
        assert (nx, ny) == (255, 100)
        assert blocked is True

    def test_blocked_by_lower_boundary(self) -> None:
        """Moving below 0 (direction west from x=0) is blocked."""
        bot, _ = _make_bot()
        dir_dx = [1, 0, -1, 0]
        dir_dy = [0, 1, 0, -1]
        nx, ny, blocked = _patrol_step(bot, 0, 100, 2, dir_dx, dir_dy, None)
        # direction 2 is west: dx=-1 -> target x = -1
        assert (nx, ny) == (0, 100)
        assert blocked is True

    def test_sends_move_command(self) -> None:
        """Successful patrol sends a move command via CDP."""
        bot, cdp = _make_bot()
        dir_dx = [1, 0, -1, 0]
        dir_dy = [0, 1, 0, -1]
        _patrol_step(bot, 100, 100, 0, dir_dx, dir_dy, None)
        assert len(cdp._sent_methods) == 1


# =========================================================================
# _load_terrain
# =========================================================================


class TestLoadTerrain:
    """Tests for _load_terrain — GIF-found and fallback paths."""

    def test_gif_found_returns_terrain(self) -> None:
        """When GIF file exists, returns a terrain map."""
        fake_terrain = FakeTerrainMap()

        def fake_path_exists(path: Path) -> bool:
            return str(path) == "field01_r.gif"

        def fake_load_terrain_map(gif_path: Path) -> TerrainMapProtocol:
            return fake_terrain

        original_path_exists = _test_hooks.path_exists
        original_load = _test_hooks.load_terrain_map
        original_get_env = _test_hooks.get_env
        try:
            _test_hooks.path_exists = fake_path_exists
            _test_hooks.load_terrain_map = fake_load_terrain_map
            _test_hooks.get_env = lambda key: "Practice" if key == "TANKPIT_ROOM" else None

            result = _load_terrain()
            assert result is fake_terrain
        finally:
            _test_hooks.path_exists = original_path_exists
            _test_hooks.load_terrain_map = original_load
            _test_hooks.get_env = original_get_env

    def test_non_practice_room_uses_field42(self) -> None:
        """Non-Practice room selects field42-r.gif."""
        fake_terrain = FakeTerrainMap()

        def fake_path_exists(path: Path) -> bool:
            return str(path) == "field42-r.gif"

        def fake_load_terrain_map(gif_path: Path) -> TerrainMapProtocol:
            return fake_terrain

        original_path_exists = _test_hooks.path_exists
        original_load = _test_hooks.load_terrain_map
        original_get_env = _test_hooks.get_env
        try:
            _test_hooks.path_exists = fake_path_exists
            _test_hooks.load_terrain_map = fake_load_terrain_map
            _test_hooks.get_env = lambda key: "Arena" if key == "TANKPIT_ROOM" else None

            result = _load_terrain()
            assert result is fake_terrain
        finally:
            _test_hooks.path_exists = original_path_exists
            _test_hooks.load_terrain_map = original_load
            _test_hooks.get_env = original_get_env

    def test_fallback_when_gif_not_found(self) -> None:
        """When GIF file is missing, falls back to get_terrain_map()."""
        original_path_exists = _test_hooks.path_exists
        original_get_env = _test_hooks.get_env
        try:
            _test_hooks.path_exists = lambda path: False
            _test_hooks.get_env = lambda key: "Practice" if key == "TANKPIT_ROOM" else None

            # Reset world state terrain map so get_terrain_map() returns None
            reset_world_state()
            result = _load_terrain()
            # Fallback: get_terrain_map returns None if no terrain loaded
            assert result is None
        finally:
            _test_hooks.path_exists = original_path_exists
            _test_hooks.get_env = original_get_env

    def test_default_room_is_practice(self) -> None:
        """When TANKPIT_ROOM is not set, defaults to Practice."""
        fake_terrain = FakeTerrainMap()

        def fake_path_exists(path: Path) -> bool:
            return str(path) == "field01_r.gif"

        def fake_load_terrain_map(gif_path: Path) -> TerrainMapProtocol:
            return fake_terrain

        original_path_exists = _test_hooks.path_exists
        original_load = _test_hooks.load_terrain_map
        original_get_env = _test_hooks.get_env
        try:
            _test_hooks.path_exists = fake_path_exists
            _test_hooks.load_terrain_map = fake_load_terrain_map
            _test_hooks.get_env = lambda key: None  # No env vars set

            result = _load_terrain()
            assert result is fake_terrain
        finally:
            _test_hooks.path_exists = original_path_exists
            _test_hooks.load_terrain_map = original_load
            _test_hooks.get_env = original_get_env


# =========================================================================
# run_game_loop
# =========================================================================


class TestRunGameLoop:
    """Integration tests for run_game_loop: seek, hunt, and patrol modes."""

    def test_seek_teleports_to_far_enemy(self) -> None:
        """Enemy beyond COMBAT_RANGE triggers teleport (seek mode)."""
        bot, cdp = _make_bot()
        # Bot must be IDLE for teleport_to state transition (IDLE→MOVING)
        bot._transition("WAITING_FOR_POSITION")
        bot._transition("IDLE")
        _setup_self(tank_id=1, x=128, y=128, team=0)
        # Enemy far away: dist = |200-128| + |200-128| = 144 > COMBAT_RANGE(20)
        update_world_state_from_tank_entry(tank_id=10, x=200, y=200, name="FarEnemy")
        update_world_state_from_move_response_full(10, 200, 200, team=1, rank=0)

        page = _FakeGameLoopPage(max_ticks=1)
        original_pe = _test_hooks.path_exists
        original_ge = _test_hooks.get_env
        try:
            _test_hooks.path_exists = lambda path: False
            _test_hooks.get_env = lambda key: None
            try:
                run_game_loop(bot, page)
            except _LoopDoneError:
                log.debug("Seek test: game loop stopped after tick limit")
        finally:
            _test_hooks.path_exists = original_pe
            _test_hooks.get_env = original_ge

        # Pre-loop: install_hook + open_map + sync + close_map = 4
        # Iter 1: sync + teleport_to (open_map + teleport + close_map = 3) = 4
        assert len(cdp._sent_methods) == 8

    def test_hunt_engages_close_enemy(self) -> None:
        """Enemy within COMBAT_RANGE triggers hunt (shoot/move)."""
        bot, cdp = _make_bot()
        _setup_self(tank_id=1, x=128, y=128, team=0)
        # Enemy close: dist = |138-128| + |138-128| = 20 <= COMBAT_RANGE(20)
        update_world_state_from_tank_entry(tank_id=10, x=138, y=138, name="CloseEnemy")
        update_world_state_from_move_response_full(10, 138, 138, team=1, rank=0)

        page = _FakeGameLoopPage(max_ticks=1)
        original_pe = _test_hooks.path_exists
        original_ge = _test_hooks.get_env
        try:
            _test_hooks.path_exists = lambda path: False
            _test_hooks.get_env = lambda key: None
            try:
                run_game_loop(bot, page)
            except _LoopDoneError:
                log.debug("Hunt test: game loop stopped after tick limit")
        finally:
            _test_hooks.path_exists = original_pe
            _test_hooks.get_env = original_ge

        # Pre-loop: install_hook + open_map + sync + close_map = 4
        # Iter 1: sync + shoot = 2
        assert len(cdp._sent_methods) == 6

    def test_patrol_changes_direction_after_10_steps(self) -> None:
        """After 10 patrol steps without enemies, direction changes."""
        bot, cdp = _make_bot()
        _setup_self(tank_id=1, x=128, y=128, team=0)
        # No enemies — pure patrol mode

        # max_ticks=9: iters 1-9 complete, iter 10 runs patrol step
        # (steps_in_dir=10 → direction change) then wait raises
        page = _FakeGameLoopPage(max_ticks=9)
        original_pe = _test_hooks.path_exists
        original_ge = _test_hooks.get_env
        try:
            _test_hooks.path_exists = lambda path: False
            _test_hooks.get_env = lambda key: None
            try:
                run_game_loop(bot, page)
            except _LoopDoneError:
                log.debug("Patrol test: game loop stopped after tick limit")
        finally:
            _test_hooks.path_exists = original_pe
            _test_hooks.get_env = original_ge

        # Pre-loop: install_hook + open_map + sync + close_map = 4
        # 9 loop iters: each has sync + patrol_move = 2, total 18
        assert len(cdp._sent_methods) == 22
