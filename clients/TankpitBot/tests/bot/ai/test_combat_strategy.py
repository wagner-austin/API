"""Focused tests for combat route primitives."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import (
    _combat_landing_candidates,
    select_new_combat_target,
    teleport_to_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import EnemyThreatDict
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _enemy_threat(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "Enemy",
) -> EnemyThreatDict:
    """Create a typed enemy threat for combat helper tests.

    Args:
        tank_id: Enemy tank identifier.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.

    Returns:
        Enemy threat payload.
    """
    return EnemyThreatDict(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=abs(x - 100) + abs(y - 100),
        damage_state=0,
        rank=1,
        team=2,
        name=name,
        is_bot=False,
        timestamp_ms=100000,
    )


class TestCombatTargetSelection:
    """Tests for combat target selection constraints."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_select_new_combat_target_allows_non_emergency_reserve_band(self) -> None:
        """Combat still acquires targets above the break threshold."""
        world, self_state = make_world(fuel=800)
        inventory = make_inventory()
        inventory["dual_shots"]["count"] = 10
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            None,
            "",
        )

        result = select_new_combat_target(ctx, [_enemy_threat()])

        if result is None:
            raise AssertionError("expected viable combat target in non-emergency reserve band")
        assert result["tank_id"] == 50

    def test_select_new_combat_target_skips_blocked_and_killed_targets(self) -> None:
        """Combat target selection ignores blocked and killed enemies."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["blocked_combat_targets"] = {"50": 99000}
        ai_state["killed_tank_ids"] = {"60": 99500}
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            None,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [
                _enemy_threat(tank_id=50, name="BlockedEnemy"),
                _enemy_threat(tank_id=60, x=118, name="KilledEnemy"),
                _enemy_threat(tank_id=70, x=116, name="ViableEnemy"),
            ],
        )

        if result is None:
            raise AssertionError("expected viable combat target")
        assert result["tank_id"] == 70
        assert result["name"] == "ViableEnemy"

    def test_select_new_combat_target_returns_none_when_no_viable_targets(self) -> None:
        """Combat target selection returns None when every threat is excluded."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state["blocked_combat_targets"] = {"50": 99000}
        ai_state["killed_tank_ids"] = {"60": 99500}
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            None,
            "",
        )

        result = select_new_combat_target(
            ctx,
            [
                _enemy_threat(tank_id=50, name="BlockedEnemy"),
                _enemy_threat(tank_id=60, x=118, name="KilledEnemy"),
            ],
        )

        assert result is None


def test_combat_landing_candidates_delegate_to_shared_helper() -> None:
    """Combat landing candidates expose shared adjacent-tile ordering."""
    world, self_state = make_world(fuel=800)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
    )

    assert _combat_landing_candidates(ctx, _enemy_threat()) == [
        (119, 100),
        (121, 100),
        (120, 101),
        (120, 99),
    ]


class TestCombatTeleportGuards:
    """Tests for combat teleport affordability and legality."""

    def setup_method(self) -> None:
        """Reset shared world-state globals before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset shared world-state globals after each test."""
        reset_world_state()

    def test_teleport_to_target_returns_none_when_unaffordable(self) -> None:
        """Combat teleport refuses targets that violate the exact fuel guard."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=190,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=520, tanks=tanks)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        result = teleport_to_target(ctx, _enemy_threat(x=190, y=100, name="FarEnemy"))

        assert result is None

    def test_teleport_to_target_returns_teleport_for_affordable_close(self) -> None:
        """Combat teleport emits a teleport decision when the landing is affordable."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="CloseEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )

        result = teleport_to_target(ctx, _enemy_threat(x=120, y=100, name="CloseEnemy"))

        if result is None:
            raise AssertionError("expected teleport decision")
        assert result["command"]["cmd_type"] == "teleport"
        assert result["behavior"]["reason"] == "teleport CloseEnemy"
