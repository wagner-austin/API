"""Tests for the sniffer formatters.

``test_formatters_details.py`` was 606 lines; the per-message
formatters are now a sibling.
"""

from __future__ import annotations

from tankpit_bot.sniffer.formatters import (
    format_combat_details,
    format_position_details,
    format_resource_details,
    format_tank_details,
)


class TestFormatFunctions:
    """Tests for message format functions using proper TypedDict instances."""

    def test_format_combat_details_shooting(self) -> None:
        """Format shoot/hit message (0x53)."""
        from tankpit_bot.protocol import ShootEventDict

        msg = ShootEventDict(
            msg_type=0x53,
            team=2,
            shooter_id=1301,
            source_x=155,
            source_y=154,
            target_x=155,
            target_y=155,
            aim_x=155,
            aim_y=155,
            weapon=1,
        )
        result = format_combat_details(msg)
        assert "shooter=1301" in result
        assert "src=(155,154)" in result
        assert "tgt=(155,155)" in result
        assert "dual" in result

    def test_format_combat_details_deactivation(self) -> None:
        """Test format_combat_details for deactivation message (0x41)."""
        from tankpit_bot.protocol import DeactivationDict

        msg = DeactivationDict(
            msg_type=0x41,
            status=0,
            victim_id=50,
            promo_eligible=False,
            killer_id=100,
            is_mine_kill=False,
        )
        result = format_combat_details(msg)
        assert "victim=50" in result
        assert "killer=100" in result

    def test_format_combat_details_unknown(self) -> None:
        """Test format_combat_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_combat_details
        msg = SyncDict(msg_type=0x3F)
        result = format_combat_details(msg)
        assert result == ""

    def test_format_tank_details_entry(self) -> None:
        """Test format_tank_details for tank entry (0x28)."""
        from tankpit_bot.protocol import TankEntryDict

        msg = TankEntryDict(
            msg_type=0x28,
            team=0,
            tank_id=100,
            rank=0,
            damage_state=0,
            score=0,
            x=50,
            y=60,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "team=0" in result

    def test_format_tank_details_remove(self) -> None:
        """Test format_tank_details for tank remove (0x58)."""
        from tankpit_bot.protocol import TankRemoveDict

        msg = TankRemoveDict(msg_type=0x58, tank_id=100)
        result = format_tank_details(msg)
        assert "tank=100 removed" in result

    def test_format_tank_details_exit_eliminated(self) -> None:
        """Test format_tank_details for 0x29 TankExit (eliminated, non-silent)."""
        from tankpit_bot.protocol import TankExitDict

        msg = TankExitDict(
            msg_type=0x29,
            team=2,
            tank_id=100,
            was_silent=False,
            was_eliminated=True,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "team=2" in result
        assert "eliminated" in result
        assert "silent" not in result

    def test_format_tank_details_exit_left_silent(self) -> None:
        """Test format_tank_details for 0x29 TankExit (left, silent)."""
        from tankpit_bot.protocol import TankExitDict

        msg = TankExitDict(
            msg_type=0x29,
            team=1,
            tank_id=200,
            was_silent=True,
            was_eliminated=False,
        )
        result = format_tank_details(msg)
        assert "tank=200" in result
        assert "left" in result
        assert "silent" in result

    def test_format_tank_details_status_sync(self) -> None:
        """Test format_tank_details for tank status sync (0x2E)."""
        from tankpit_bot.protocol import TankStatusSyncDict

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=1,
            tank_id=100,
            damage_state=1,
            rank=3,
            lb_score=5,
            promo_state=0,
            promo_bar_lit=None,
            fuel=None,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "sergeant" in result  # rank 3
        # Tier = fuel quartile (0 = near death .. 3 = full): 1 is medium.
        assert "medium" in result  # damage_state 1
        assert "lb=5" in result

    def test_format_tank_details_status(self) -> None:
        """Test format_tank_details for tank status (0x3E)."""
        from tankpit_bot.protocol import TankStatusDict

        msg = TankStatusDict(
            msg_type=0x3E,
            team=2,
            rank=5,
            damage_state=0,
            tank_id=100,
            decoration_state=b"\x00\x00\x00\x00",
            leaderboard_score=1000,
            leaderboard_position=0,
            name="",
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "captain" in result  # rank 5
        assert "blue" in result  # team 2
        assert "lb=1000" in result

    def test_format_tank_details_info(self) -> None:
        """Test format_tank_details for tank info (0x21)."""
        from tankpit_bot.protocol import TankInfoDict

        msg = TankInfoDict(
            msg_type=0x21,
            tank_id=100,
            team=1,
            decoration_state=b"\x00\x00\x00\x00",
            persistent_tank_id=0,
            name="InfoTank",
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "purple" in result  # team 1
        assert "InfoTank" in result

    def test_format_tank_details_movement(self) -> None:
        """Test format_tank_details for movement (0x47)."""
        from tankpit_bot.protocol import MovementDict

        msg = MovementDict(
            msg_type=0x47,
            tank_id=100,
            start_x=50,
            start_y=60,
            direction=2,
            flag=0,
            lb_score=500,
            rank=1,
            damage_state=0,
            is_carrying=False,
            waypoints=[],
            path_tiles=0,
            path="",
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "dir=2" in result
        assert "lb=500" in result

    def test_format_tank_details_movement_response(self) -> None:
        """Test format_tank_details for movement response (0x3D)."""
        from tankpit_bot.protocol import MovementResponseDict

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=100,
            x=50,
            y=60,
            direction=2,
            damage_state=0,
            rank=4,
            lb_score=10,
            carrying=0,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "dir=2" in result
        assert "lieutenant" in result  # rank 4
        assert "lb=10" in result

    def test_format_tank_details_0x48(self) -> None:
        """Test format_tank_details for 0x48 message (EnemyDetection)."""
        from tankpit_bot.protocol import EnemyDetectionDict

        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=100,
            x=50,
            y=60,
            rank=6,
            team=0,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "major" in result  # rank 6

    def test_format_tank_details_unknown(self) -> None:
        """Test format_tank_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_tank_details
        msg = SyncDict(msg_type=0x3F)
        result = format_tank_details(msg)
        assert result == ""

    def test_format_resource_details_fuel_refill(self) -> None:
        """Test format_resource_details for fuel refill (0x44)."""
        from tankpit_bot.protocol import FuelGainDict

        msg = FuelGainDict(msg_type=0x44, fuel_total=500, is_free=True, flag=0)
        result = format_resource_details(msg)
        assert "fuel=500" in result
        assert "free=True" in result

    def test_format_resource_details_fuel_deposit(self) -> None:
        """Test format_resource_details for fuel deposit (0x64)."""
        from tankpit_bot.protocol import FuelDepositDict

        msg = FuelDepositDict(msg_type=0x64, fuel_total=1000)
        result = format_resource_details(msg)
        assert "fuel=1000" in result

    def test_format_resource_details_item_pickup(self) -> None:
        """Test format_resource_details for item pickup (0x49)."""
        from tankpit_bot.protocol import InventoryDict

        msg = InventoryDict(
            msg_type=0x49,
            show=True,
            alternate=False,
            counts=[1, 2, 3, 0, 1],
            enabled=[True, True, True, True, True],
        )
        result = format_resource_details(msg)
        assert "counts=" in result

    def test_format_resource_details_cache_update(self) -> None:
        """Test format_resource_details for cache update (0x43)."""
        from tankpit_bot.protocol import CacheUpdateDict

        msg = CacheUpdateDict(msg_type=0x43, updates=[(10, 20, 100), (30, 40, -1)])
        result = format_resource_details(msg)
        assert "updates=2" in result

    def test_format_resource_details_unknown(self) -> None:
        """Test format_resource_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_resource_details
        msg = SyncDict(msg_type=0x3F)
        result = format_resource_details(msg)
        assert result == ""

    def test_format_position_details_mine_placement(self) -> None:
        """Test format_position_details for mine placement (0x4B)."""
        from tankpit_bot.container import MinePlacementDict

        msg = MinePlacementDict(
            msg_type=0x4B,
            mine_type=0,
            tank_id=100,
            positions=[(10, 20), (30, 40)],
        )
        result = format_position_details(msg)
        assert "tank=100" in result
        assert "count=2" in result
