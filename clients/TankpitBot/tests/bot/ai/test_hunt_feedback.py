"""Feedback and cooldown integration tests for HUNT routing."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


class TestDecideCombatFeedback:
    """Tests for combat feedback handling in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_hit_feedback_after_kill_sees_no_enemy(self) -> None:
        """After a kill, deactivated enemies no longer participate in threat selection."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=0,
                y=0,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = make_scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state_with_shot, inventory, 100000, None, "hit")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"
        assert decision["updated_ai_state"]["last_shot_target_id"] == -1

    def test_miss_with_no_target_in_world_opens_map(self) -> None:
        """Miss feedback with no target state falls through to reacquisition."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state_with_shot, inventory, 100000, None, "miss")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"

    def test_hit_feedback_continues_normally(self) -> None:
        """Hit feedback preserves normal combat routing when the target remains visible."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = make_scanned_ai_state()
        ai_state_with_shot = AIStateDict(
            **{
                **ai_state,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
                "last_map_open_ms": 94000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state_with_shot, inventory, 100000, None, "hit")

        assert decision["behavior"]["reason"] != "kill_confirmed"
        assert decision["behavior"]["reason"] != "miss_relocate"
        assert decision["command"]["cmd_type"] == "map_open"

    def test_no_feedback_when_no_shot_pending(self) -> None:
        """Empty combat feedback leaves normal planning unchanged."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"


class TestDecideShotTracking:
    """Tests for shot target tracking in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_shoot_command_records_target(self) -> None:
        """Shoot decisions record the target for next-tick feedback handling."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=103,
                y=103,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        if decision["command"]["cmd_type"] == "shoot":
            assert decision["updated_ai_state"]["last_shot_target_id"] == 50
            assert decision["updated_ai_state"]["last_shot_target_name"] == "Enemy"


class TestDecideKillCooldown:
    """Tests for kill cooldown filtering in decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_killed_tanks_filtered_from_world(self) -> None:
        """Killed tanks do not remain eligible HUNT targets."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="KilledEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "killed_tank_ids": {"50": 90000},
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        if decision["behavior"]["mode"] == "HUNT":
            assert (
                decision["behavior"]["target_x"],
                decision["behavior"]["target_y"],
            ) != (105, 105)

    def test_miss_reopens_map_when_target_far(self) -> None:
        """Miss feedback returns HUNT to reacquisition for distant targets."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=110,
                y=110,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 110,
                "combat_target_y": 110,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "miss")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_miss_always_reacquires_even_when_close(self) -> None:
        """Miss feedback triggers reacquisition even when the target was adjacent."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "miss")

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_closing_recloses_when_not_cardinally_adjacent(self) -> None:
        """Diagonal landing positions continue HUNT close behavior instead of shooting."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=99, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 100,
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "")

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["behavior"]["reason"] == "teleport Enemy"

    def test_closing_shoots_when_cardinally_adjacent(self) -> None:
        """Closing combat engages once the landed position is cardinally usable."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=99,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=99, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 101,
                "combat_target_y": 99,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, "")

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["updated_ai_state"]["last_shot_target_id"] == 50

    def test_expired_kills_removed(self) -> None:
        """Expired kill cooldown entries are removed from the updated AI state."""
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "killed_tank_ids": {"50": 50000},
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert "50" not in decision["updated_ai_state"]["killed_tank_ids"]
