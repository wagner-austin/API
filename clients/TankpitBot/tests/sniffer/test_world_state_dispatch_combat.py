"""Tests for sniffer world state combat hit and teleport landed tracking."""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_combat_hit,
    check_and_clear_our_shot_response,
    check_and_clear_teleport_landed,
    mark_combat_hit,
    mark_teleport_landed,
    peek_combat_hit,
    peek_our_shot_response,
)
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.sniffer.world_state_inventory import (
    get_inventory_state,
    update_inventory_from_protocol,
)


class TestCombatHitTracking:
    """Tests for mark_combat_hit and check_and_clear_combat_hit."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_check_and_clear_returns_false_by_default(self) -> None:
        """check_and_clear_combat_hit returns False when no hit recorded."""
        assert check_and_clear_combat_hit(get_world_service()) is False

    def test_mark_and_check_returns_true(self) -> None:
        """mark_combat_hit sets flag, check_and_clear returns True then False."""
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert check_and_clear_combat_hit(get_world_service()) is True
        # Second call returns False (cleared)
        assert check_and_clear_combat_hit(get_world_service()) is False

    def test_mark_with_zero_weapon_byte_is_miss(self) -> None:
        """mark_combat_hit with weapon_byte=0 does not set hit flag."""
        mark_combat_hit(get_world_service(), weapon_byte=0)
        assert check_and_clear_combat_hit(get_world_service()) is False

    def test_zero_weapon_byte_sets_our_shot_response(self) -> None:
        """mark_combat_hit with weapon_byte=0 still sets the shot response flag."""
        mark_combat_hit(get_world_service(), weapon_byte=0)
        assert peek_our_shot_response(get_world_service()) is True
        assert check_and_clear_our_shot_response(get_world_service()) is True
        assert check_and_clear_our_shot_response(get_world_service()) is False

    def test_nonzero_weapon_byte_sets_our_shot_response(self) -> None:
        """mark_combat_hit with weapon_byte>0 sets both hit and response flags."""
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert peek_our_shot_response(get_world_service()) is True
        assert peek_combat_hit(get_world_service()) is True

    def test_our_shot_response_default_false(self) -> None:
        """check_and_clear_our_shot_response returns False by default."""
        assert check_and_clear_our_shot_response(get_world_service()) is False
        assert peek_our_shot_response(get_world_service()) is False

    def test_dual_hit_decrements_dual_count(self) -> None:
        """mark_combat_hit with weapon_byte=1 decrements dual_shots count."""
        update_inventory_from_protocol(
            get_world_service(),
            [0, 7, 0, 0, 10],
            [False, True, False, False, True],
        )
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert get_inventory_state(get_world_service())["dual_shots"]["count"] == 6
        assert get_inventory_state(get_world_service())["dual_shots"]["enabled"] is True

    def test_missile_hit_decrements_missile_count(self) -> None:
        """mark_combat_hit with weapon_byte=2 decrements missile_shots count."""
        update_inventory_from_protocol(
            get_world_service(),
            [0, 0, 5, 0, 10],
            [False, False, True, False, True],
        )
        mark_combat_hit(get_world_service(), weapon_byte=2)
        assert get_inventory_state(get_world_service())["missile_shots"]["count"] == 4

    def test_homing_hit_decrements_homing_count(self) -> None:
        """mark_combat_hit with weapon_byte=3 decrements homing_shots count."""
        update_inventory_from_protocol(
            get_world_service(),
            [0, 0, 0, 3, 10],
            [False, False, False, True, True],
        )
        mark_combat_hit(get_world_service(), weapon_byte=3)
        assert get_inventory_state(get_world_service())["homing_shots"]["count"] == 2

    def test_hit_decrement_does_not_go_below_zero(self) -> None:
        """mark_combat_hit does not decrement below zero."""
        update_inventory_from_protocol(
            get_world_service(),
            [0, 0, 0, 0, 10],
            [False, True, False, False, True],
        )
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert get_inventory_state(get_world_service())["dual_shots"]["count"] == 0

    def test_consecutive_hits_deplete_dual(self) -> None:
        """Multiple dual hits decrement count to zero progressively."""
        update_inventory_from_protocol(
            get_world_service(),
            [0, 3, 0, 0, 10],
            [False, True, False, False, True],
        )
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert get_inventory_state(get_world_service())["dual_shots"]["count"] == 2
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert get_inventory_state(get_world_service())["dual_shots"]["count"] == 1
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert get_inventory_state(get_world_service())["dual_shots"]["count"] == 0
        # Fourth hit: already at zero, stays at zero
        mark_combat_hit(get_world_service(), weapon_byte=1)
        assert get_inventory_state(get_world_service())["dual_shots"]["count"] == 0

    def test_reset_clears_our_shot_response(self) -> None:
        """reset_world_state clears the our_shot_response flag."""
        mark_combat_hit(get_world_service(), weapon_byte=0)
        reset_world_state()
        assert peek_our_shot_response(get_world_service()) is False


class TestTeleportLandedTracking:
    """Tests for mark_teleport_landed and check_and_clear_teleport_landed."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_check_returns_false_by_default(self) -> None:
        """check_and_clear_teleport_landed returns False with no teleport."""
        assert check_and_clear_teleport_landed(get_world_service()) is False

    def test_mark_and_check_returns_true(self) -> None:
        """mark_teleport_landed sets flag, check returns True then False."""
        mark_teleport_landed(get_world_service())
        assert check_and_clear_teleport_landed(get_world_service()) is True
        assert check_and_clear_teleport_landed(get_world_service()) is False

    def test_dispatch_teleport_landed_sets_flag(self) -> None:
        """Container message with msg_type=teleport_landed marks landing."""
        from tankpit_bot.container.types import TeleportLandedDict

        msg = TeleportLandedDict(msg_type="teleport_landed", subtype=0)
        dispatch_world_state_update(get_world_service(), msg)
        assert check_and_clear_teleport_landed(get_world_service()) is True

    def test_reset_clears_teleport_flag(self) -> None:
        """reset_world_state clears the teleport landed flag."""
        mark_teleport_landed(get_world_service())
        reset_world_state()
        assert check_and_clear_teleport_landed(get_world_service()) is False
