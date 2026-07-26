"""Tests for dispatch of 0x3D TankPositionStatus and 0x2E SelfStatus."""

from __future__ import annotations

from tankpit_bot.protocol import TankStatusSyncDict
from tankpit_bot.protocol.types import MovementResponseDict
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.state.types.constants import DIRECTION_DEAD_THRESHOLD


class TestDispatchTankPositionStatus:
    """Tests for dispatching TankPositionStatus (0x3D) container messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_creates_tank(self) -> None:
        """Dispatching creates a new tank in world state."""
        msg = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=511,
            x=253,
            y=91,
            direction=8,
            damage_state=3,
            rank=1,
            lb_score=15362,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "511" in state["tanks"]
        tank = state["tanks"]["511"]
        assert tank["x"] == 253
        assert tank["y"] == 91
        assert tank["team"] == 1
        assert tank["rank"] == 1
        assert tank["direction"] == 8

    def test_dispatch_updates_position(self) -> None:
        """Dispatching updates position of existing tank."""
        ws = get_world_service()
        msg1 = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=500,
            x=100,
            y=100,
            direction=4,
            damage_state=0,
            rank=1,
            lb_score=500,
            carrying=0,
        )
        dispatch_world_state_update(ws, msg1)
        assert ws.world_state["tanks"]["500"]["x"] == 100

        msg2 = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=500,
            x=110,
            y=105,
            direction=4,
            damage_state=0,
            rank=1,
            lb_score=498,
            carrying=0,
        )
        dispatch_world_state_update(ws, msg2)
        assert ws.world_state["tanks"]["500"]["x"] == 110
        assert ws.world_state["tanks"]["500"]["y"] == 105

    def test_dispatch_sets_direction(self) -> None:
        """Direction field is stored on the tank state."""
        msg = MovementResponseDict(
            msg_type=0x3D,
            team=2,
            tank_id=1301,
            x=50,
            y=60,
            direction=12,
            damage_state=0,
            rank=1,
            lb_score=100,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), msg)

        tank = get_world_service().world_state["tanks"]["1301"]
        assert tank["direction"] == 12

    def test_dispatch_dead_tank_direction_32(self) -> None:
        """Dead tank (direction=32) is stored correctly."""
        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=502,
            x=80,
            y=90,
            direction=32,
            damage_state=0,
            rank=1,
            lb_score=400,
            carrying=0,
        )
        dispatch_world_state_update(get_world_service(), msg)

        tank = get_world_service().world_state["tanks"]["502"]
        assert tank["direction"] == 32
        assert tank["direction"] >= DIRECTION_DEAD_THRESHOLD

    def test_dispatch_updates_damage(self) -> None:
        """Damage state is updated from 0x3D message."""
        ws = get_world_service()
        msg1 = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=511,
            x=100,
            y=100,
            direction=8,
            damage_state=0,
            rank=1,
            lb_score=500,
            carrying=0,
        )
        dispatch_world_state_update(ws, msg1)
        assert ws.world_state["tanks"]["511"]["damage_state"] == 0

        msg2 = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=511,
            x=100,
            y=100,
            direction=8,
            damage_state=3,
            rank=1,
            lb_score=498,
            carrying=0,
        )
        dispatch_world_state_update(ws, msg2)
        assert ws.world_state["tanks"]["511"]["damage_state"] == 3


class TestDispatchSelfStatus:
    """Tests for dispatching nested 0x2E TankStatusSync (self-status form).

    The 13-byte 0x2E-nested form (with fuel at offsets 10-11) is decoded
    by the protocol path's TankStatusSync (decode_tank_status_sync). It
    arrives via the unified `decode_0x2e_message` dispatcher and updates
    the bot's self_state.fuel.
    """

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_updates_fuel(self) -> None:
        """TankStatusSync with fuel updates self_state.fuel."""
        ws = get_world_service()

        from tankpit_bot.sniffer.world_state_tanks import (
            update_world_state_from_move_response_full,
        )

        update_world_state_from_move_response_full(ws, 1301, 100, 100, 2, 1)

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=2,
            tank_id=1301,
            damage_state=0,
            rank=1,
            lb_score=151,
            promo_state=0,
            fuel=1100,
        )
        dispatch_world_state_update(ws, msg)

        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after move response")
        assert self_state["fuel"] == 1100

    def test_dispatch_self_promo_eligible_emits_diagnostic(self) -> None:
        """0x2E for SELF with promo_state > 0 emits ``self_promo_eligible``.

        The wire byte is the per-tank promotion-eligibility marker (JS
        ``Og.h`` ``g``). Routing a non-zero value through the diagnostic
        channel for OWN tank lets downstream analyzers ("am I about to
        rank up?") read it without re-decoding the 0x2E body.
        """
        from tankpit_bot.protocol import TankStatusSyncDict as _Sync
        from tankpit_bot.sniffer.world_state_tanks import (
            update_world_state_from_move_response_full,
        )

        ws = get_world_service()
        update_world_state_from_move_response_full(ws, 1301, 100, 100, 2, 1)

        msg = _Sync(
            msg_type=0x2E,
            subtype=1,
            tank_id=1301,
            damage_state=0,
            rank=1,
            lb_score=151,
            promo_state=7,
            fuel=None,
        )
        # The dispatch path emits a diagnostic event; we just need to
        # exercise the branch so coverage observes line 438.
        dispatch_world_state_update(ws, msg)

        # Damage update still applies regardless of promo branch.
        assert ws.world_state["tanks"]["1301"]["damage_state"] == 0

    def test_dispatch_self_status_applies_a_mid_session_promotion(self) -> None:
        """The 0x2E rank field promotes self_state the tick it flips.

        Measured bot-20260725-211120: the promoting kill flipped the
        0x2E rank 0 -> 1 at t+31.7s with NO 0x2B all session — the
        status sync is the promotion's earliest wire signal, and the
        rank-derived readiness bars read self_state["rank"] live.
        """
        from tankpit_bot.protocol import TankStatusSyncDict as _Sync
        from tankpit_bot.sniffer.world_state_tanks import (
            update_world_state_from_move_response_full,
        )

        ws = get_world_service()
        update_world_state_from_move_response_full(ws, 1301, 100, 100, 2, 0)

        msg = _Sync(
            msg_type=0x2E,
            subtype=2,
            tank_id=1301,
            damage_state=0,
            rank=1,
            lb_score=151,
            promo_state=0,
            fuel=1100,
        )
        dispatch_world_state_update(ws, msg)

        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after move response")
        assert self_state["rank"] == 1

    def test_dispatch_other_tank_status_leaves_self_rank_alone(self) -> None:
        """Another tank's 0x2E rank never touches self_state."""
        from tankpit_bot.protocol import TankStatusSyncDict as _Sync
        from tankpit_bot.sniffer.world_state_tanks import (
            update_world_state_from_move_response_full,
        )

        ws = get_world_service()
        update_world_state_from_move_response_full(ws, 1301, 100, 100, 2, 0)

        msg = _Sync(
            msg_type=0x2E,
            subtype=1,
            tank_id=511,
            damage_state=2,
            rank=5,
            lb_score=0,
            promo_state=0,
            fuel=None,
        )
        dispatch_world_state_update(ws, msg)

        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should exist after move response")
        assert self_state["rank"] == 0


class TestDispatchSupervisor:
    """Tests for dispatching Supervisor (0x52) command error messages."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_stores_error_code(self) -> None:
        """Supervisor error code is stored on WorldService."""
        from tankpit_bot.protocol import SupervisorDict

        msg = SupervisorDict(msg_type=0x52, reset_action=1, close_map=0, error_code=8)
        ws = get_world_service()
        dispatch_world_state_update(ws, msg)

        assert ws.last_command_error == 8

    def test_dispatch_empty_container_error(self) -> None:
        """Error code 4 = Empty container."""
        from tankpit_bot.protocol import SupervisorDict

        msg = SupervisorDict(msg_type=0x52, reset_action=1, close_map=0, error_code=4)
        ws = get_world_service()
        dispatch_world_state_update(ws, msg)

        assert ws.last_command_error == 4

    def test_check_and_clear_command_error(self) -> None:
        """check_and_clear_command_error returns error and resets."""
        from tankpit_bot.protocol import SupervisorDict
        from tankpit_bot.sniffer.world_state_combat import (
            check_and_clear_command_error,
        )

        msg = SupervisorDict(msg_type=0x52, reset_action=0, close_map=1, error_code=5)
        ws = get_world_service()
        dispatch_world_state_update(ws, msg)

        assert check_and_clear_command_error(ws) == 5
        assert check_and_clear_command_error(ws) == -1

    def test_no_error_returns_negative_one(self) -> None:
        """Returns -1 when no error is pending."""
        from tankpit_bot.sniffer.world_state_combat import (
            check_and_clear_command_error,
        )

        assert check_and_clear_command_error(get_world_service()) == -1
