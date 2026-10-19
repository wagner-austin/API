"""Movement/status wire dispatch preserves the premises of fuel suspicion."""

from __future__ import annotations

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.protocol import MovementResponseDict, TankStatusSyncDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tests.bot.ai._support import make_container, make_world


@pytest.mark.parametrize("rank,direction,confidence", [(6, 0, 0.5), (7, 0, 1.0), (6, 32, 1.0)])
def test_wire_refuel_pair_distinguishes_unchanged_rank_from_promotion(
    rank: int,
    direction: int,
    confidence: float,
) -> None:
    """The real dispatcher carries both fuel tier and rank into the inference."""
    _test_hooks.get_current_time_ms = lambda: 100000
    ws = WorldService()
    world, own = make_world(
        fuel=740,
        containers={"148,249": make_container(148, 249, 580, is_fuel=True)},
    )
    ws.world_state = world
    dispatch_world_state_update(
        ws,
        MovementResponseDict(
            msg_type=0x3D,
            team=2,
            tank_id=814,
            x=148,
            y=249,
            direction=direction,
            damage_state=2,
            rank=6,
            lb_score=0,
            carrying=0,
        ),
    )
    assert ws.world_state["tanks"]["814"]["damage_state"] == 2
    dispatch_world_state_update(
        ws,
        TankStatusSyncDict(
            msg_type=0x2E,
            subtype=0,
            tank_id=814,
            damage_state=3,
            rank=rank,
            lb_score=0,
            promo_state=0,
            promo_bar_lit=None,
            fuel=None,
        ),
    )
    assert ws.world_state["tanks"]["814"]["rank"] == rank
    assert ws.world_state["containers"]["148,249"]["confidence"] == confidence
    assert ws.world_state["containers"]["148,249"]["volume"] == 580
    assert ws.world_state["self_state"] == own
