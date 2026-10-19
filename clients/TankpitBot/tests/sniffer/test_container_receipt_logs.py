"""World cache receipts never claim that our tank collected the fuel."""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_container_pickup
from tests._runtime_logging_support import capture_runtime_events
from tests.bot.ai._support import make_container, make_world


def test_container_receipts_log_remaining_stock_without_crediting_own_fuel() -> None:
    """Partial and empty receipts log exact neutral wording and preserve fuel."""
    ws = WorldService()
    world, own = make_world(
        fuel=256,
        containers={"148,249": make_container(148, 249, 580, is_fuel=True)},
    )
    ws.world_state = world
    with capture_runtime_events() as records:
        update_world_state_from_container_pickup(ws, 148, 249, 200)
        assert ws.world_state["containers"]["148,249"]["volume"] == 200
        update_world_state_from_container_pickup(ws, 148, 249, 0)
    assert [record.getMessage() for record in records] == [
        "WORLD: Container updated at (148, 249) -- 200 fuel remains",
        "WORLD: Container cleared at (148, 249)",
    ]
    assert ws.world_state["self_state"] == own
    assert ws.world_state["containers"] == {}
