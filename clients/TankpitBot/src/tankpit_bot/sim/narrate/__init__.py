"""Wire narration: what ONE connection is told about a resolved action.

The simulator's emission side used to do two jobs in one function —
mutate the world through a law module, then append the messages the
single client should see. That is indistinguishable from correct while
exactly one connection exists, and wrong the moment a second appears:
calling such a function once per observer applies every mutation once
per observer, so a radar scan would cost N x 10 fuel and one mine press
would place N mines ([[physics-module-roadmap]]).

The two jobs are now separate. The law modules RESOLVE — they own every
effect an action has on the world, including the fuel it costs — and
return a typed outcome. The functions here NARRATE: pure, no mutation,
taking the resolved outcome plus the ``observer_id`` being narrated
for, returning the messages that connection receives. Which messages
those are is measured, not assumed ([[recipient-policy]]).
"""

from __future__ import annotations

from tankpit_bot.sim.narrate.combat import narrate_corpse_removals, narrate_shot
from tankpit_bot.sim.narrate.movement import (
    TELEPORT_LANDED_SUBTYPE,
    narrate_fuel_pickup,
    narrate_move,
    narrate_teleport,
    pickup_message,
)
from tankpit_bot.sim.narrate.resources import (
    narrate_equipment_pickup,
    narrate_equipment_toggle,
    narrate_fuel_deposit,
    narrate_mine_press,
    narrate_radar,
)
from tankpit_bot.sim.narrate.world import narrate_block_action, narrate_chat

__all__ = [
    "TELEPORT_LANDED_SUBTYPE",
    "narrate_block_action",
    "narrate_chat",
    "narrate_corpse_removals",
    "narrate_equipment_pickup",
    "narrate_equipment_toggle",
    "narrate_fuel_deposit",
    "narrate_fuel_pickup",
    "narrate_mine_press",
    "narrate_move",
    "narrate_radar",
    "narrate_shot",
    "narrate_teleport",
    "pickup_message",
]
