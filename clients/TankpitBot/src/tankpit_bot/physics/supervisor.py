"""Supervisor (0x52) refusal laws — one implementation, two consumers.

The server answers a command it will not honor with a 0x52 carrying
an error code (``SUPERVISOR_ERROR_*`` in ``protocol/constants.py``,
mined from the tpclient.js ``Gb[]`` string table). Several of those
refusals are pure functions of state the client already knows — fuel
versus rank capacity, slot counts versus rank capacity, teleport cost
versus fuel — so both sides of the codebase need the same law:

* the **sim server** consumes these predicates to EMIT the refusal
  (``sim/emissions.py``, ``sim/equipment.py``, ``sim/actions.py``);
* the **production bot** consumes them to PREDICT the refusal before
  dispatch (``bot/executor.py``) and to steer target choice
  (``bot/ai/combat_strategy.py``).

History: until 2026-08-03 the laws lived inline at the sim emission
sites and the bot had no predictor at all — the 20-kill soak
bot-20260802-205105 dispatched 48 fuel pickups at exactly-full fuel,
every one refused code 5 by the live server ([[fuel-system]],
[[capture-differ]]). The bot's belief proved each refusal before the
send; nothing consulted it.

Prediction discipline: a predicate refuses ONLY what the caller's
belief PROVES. Races the client cannot see (a container drained by
another tank between scan and pickup) must stay optimistic — dispatch
and learn is the correct behavior there, and the byte-mined
choreography prices such a refusal at one walk.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_EMPTY_CONTAINER,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    SUPERVISOR_ERROR_INVENTORY_FULL,
    SUPERVISOR_ERROR_TANK_FULL,
)

TELEPORT_RING1_COST_SLACK = 9
"""Cost slack covering ring-1 landing displacement when predicting a
teleport refusal from the REQUESTED target: the server lands within
chebyshev 1 of the click (E -> N -> W -> S displacement law), so the
cheapest legal landing costs at least
``floor(6 * (euclid - sqrt(2))) >= floor(6 * euclid) - 9``. A
dispatch-side prediction is therefore only CERTAIN when
``teleport_refusal(fuel, cost_to_target - TELEPORT_RING1_COST_SLACK)``
still refuses — the composition ``bot/executor.py`` applies. The live
differ prices this lane at 0.3% of 7,364 teleports answering code 8."""


def fuel_pickup_close_code(remaining: int) -> int:
    """Return the 0x52 code closing a fuel-pickup choreography.

    The byte-mined close-by-stockedness law (2026-08-01, ~1,600
    archive windows, [[fuel-system]]): a fuel pickup's closing 0x52
    reads code 5 ("Tank full") while the container keeps ANY fuel and
    code 4 ("Empty container") once it is drained — code 5 doubles as
    the clamp SUCCESS receipt, not only a refusal.

    Args:
        remaining: The container's volume after the transfer.

    Returns:
        ``SUPERVISOR_ERROR_TANK_FULL`` (5) while stocked, else
        ``SUPERVISOR_ERROR_EMPTY_CONTAINER`` (4).
    """
    if remaining > 0:
        return SUPERVISOR_ERROR_TANK_FULL
    return SUPERVISOR_ERROR_EMPTY_CONTAINER


def fuel_pickup_refusal(fuel: int, rank: int, container_volume: int) -> int | None:
    """Predict the server's refusal of a fuel pickup, if provable.

    Two locally-provable outcomes of the measured choreography:

    * a KNOWN-drained container transfers nothing and closes code 4
      (the walk still executes on the real server — the prediction
      spares the whole trip);
    * a tank at rank capacity transfers nothing from a stocked
      container and closes code 5 (the no-transfer branches of
      [[fuel-system]]; 48 live receipts in bot-20260802-205105).

    A stocked container with fuel headroom transfers — no refusal.
    A refusal IS a predicted no-transfer: its code is
    :func:`fuel_pickup_close_code` applied to the container, which is
    why a full tank at a drained container answers 4, not 5.

    Args:
        fuel: The tank's believed fuel.
        rank: The tank's true rank (0-8).
        container_volume: The container's believed volume.

    Returns:
        The predicted 0x52 code, or None when the pickup transfers.
    """
    if container_volume <= 0 or fuel >= fuel_capacity(rank):
        return fuel_pickup_close_code(container_volume)
    return None


def equipment_pickup_refusal(counts: list[int], rank: int) -> int | None:
    """Predict the server's refusal of an equipment pickup, if provable.

    The archive law (2026-07-22, 1,149 grants): every successful
    pickup grants exactly one DEFICIENT slot; with all five slots at
    the rank cap the server refuses code 7 and the container stays.
    Any single deficient slot makes the pickup grantable (slot choice
    is the server's).

    Args:
        counts: The five believed slot counts (armor, dual, missile,
            homing, radar).
        rank: The tank's true rank (0-8).

    Returns:
        ``SUPERVISOR_ERROR_INVENTORY_FULL`` (7) when every slot is at
        the rank cap, else None.
    """
    cap = inventory_capacity(rank)
    if all(count >= cap for count in counts):
        return SUPERVISOR_ERROR_INVENTORY_FULL
    return None


def teleport_refusal(fuel: int, cost: int) -> int | None:
    """The teleport affordability law at a RESOLVED landing tile.

    The sim router charges ``floor(6 x euclid)`` to the actual landing
    and refuses code 8 when the charge exceeds fuel (equal spends the
    tank dry and lands). Consumed by ``sim/actions.py`` at resolution
    time.

    Args:
        fuel: The tank's fuel.
        cost: The teleport cost to the resolved landing tile.

    Returns:
        ``SUPERVISOR_ERROR_INSUFFICIENT_FUEL`` (8) when the cost
        exceeds fuel, else None.
    """
    if cost > fuel:
        return SUPERVISOR_ERROR_INSUFFICIENT_FUEL
    return None


__all__ = [
    "TELEPORT_RING1_COST_SLACK",
    "equipment_pickup_refusal",
    "fuel_pickup_close_code",
    "fuel_pickup_refusal",
    "teleport_refusal",
]
