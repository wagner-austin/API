"""Conservative container evidence from an observed enemy refuel."""

from __future__ import annotations

from tankpit_bot.facts.provenance import make_provenance, make_source_ref
from tankpit_bot.facts.source import FactSource
from tankpit_bot.state.types import (
    ContainerStateDict,
    TankObservation,
    TankStateDict,
    WorldStateDict,
    coord_key,
)
from tankpit_bot.types.constants import DIRECTION_DEAD_THRESHOLD, TankLiveness

_POSITION_SOURCES = {
    FactSource.WIRE_0X3D_MOVEMENT,
    FactSource.WIRE_0X47_MOVEMENT,
}
_REFUEL_POSITION_MAX_AGE_MS = 2000


def containers_after_enemy_refuel(
    state: WorldStateDict,
    previous: TankStateDict | None,
    current: TankStateDict,
    obs: TankObservation,
) -> dict[str, ContainerStateDict]:
    """Downgrade fuel after a fresh movement/status pair suggests consumption.

    The quartile increase establishes neither an exact transfer nor an empty
    container. Keep its advertised volume and observation time. Require a
    preceding movement observation, so map/DOM coordinates and respawns cannot
    establish this inference. Repeated status messages do not compound it.

    Args:
        state: World before the observation.
        previous: Previously observed tank, if known.
        current: Tank after applying the observation.
        obs: Incoming tank observation.

    Returns:
        Container registry with a suspect belief, or the unchanged registry.
    """
    own = state["self_state"]
    if previous is None or own is None:
        return state["containers"]
    if current["is_self"] or current["team"] == own["team"]:
        return state["containers"]
    if (
        previous["liveness"] is not TankLiveness.ALIVE
        or current["liveness"] is not TankLiveness.ALIVE
        or previous["direction"] >= DIRECTION_DEAD_THRESHOLD
    ):
        return state["containers"]
    if current["rank"] != previous["rank"]:
        return state["containers"]
    if current["damage_state"] <= previous["damage_state"]:
        return state["containers"]
    if (
        not obs["is_wire_sourced"]
        or obs["fact_source"] is not FactSource.WIRE_0X2E_TANK_STATUS_SYNC
    ):
        return state["containers"]
    if previous["provenance"]["origin"] not in _POSITION_SOURCES:
        return state["containers"]
    age = obs["timestamp_ms"] - previous["last_position_update_ms"]
    if not 0 <= age <= _REFUEL_POSITION_MAX_AGE_MS:
        return state["containers"]
    key = coord_key(previous["x"], previous["y"])
    container = state["containers"].get(key)
    if container is None or not container["is_fuel"] or container["volume"] <= 0:
        return state["containers"]
    suspect = container.copy()
    suspect["confidence"] = min(container["confidence"], 0.5)
    suspect["provenance"] = make_provenance(
        FactSource.CLIENT_SIDE_INFERENCE,
        [
            make_source_ref(container["provenance"]["origin"], container["timestamp_ms"]),
            make_source_ref(previous["provenance"]["origin"], previous["timestamp_ms"]),
            make_source_ref(obs["fact_source"], obs["timestamp_ms"]),
        ],
    )
    containers = dict(state["containers"])
    containers[key] = suspect
    return containers
