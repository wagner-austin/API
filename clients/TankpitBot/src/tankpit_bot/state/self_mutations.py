"""Mutations to the bot's own state.

Position, rank, and fuel for the player tank. One of four
subject-scoped mutation modules alongside
:mod:`tankpit_bot.state.tank_mutations`,
:mod:`tankpit_bot.state.terrain_mutations`, and
:mod:`tankpit_bot.state.container_mutations`.
"""

from __future__ import annotations

from tankpit_bot.facts.provenance import make_provenance
from tankpit_bot.facts.source import FactSource
from tankpit_bot.state.types import (
    WorldStateDict,
    make_self_state,
)


def update_self_from_movement_response(
    state: WorldStateDict,
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    leaderboard_position: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Update self state from MovementResponse message.

    Args:
        state: Current world state.
        tank_id: Player's tank ID.
        x: New X coordinate.
        y: New Y coordinate.
        team: Team ID.
        rank: Military rank.
        leaderboard_position: Leaderboard position.
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated self state.
    """
    new_self = make_self_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        fuel=state["self_state"]["fuel"] if state["self_state"] else 0,
        leaderboard_position=leaderboard_position,
        observed_ms=timestamp_ms,
        provenance=make_provenance("wire_0x3D_movement", []),
    )
    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def update_self_position(
    state: WorldStateDict,
    x: int,
    y: int,
    timestamp_ms: int,
    fact_source: FactSource = "wire_0x3D_movement",
) -> WorldStateDict:
    """Update self position without changing viewport bounds.

    Updates the self_state x,y coordinates and preserves the current viewport.
    Creates a minimal self_state if none exists.

    Args:
        state: Current world state.
        x: New X coordinate.
        y: New Y coordinate.
        timestamp_ms: Message timestamp.
        fact_source: Wire channel the position arrived on. Defaults to
            the canonical 0x3D self-position channel; the 0x47 waypoint
            path passes its own.

    Returns:
        New WorldStateDict with updated position.
    """
    # Get or create self_state with updated position
    if state["self_state"] is not None:
        new_self = make_self_state(
            tank_id=state["self_state"]["tank_id"],
            x=x,
            y=y,
            team=state["self_state"]["team"],
            rank=state["self_state"]["rank"],
            fuel=state["self_state"]["fuel"],
            leaderboard_position=state["self_state"]["leaderboard_position"],
            observed_ms=timestamp_ms,
            provenance=make_provenance(fact_source, []),
        )
    else:
        # Create minimal self_state when we first learn position
        new_self = make_self_state(
            tank_id=0,
            x=x,
            y=y,
            team=0,
            rank=0,
            fuel=0,
            leaderboard_position=0,
            observed_ms=timestamp_ms,
            provenance=make_provenance(fact_source, []),
        )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def update_self_rank(
    state: WorldStateDict,
    rank: int,
    timestamp_ms: int,
    fact_source: FactSource,
) -> WorldStateDict:
    """Apply a wire-observed self rank, preserving everything else.

    A mid-session promotion flips the rank field of the self-addressed
    0x2E/0x47/0x3D statements the same tick as the promoting kill
    (measured bot-20260725-211120: 0x2E rank 0 -> 1 at t+31.7s, the
    kill tick), and every rank-derived readiness bar and capacity
    reads ``self_state["rank"]`` — dropping the update left the bot
    believing its old caps for the rest of that session.

    Args:
        state: Current world state.
        rank: Wire-observed rank of the self tank.
        timestamp_ms: Message timestamp.
        fact_source: Wire channel the rank arrived on.

    Returns:
        New WorldStateDict with the rank applied, or ``state``
        unchanged when there is no self state yet or the rank already
        matches.
    """
    current = state["self_state"]
    if current is None or current["rank"] == rank:
        return state
    new_self = make_self_state(
        tank_id=current["tank_id"],
        x=current["x"],
        y=current["y"],
        team=current["team"],
        rank=rank,
        fuel=current["fuel"],
        leaderboard_position=current["leaderboard_position"],
        observed_ms=timestamp_ms,
        provenance=make_provenance(fact_source, []),
    )
    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def set_self_fuel(
    state: WorldStateDict,
    fuel: int,
    timestamp_ms: int,
    fact_source: FactSource = "wire_0x2E_tank_status_sync",
) -> WorldStateDict:
    """Set self fuel to absolute value (from inventory or sync messages).

    Args:
        state: Current world state.
        fuel: Absolute fuel value.
        timestamp_ms: Message timestamp.
        fact_source: Wire channel the fuel total arrived on (0x2E sync,
            0x44 fuel gain, or 0x64 fuel total).

    Returns:
        New WorldStateDict with updated fuel, or unchanged if no self_state.
    """
    if state["self_state"] is None:
        return state

    new_self = make_self_state(
        tank_id=state["self_state"]["tank_id"],
        x=state["self_state"]["x"],
        y=state["self_state"]["y"],
        team=state["self_state"]["team"],
        rank=state["self_state"]["rank"],
        fuel=max(0, fuel),
        leaderboard_position=state["self_state"]["leaderboard_position"],
        observed_ms=timestamp_ms,
        provenance=make_provenance(fact_source, []),
    )

    return WorldStateDict(
        self_state=new_self,
        tanks=state["tanks"],
        containers=state["containers"],
        mines=state["mines"],
        terrain=state["terrain"],
        viewport=state["viewport"],
        scanned_tiles=state["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def set_self_rank(
    state: WorldStateDict,
    rank: int,
    timestamp_ms: int,
) -> WorldStateDict:
    """Set self rank to absolute value from a Promotion message (0x2B Rf).

    Delegates to :func:`update_self_rank` with the 0x2B provenance —
    the 0x2B banner is one of FOUR self-rank channels; the
    0x2E/0x3D/0x47 rank fields flip at the promoting kill and can
    arrive first (bot-20260725-211120 carried no 0x2B at all).

    Args:
        state: Current world state.
        rank: New rank index (0-8 per the JS rank table).
        timestamp_ms: Message timestamp.

    Returns:
        New WorldStateDict with updated self rank, or unchanged if no
        self_state has been established yet (rank can't precede join)
        or the rank already matches.
    """
    return update_self_rank(state, rank, timestamp_ms, "wire_0x2B_promotion")


__all__ = [
    "set_self_fuel",
    "set_self_rank",
    "update_self_from_movement_response",
    "update_self_position",
    "update_self_rank",
]
