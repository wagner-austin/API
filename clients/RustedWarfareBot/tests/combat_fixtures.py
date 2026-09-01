"""The skirmish the fighting policies are argued against, shared by both.

Choosing a target and releasing a wave are separate decisions over one roster
(`test_policy_combat`, `test_policy_muster`). They must agree about what a tank
costs and how far it shoots, so the catalogue and the profiles derived from it
live here in one copy.

These are constructors over :mod:`tests.wire_fixtures`, not mocks: what comes
out is the same TypedDict the decoder produces from live bytes.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import entity, profiles_for


def weapon(damage: float = 17.0, reach: float = 110.0) -> Weapon:
    """Build a weapon that fires one shot per volley."""
    return Weapon(
        shoot_delay=50.0,
        attack_range=reach,
        direct_damage=damage,
        direct_damage_volley=damage,
        area_damage=0.0,
        area_damage_volley=0.0,
    )


def unit_stats(type_name: str, *, speed: float = 1.0, armed: bool = True) -> UnitStats:
    """Build one catalogue entry, armed and mobile unless said otherwise."""
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=350,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=weapon() if armed else None,
    )


CATALOGUE = {
    "c_tank": unit_stats("c_tank"),
    "builder": unit_stats("builder", speed=0.6, armed=False),
    "commandCenter": unit_stats("commandCenter", speed=0.0, armed=False),
    "c_turret_t1": unit_stats("c_turret_t1", speed=0.0),
    "editorOrBuilder": unit_stats("editorOrBuilder", speed=0.0, armed=False),
}

#: Combat profiles derived from the catalogue above, so a unit cannot be armed
#: in one table and unarmed in the other. Ground-only, like every land unit the
#: base game lets a player build; the layer tests override what they need.
PROFILES = profiles_for(CATALOGUE)


def unit(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
    hostile: bool = False,
    complete: bool = True,
) -> Entity:
    """Build an entity record, owned and finished unless said otherwise."""
    return entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=hostile,
        movement="LAND",
        group=1,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=0,
    )


def sample(*entities: Entity) -> Sample:
    """Build one observation from a roster, with nothing else in the world."""
    return Sample(
        frame=1,
        clock_ms=10,
        credits=4000,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=tuple(entities),
        pools=(),
        players=(),
        options=(),
        refusals=(),
    )
