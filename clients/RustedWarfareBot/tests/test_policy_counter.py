"""Tilting the mix toward the visible threat, exercised as the pure function it is.

No socket and no game: hostiles and a composition go in, a composition comes
out. The rule under test is the share arithmetic and the three cases that fall
out of "only repeat what was asked for" -- nothing visible flying, no anti-air
to repeat, and everything visible flying.
"""

from __future__ import annotations

from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.counter import counter_composition
from rw_bot.wire.state import Entity
from tests.wire_fixtures import entity, profile

_PROFILES: dict[str, CombatProfile] = {
    "c_tank": profile("c_tank", 110.0),
    "c_aa": profile("c_aa", 120.0, air=True),
    "c_missile": profile("c_missile", 200.0, air=True),
    # Unarmed, which is what a builder looks like to the registry.
    "builder": profile("builder", 0.0, land=False),
}


def _heli(unit_id: int) -> Entity:
    return entity(unit_id, "heli", mine=False, hostile=True, flying=True)


def _ground(unit_id: int) -> Entity:
    return entity(unit_id, "enemy_tank", mine=False, hostile=True)


def test_nothing_visible_leaves_the_mix_alone() -> None:
    """Fog is not evidence, in either direction."""
    mix = ("c_tank", "c_tank", "c_aa")
    assert counter_composition(mix, (), _PROFILES) == mix


def test_a_ground_threat_leaves_the_mix_alone() -> None:
    mix = ("c_tank", "c_tank", "c_aa")
    assert counter_composition(mix, (_ground(1), _ground(2)), _PROFILES) == mix


def test_a_mix_without_anti_air_is_not_given_any() -> None:
    """Which unit answers air is the doctrine's question, not this function's.

    The gap stays visible where it already shows -- the report's engageable
    count ([[mechanics-combat-profile]]).
    """
    mix = ("c_tank", "c_tank")
    assert counter_composition(mix, (_heli(1),), _PROFILES) == mix


def test_anti_air_is_repeated_until_its_share_covers_the_air_share() -> None:
    """Half the visible threat flies, so half the mix must reach the air."""
    tilted = counter_composition(
        ("c_tank", "c_tank", "c_tank", "c_aa"),
        (_heli(1), _heli(2), _ground(3), _ground(4)),
        _PROFILES,
    )
    airworthy = sum(1 for name in tilted if _PROFILES[name]["hits_air"])
    assert tilted[:4] == ("c_tank", "c_tank", "c_tank", "c_aa")
    assert airworthy / len(tilted) >= 0.5


def test_a_mix_already_covering_the_share_is_left_alone() -> None:
    mix = ("c_aa", "c_aa", "c_tank")
    assert counter_composition(mix, (_heli(1), _ground(2)), _PROFILES) == mix


def test_an_all_air_threat_drops_the_armed_ground_only_types() -> None:
    """Producing a unit that can reach nothing in sight is producing a loss.

    The builder stays: it is in the mix for the economy, not the fight.
    """
    tilted = counter_composition(
        ("builder", "c_tank", "c_tank", "c_aa"),
        (_heli(1), _heli(2)),
        _PROFILES,
    )
    assert tilted == ("builder", "c_aa")


def test_two_anti_air_types_are_repeated_in_their_stated_ratio() -> None:
    """The tilt cycles the capable types rather than multiplying one of them."""
    tilted = counter_composition(
        ("c_tank", "c_tank", "c_tank", "c_tank", "c_aa", "c_missile"),
        (_heli(1), _heli(2), _heli(3), _ground(4)),
        _PROFILES,
    )
    added = tilted[6:]
    assert set(added) <= {"c_aa", "c_missile"}
    aa = sum(1 for name in added if name == "c_aa")
    missile = sum(1 for name in added if name == "c_missile")
    assert abs(aa - missile) <= 1
