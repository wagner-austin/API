"""Tilting the mix toward the visible threat, exercised as the pure function it is.

No socket and no game: hostiles and a composition go in, a composition comes
out. The rule under test is the share arithmetic and the three cases that fall
out of "only repeat what was asked for" -- nothing visible flying, no anti-air
to repeat, and everything visible flying.
"""

from __future__ import annotations

from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.counter import counter_composition, fleet_types
from rw_bot.policy.doctrine import NAVTILT_ALWAYS, NAVTILT_BLOODIED
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
    assert counter_composition(mix, (), _PROFILES, 0, False) == mix


def test_a_ground_threat_leaves_the_mix_alone() -> None:
    mix = ("c_tank", "c_tank", "c_aa")
    assert counter_composition(mix, (_ground(1), _ground(2)), _PROFILES, 0, False) == mix


def test_a_mix_without_anti_air_is_not_given_any() -> None:
    """Which unit answers air is the doctrine's question, not this function's.

    The gap stays visible where it already shows -- the report's engageable
    count ([[mechanics-combat-profile]]).
    """
    mix = ("c_tank", "c_tank")
    assert counter_composition(mix, (_heli(1),), _PROFILES, 0, False) == mix


def test_anti_air_is_repeated_until_its_share_covers_the_air_share() -> None:
    """Half the visible threat flies, so half the mix must reach the air."""
    tilted = counter_composition(
        ("c_tank", "c_tank", "c_tank", "c_aa"),
        (_heli(1), _heli(2), _ground(3), _ground(4)),
        _PROFILES,
        0,
        False,
    )
    airworthy = sum(1 for name in tilted if _PROFILES[name]["hits_air"])
    assert tilted[:4] == ("c_tank", "c_tank", "c_tank", "c_aa")
    assert airworthy / len(tilted) >= 0.5


def test_a_mix_already_covering_the_share_is_left_alone() -> None:
    mix = ("c_aa", "c_aa", "c_tank")
    assert counter_composition(mix, (_heli(1), _ground(2)), _PROFILES, 0, False) == mix


def test_an_all_air_threat_drops_the_armed_ground_only_types() -> None:
    """Producing a unit that can reach nothing in sight is producing a loss.

    The builder stays: it is in the mix for the economy, not the fight.
    """
    tilted = counter_composition(
        ("builder", "c_tank", "c_tank", "c_aa"),
        (_heli(1), _heli(2)),
        _PROFILES,
        0,
        False,
    )
    assert tilted == ("builder", "c_aa")


def test_two_anti_air_types_are_repeated_in_their_stated_ratio() -> None:
    """The tilt cycles the capable types rather than multiplying one of them."""
    tilted = counter_composition(
        ("c_tank", "c_tank", "c_tank", "c_tank", "c_aa", "c_missile"),
        (_heli(1), _heli(2), _heli(3), _ground(4)),
        _PROFILES,
        0,
        False,
    )
    added = tilted[6:]
    assert set(added) <= {"c_aa", "c_missile"}
    aa = sum(1 for name in added if name == "c_aa")
    missile = sum(1 for name in added if name == "c_missile")
    assert abs(aa - missile) <= 1


def _ship(unit_id: int) -> Entity:
    """A surface warship: WATER layer, guns that reach 240 world units."""
    return entity(unit_id, "warship", mine=False, hostile=True, movement="WATER")


_NAVAL_PROFILES: dict[str, CombatProfile] = {
    **_PROFILES,
    "warship": profile("warship", 240.0),
    # Outranges the warship from the shore, which is the naval answer the
    # profiles hand over mechanically ([[policy-exact-timing]]).
    "c_artillery": profile("c_artillery", 290.0),
}


def test_a_fleet_repeats_the_type_that_outranges_it() -> None:
    """Half the visible threat floats, so half the mix must outgun the fleet."""
    tilted = counter_composition(
        ("c_tank", "c_tank", "c_tank", "c_artillery"),
        (_ship(1), _ship(2), _ground(3), _ground(4)),
        _NAVAL_PROFILES,
        1,
        False,
    )
    assert tilted[:4] == ("c_tank", "c_tank", "c_tank", "c_artillery")
    outgunning = sum(1 for name in tilted if name == "c_artillery")
    assert outgunning / len(tilted) >= 0.5


def test_a_mix_nothing_in_which_outranges_the_fleet_is_left_alone() -> None:
    """Which unit outranges a fleet is the doctrine's question, not this one's."""
    mix = ("c_tank", "c_tank", "c_aa")
    assert counter_composition(mix, (_ship(1), _ground(2)), _NAVAL_PROFILES, 1, False) == mix


def test_an_all_naval_picture_drops_the_outgunned() -> None:
    """Producing a unit that fights inside the fleet's guns is producing a loss.

    The builder stays: it is in the mix for the economy, not the fight.
    """
    tilted = counter_composition(
        ("builder", "c_tank", "c_artillery"),
        (_ship(1), _ship(2)),
        _NAVAL_PROFILES,
        1,
        False,
    )
    assert tilted == ("builder", "c_artillery")


def test_a_mix_already_outgunning_the_naval_share_is_left_alone() -> None:
    mix = ("c_artillery", "c_artillery", "c_tank")
    assert counter_composition(mix, (_ship(1), _ground(2)), _NAVAL_PROFILES, 1, False) == mix


def test_the_air_and_naval_tilts_compose_on_one_picture() -> None:
    """A helicopter and a warship in sight each pull their own answer."""
    tilted = counter_composition(
        ("c_tank", "c_tank", "c_aa", "c_artillery"),
        (_heli(1), _ship(2), _ground(3), _ground(4)),
        _NAVAL_PROFILES,
        1,
        False,
    )
    assert tilted[:4] == ("c_tank", "c_tank", "c_aa", "c_artillery")
    assert sum(1 for name in tilted if name == "c_aa") >= 1
    assert sum(1 for name in tilted if name == "c_artillery") >= 1


def test_the_naval_clause_stays_silent_when_the_doctrine_says_off() -> None:
    """The control arm's whole meaning: same code, same fleet, no tilt."""
    mix = ("c_tank", "c_tank", "c_artillery")
    assert counter_composition(mix, (_ship(1), _ground(2)), _NAVAL_PROFILES, 0, False) == mix


def test_an_armed_naval_clause_with_no_fleet_in_sight_changes_nothing() -> None:
    """The tilt spends nothing until a fleet is actually seen."""
    mix = ("c_tank", "c_tank", "c_artillery")
    assert counter_composition(mix, (_ground(1), _ground(2)), _NAVAL_PROFILES, 1, False) == mix


def test_the_bloodied_clause_fires_only_after_the_fleet_kills() -> None:
    """Two panels' calibration as code: the ungated tilt re-rolled winning
    seeds and the deficit gate still fired inside winning games, so
    NAVTILT_BLOODIED waits for the failure mode itself -- WATER-movers
    killing units of ours -- and a game the fleet never touched can never
    be perturbed."""
    mix = ("c_tank", "c_tank", "c_tank", "c_artillery")
    picture = (_ship(1), _ship(2), _ground(3), _ground(4))
    unbloodied = counter_composition(mix, picture, _NAVAL_PROFILES, NAVTILT_BLOODIED, False)
    assert unbloodied == mix
    bloodied = counter_composition(mix, picture, _NAVAL_PROFILES, NAVTILT_BLOODIED, True)
    assert len(bloodied) > len(mix)
    assert set(bloodied[len(mix) :]) == {"c_artillery"}
    always = counter_composition(mix, picture, _NAVAL_PROFILES, NAVTILT_ALWAYS, False)
    assert always == bloodied


def test_the_bloodied_gate_never_touches_the_air_clause() -> None:
    """The air tilt predates the gate and stays unconditional: anti-air
    was never measured to re-roll wins, and the gate guards only the
    clause whose panels convicted it."""
    mix = ("c_tank", "c_tank", "c_tank", "c_aa")
    picture = (_heli(1), _heli(2), _ground(3), _ground(4))
    tilted = counter_composition(mix, picture, _PROFILES, NAVTILT_BLOODIED, False)
    airworthy = sum(1 for name in tilted if _PROFILES[name]["hits_air"])
    assert airworthy / len(tilted) >= 0.5


def test_fleet_types_names_the_water_movers_once_each() -> None:
    """The gate's memory feed: naval names in first-seen order, repeats
    collapsed, everything else left out."""
    picture = (_ship(1), _ground(2), _ship(3), _heli(4))
    assert fleet_types(picture) == ("warship",)
    assert fleet_types((_ground(1), _heli(2))) == ()
