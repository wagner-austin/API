"""Wave memory between observations, exercised without a match running.

:class:`~rw_bot.policy.dispatch.WaveController` is the state six loop locals
used to be, so what is tested here is exactly what could only be exercised by
playing a whole match before: that gathering happens below the wave size, that
release converts the reserve to attacks, and that neither a rally nor an attack
is ever re-sent to a unit already carrying it -- the engine runs a waypoint
until it is replaced, so a repeat resets the walk ([[issuing-orders]]).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.combat import FIRST_WAVE, Engagement
from rw_bot.policy.dispatch import WaveController, dispatch_attacks, gather_reserve
from rw_bot.wire.command import AttackOrder
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import entity, pool, profile, sample


def _unit(type_name: str, *, speed: float = 1.0, armed: bool = True) -> UnitStats:
    weapon = Weapon(
        shoot_delay=50.0,
        attack_range=110.0,
        direct_damage=17.0,
        direct_damage_volley=17.0,
        area_damage=0.0,
        area_damage_volley=0.0,
    )
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
        weapon=weapon if armed else None,
    )


_CATALOGUE: dict[str, UnitStats] = {
    "commandCenter": _unit("commandCenter", speed=0.0, armed=False),
    "c_tank": _unit("c_tank"),
}

_PROFILES: dict[str, CombatProfile] = {
    "commandCenter": profile("commandCenter", 0.0, land=False),
    "c_tank": profile("c_tank", 110.0),
}


def _tank(unit_id: int, x: float = 500.0, y: float = 500.0) -> Entity:
    return entity(unit_id, "c_tank", x=x, y=y)


def _world(*army: Entity, hostiles: tuple[Entity, ...] = ()) -> Sample:
    anchor = entity(1, "commandCenter", x=0.0, y=0.0)
    return sample(anchor, *army, *hostiles)


def test_the_gate_states_its_need_and_it_climbs_with_the_ladder() -> None:
    """The figure the raid's draft is arbitrated against, read through the
    same function muster uses so the two cannot drift ([[policy-raid]])."""
    waves = WaveController(ladder=(3, 5))
    assert waves.need() == 3
    army = tuple(_tank(10 + n) for n in range(3))
    waves.command(_world(*army), _CATALOGUE, _PROFILES, army)
    assert waves.need() == 5


def test_the_opening_need_holds_at_the_first_rung_while_the_gate_climbs() -> None:
    """The hunt's bar: the escalating rung starved its first screen to zero
    fires, so the hunt arbitrates against the opening's own figure -- which
    must stay put exactly when need() moves."""
    waves = WaveController(ladder=(3, 5))
    assert waves.opening_need() == 3
    army = tuple(_tank(10 + n) for n in range(3))
    waves.command(_world(*army), _CATALOGUE, _PROFILES, army)
    assert waves.need() == 5
    assert waves.opening_need() == 3


def test_below_the_first_wave_the_army_gathers_and_nobody_attacks() -> None:
    waves = WaveController()
    army = (_tank(10), _tank(11))
    assert len(army) < FIRST_WAVE
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    moves, attacks = waves.command(_world(*army, hostiles=(hostile,)), _CATALOGUE, _PROFILES, army)
    assert [move["unit_id"] for move in moves] == [10, 11]
    assert attacks == ()
    assert waves.rallied == 2
    assert waves.attack_orders == 0


def test_a_rally_is_sent_once_per_stint_not_once_per_sample() -> None:
    waves = WaveController()
    army = (_tank(10), _tank(11))
    world = _world(*army)
    waves.command(world, _CATALOGUE, _PROFILES, army)
    moves, _ = waves.command(world, _CATALOGUE, _PROFILES, army)
    assert moves == ()
    assert waves.rallied == 2


def test_a_full_reserve_is_released_and_attacks_together() -> None:
    waves = WaveController()
    army = tuple(_tank(10 + n) for n in range(FIRST_WAVE))
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    world = _world(*army, hostiles=(hostile,))
    moves, attacks = waves.command(world, _CATALOGUE, _PROFILES, army)
    assert moves == ()
    assert [attack["target_id"] for attack in attacks] == [9, 9, 9]
    assert waves.attack_orders == FIRST_WAVE


def test_an_attack_is_not_re_sent_while_the_pairing_holds() -> None:
    """Re-issuing an identical attack replaces the order with a copy of itself,
    and the unit never closes the distance.
    """
    waves = WaveController()
    army = tuple(_tank(10 + n) for n in range(FIRST_WAVE))
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    world = _world(*army, hostiles=(hostile,))
    waves.command(world, _CATALOGUE, _PROFILES, army)
    _, attacks = waves.command(world, _CATALOGUE, _PROFILES, army)
    assert attacks == ()
    assert waves.attack_orders == FIRST_WAVE


def test_killed_counts_attacked_targets_no_longer_visible() -> None:
    """Named for what was observed: a retreat into fog reads the same way."""
    waves = WaveController()
    army = tuple(_tank(10 + n) for n in range(FIRST_WAVE))
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    waves.command(_world(*army, hostiles=(hostile,)), _CATALOGUE, _PROFILES, army)
    assert waves.killed({9}) == 0
    assert waves.killed(set()) == 1


def test_forward_posts_the_reserve_at_the_frontier_extractor() -> None:
    """Six batches agree matches are decided by extractor drops 688-1,766
    units from where the army stands ([[policy-holding-ground]]); forward
    moves the gathering to where the match is decided. The farthest owned
    extractor from the anchor is the frontier one; an upgraded tier still
    counts, by the same any-tier test the plan's progress count trusts."""
    near = entity(30, "extractorT1", x=300.0, y=0.0)
    far = entity(31, "extractorT2", x=900.0, y=0.0)
    posted = sample(entity(1, "commandCenter", x=0.0, y=0.0), near, far, _tank(10))
    waves = WaveController(ladder=(10,), forward=True)
    moves, _ = waves.command(posted, _CATALOGUE, _PROFILES, (_tank(10),))
    assert [(m["x"], m["y"]) for m in moves] == [(900.0, 0.0)]
    # Off, the same world gathers at the anchor -- the measured behaviour.
    home = WaveController(ladder=(10,))
    moves, _ = home.command(posted, _CATALOGUE, _PROFILES, (_tank(10),))
    assert [(m["x"], m["y"]) for m in moves] == [(0.0, 0.0)]


def test_forward_falls_back_to_the_anchor_before_any_extractor_stands() -> None:
    """An opening with no economy yet gathers at the base, not nowhere."""
    bare = _world(_tank(10))
    waves = WaveController(ladder=(10,), forward=True)
    moves, _ = waves.command(bare, _CATALOGUE, _PROFILES, (_tank(10),))
    assert [(m["x"], m["y"]) for m in moves] == [(0.0, 0.0)]


def test_gather_reserve_marks_each_unit_and_never_repeats_it() -> None:
    rallying: set[int] = set()
    world = _world(_tank(10), _tank(11))
    reserve = (_tank(10), _tank(11))
    first = gather_reserve(world, _CATALOGUE, reserve, rallying)
    assert [order["unit_id"] for order in first] == [10, 11]
    assert gather_reserve(world, _CATALOGUE, reserve, rallying) == ()
    assert rallying == {10, 11}


def test_dispatch_attacks_skips_an_attacker_already_on_that_target() -> None:
    ordered: dict[int, int] = {10: 9}
    attacked: set[int] = set()
    engagements = (
        Engagement(attacker_id=10, target_id=9, reason=""),
        Engagement(attacker_id=11, target_id=9, reason=""),
    )
    sent: tuple[AttackOrder, ...] = dispatch_attacks(engagements, ordered, attacked)
    assert [order["unit_id"] for order in sent] == [11]
    assert ordered == {10: 9, 11: 9}
    assert attacked == {9}


def test_a_raider_inside_our_ground_pulls_the_reserve_onto_it() -> None:
    """The wave gate stops trickling into defended ground; it was never an
    argument for watching a raider kill the extractor beside the rally point.
    """
    waves = WaveController(intercept=True)
    army = (_tank(10), _tank(11))
    raider = entity(9, "c_tank", mine=False, hostile=True, x=200.0, y=0.0)
    world = _world(*army, hostiles=(raider,))
    _, attacks = waves.command(world, _CATALOGUE, _PROFILES, army)
    assert [attack["target_id"] for attack in attacks] == [9, 9]
    assert waves.intercepts == 2


def test_a_capped_guard_commits_only_the_nearest_detachment() -> None:
    """The cost case that makes the cap a question: one match logged 870
    intercepts and never massed an attack. An interception is a race with the
    damage the intruder is doing, so the detachment is the nearest engageable
    units and no more; the rest of the reserve keeps gathering toward the
    wave the offence still needs ([[policy-holding-ground]])."""
    waves = WaveController(ladder=(10,), intercept=True, guard_cap=2)
    army = (_tank(10, x=150.0), _tank(11, x=90.0), _tank(12, x=140.0), _tank(13, x=600.0))
    raider = entity(9, "c_tank", mine=False, hostile=True, x=200.0, y=0.0)
    world = _world(*army, hostiles=(raider,))
    _, attacks = waves.command(world, _CATALOGUE, _PROFILES, army)
    # 10 (50 away) and 12 (60 away) race; 11 and 13 keep gathering.
    assert [attack["unit_id"] for attack in attacks] == [10, 12]
    assert waves.intercepts == 2


def test_a_zero_cap_commits_the_whole_reserve() -> None:
    """Zero is the shipped behaviour both guard A/Bs measured, so it is a
    value rather than a special case."""
    waves = WaveController(ladder=(10,), intercept=True, guard_cap=0)
    army = (_tank(10), _tank(11), _tank(12))
    raider = entity(9, "c_tank", mine=False, hostile=True, x=200.0, y=0.0)
    _, attacks = waves.command(_world(*army, hostiles=(raider,)), _CATALOGUE, _PROFILES, army)
    assert [attack["unit_id"] for attack in attacks] == [10, 11, 12]


def test_the_riposte_releases_the_reserve_when_the_intrusion_ends() -> None:
    """The human counter-punch, as one flag.

    A raider stands on our ground, dies (or leaves), and the NEXT
    observation releases the whole gathered reserve below the ladder's rung
    -- the enemy's attack burned itself out, and the window before its next
    group finishes staging is when a stockpile converts
    ([[ai-opponent-strategy]]). Without the flag the same three ticks keep
    the reserve gathering toward a ten-unit rung it has not reached.
    """
    army = (_tank(10), _tank(11), _tank(12), _tank(13))
    raider = entity(9, "c_tank", mine=False, hostile=True, x=200.0, y=0.0)
    intruded = _world(*army, hostiles=(raider,))
    quiet = _world(*army)

    waves = WaveController(ladder=(10,), intercept=True, riposte=True)
    waves.command(intruded, _CATALOGUE, _PROFILES, army)
    waves.command(quiet, _CATALOGUE, _PROFILES, army)
    assert waves.released() == frozenset()
    waves.command(quiet, _CATALOGUE, _PROFILES, army)
    assert waves.released() == {10, 11, 12, 13}

    plain = WaveController(ladder=(10,), intercept=True)
    plain.command(intruded, _CATALOGUE, _PROFILES, army)
    plain.command(quiet, _CATALOGUE, _PROFILES, army)
    plain.command(quiet, _CATALOGUE, _PROFILES, army)
    assert plain.released() == frozenset()


def test_a_riposte_missed_is_not_banked() -> None:
    """A riposte with too few units is dropped, not saved for a moment that
    has lost its window: the reserve keeps gathering toward the rung."""
    army = (_tank(10), _tank(11))
    raider = entity(9, "c_tank", mine=False, hostile=True, x=200.0, y=0.0)
    waves = WaveController(ladder=(10,), intercept=True, riposte=True)
    waves.command(_world(*army, hostiles=(raider,)), _CATALOGUE, _PROFILES, army)
    waves.command(_world(*army), _CATALOGUE, _PROFILES, army)
    waves.command(_world(*army), _CATALOGUE, _PROFILES, army)
    assert waves.released() == frozenset()


def test_a_guard_is_sent_home_again_once_the_raid_ends() -> None:
    """A guard forgets it was ever rallied, so the gather pass re-rallies it
    rather than leaving it standing where the fight finished.
    """
    waves = WaveController(intercept=True)
    army = (_tank(10), _tank(11))
    raider = entity(9, "c_tank", mine=False, hostile=True, x=200.0, y=0.0)
    waves.command(_world(*army, hostiles=(raider,)), _CATALOGUE, _PROFILES, army)
    # The raid ends: the raider is gone, and the guards get fresh rally orders
    # on the next observation.
    _, attacks = waves.command(_world(*army), _CATALOGUE, _PROFILES, army)
    assert attacks == ()
    quiet, _ = waves.command(_world(*army), _CATALOGUE, _PROFILES, army)
    assert [move["unit_id"] for move in quiet] == [10, 11]


def test_a_distant_hostile_does_not_bypass_the_wave_gate() -> None:
    """Only intrusion bypasses the gate, and only intrusion should: attacking
    out is the wave's business, on the wave's terms.
    """
    waves = WaveController(intercept=True)
    army = (_tank(10), _tank(11))
    afar = entity(9, "c_tank", mine=False, hostile=True, x=5000.0, y=5000.0)
    _, attacks = waves.command(_world(*army, hostiles=(afar,)), _CATALOGUE, _PROFILES, army)
    assert attacks == ()
    assert waves.intercepts == 0


def test_the_allin_releases_everything_from_its_observation_onward() -> None:
    """Release on time, not on size.

    Forty-seven Impossible matches released on size and met an army that
    had compounded past answering, so the all-in holds everything to a
    chosen observation and then releases every muster -- the dump, and the
    stream of reinforcements behind it ([[policy-combat]]).
    """
    army = (_tank(10), _tank(11), _tank(12), _tank(13))
    quiet = _world(*army)

    waves = WaveController(ladder=(10,), allin_at=3)
    waves.command(quiet, _CATALOGUE, _PROFILES, army)
    waves.command(quiet, _CATALOGUE, _PROFILES, army)
    assert waves.released() == frozenset()
    waves.command(quiet, _CATALOGUE, _PROFILES, army)
    assert waves.released() == {10, 11, 12, 13}
    # The stream respects the anti-trickle floor: one straggler is held,
    # a first-wave's worth goes straight in.
    grown = (*army, _tank(14))
    waves.command(_world(*grown), _CATALOGUE, _PROFILES, grown)
    assert waves.released() == {10, 11, 12, 13}
    packet = (*grown, _tank(15), _tank(16))
    waves.command(_world(*packet), _CATALOGUE, _PROFILES, packet)
    assert waves.released() == {10, 11, 12, 13, 14, 15, 16}

    patient = WaveController(ladder=(10,))
    for _ in range(4):
        patient.command(quiet, _CATALOGUE, _PROFILES, army)
    assert patient.released() == frozenset()


def test_a_held_reserve_gathers_at_the_line_point_not_the_anchor() -> None:
    """The choke-holding verb: hold 50 posts the reserve at the midpoint of
    the anchor-to-mirror line -- the funnel's mouth on a symmetric map --
    instead of the base the terrain screen watched armies die defending
    from (log 2026-08-09). The mirror is the anchor reflected through the
    pool centroid, so with the anchor at the origin and one pool at
    (400, 0) the mirror is (800, 0) and the midpoint (400, 0)."""
    anchor = entity(1, "commandCenter", x=0.0, y=0.0)
    world = sample(anchor, _tank(10), pools=(pool(x=400.0, y=0.0),))
    rallying: set[int] = set()
    moves = gather_reserve(world, _CATALOGUE, (_tank(10),), rallying, hold=50)
    assert [(m["x"], m["y"]) for m in moves] == [(400.0, 0.0)]


def test_hold_outranks_forward_and_a_poolless_map_falls_back_to_the_anchor() -> None:
    """One field says where the army stands: hold wins over forward, and a
    map with no pools has no mirror -- the reserve keeps the anchor rather
    than inventing a point."""
    anchor = entity(1, "commandCenter", x=0.0, y=0.0)
    world = sample(anchor, _tank(10))
    rallying: set[int] = set()
    moves = gather_reserve(world, _CATALOGUE, (_tank(10),), rallying, forward=True, hold=50)
    assert [(m["x"], m["y"]) for m in moves] == [(0.0, 0.0)]
