"""Reach, and the layers that reach covers.

The headline cases run against the real dump under ``wiki/sources/m11-pools/``,
so what is tested is the contract against bytes the agent actually wrote. The
engine's own attackability branch is transcribed in :func:`can_engage`, and the
cases below walk every arm of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.mechanics.combat_profile import (
    CombatProfileError,
    can_engage,
    decode_combat_profiles,
    encode_combat_profile,
    is_armed,
    profile_of,
    reach_of,
)
from rw_bot.mechanics.registry_dump import RegistryDumpError
from rw_bot.validation import DecodeError
from tests.wire_fixtures import entity, profile

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DUMP = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"

_RECORD = (
    '{"kind":"unitcombat","index":0,"name":"turret","attack_range":165.0,'
    '"hits_land":true,"hits_air":false,"hits_underwater":false,'
    '"hits_land_out_of_water":true}'
)


def _dump_lines() -> list[str]:
    return _DUMP.read_text(encoding="utf-8").splitlines()


def test_the_real_dump_describes_every_registered_type() -> None:
    """173, against the 90 the stat catalogue carries.

    That gap is the whole reason this decoder exists: reading reach from
    ``-printunits`` treated 48 armed types as harmless, every turret among them.
    """
    profiles = decode_combat_profiles(_dump_lines())
    assert len(profiles) == 173


def test_the_only_unit_the_plan_builds_cannot_shoot_aircraft() -> None:
    """Read from the live registry, and it agrees with ``tanks/tank.ini``.

    ``maxAttackRange: 130``, ``canAttackFlyingUnits: false``,
    ``canAttackUnderwaterUnits: false``. This is the fact the combat policy was
    missing entirely ([[mechanics-combat-profile]]).
    """
    tank = decode_combat_profiles(_dump_lines())["c_tank"]
    assert tank["attack_range"] == 130.0
    assert tank["hits_land"] is True
    assert tank["hits_air"] is False
    assert tank["hits_underwater"] is False


def test_an_anti_air_turret_reports_that_it_cannot_hit_the_ground() -> None:
    """The case that poisoned the threat model.

    Armed, 250 units of reach, and unable to touch a builder walking past. Reach
    alone made it rule out every pool within 250 units of one.
    """
    turret = decode_combat_profiles(_dump_lines())["antiAirTurret"]
    assert turret["attack_range"] == 250.0
    assert turret["hits_air"] is True
    assert turret["hits_land"] is False


def test_exactly_the_underwater_hulls_carry_torpedoes() -> None:
    """``hits_land_out_of_water`` false, which is what a torpedo means.

    Carried rather than assumed: it is true for every armed type in the game
    except these four, and those four are precisely the ones that matter on a
    water map. A torpedo has to strike something in the water, so a submarine
    lurking offshore is not a reason to refuse a pool inland.
    """
    profiles = decode_combat_profiles(_dump_lines())
    torpedoes = sorted(
        p["type_name"]
        for p in profiles.values()
        if p["attack_range"] > 0.0 and not p["hits_land_out_of_water"]
    )
    assert torpedoes == [
        "c_amphibiousJet_underwater",
        "heavySub",
        "lightSub",
        "nautilusSubmarine",
    ]
    assert all(profiles[name]["hits_underwater"] for name in torpedoes)


def test_an_unarmed_type_reaches_no_layer_at_all() -> None:
    """Zero is an answer, not an absence.

    The base predicates return true for air and land regardless of armament,
    because the engine only consults them once a weapon is established. Reporting
    that unfiltered would put "a Builder can shoot aircraft" on the wire.
    """
    builder = decode_combat_profiles(_dump_lines())["builder"]
    assert builder["attack_range"] == 0.0
    assert builder["hits_land"] is False
    assert builder["hits_air"] is False
    assert builder["hits_underwater"] is False


def test_the_decoder_skips_the_kinds_it_does_not_own() -> None:
    """One file, three kinds, three decoders that each project their own."""
    lines = [
        '{"kind":"unittype","index":0,"name":"extractorT1","needs_pool":true}',
        '{"kind":"buildedge","index":0,"producer":"builder","produces":"landFactory"}',
        _RECORD,
    ]
    assert list(decode_combat_profiles(lines)) == ["turret"]


def test_an_unknown_kind_is_rejected() -> None:
    with pytest.raises(RegistryDumpError) as caught:
        decode_combat_profiles(['{"kind":"nonsense","index":0,"name":"x"}'])
    assert caught.value.code == "RW-REGISTRY-001"


def test_a_repeated_type_name_is_rejected() -> None:
    """It is the join key to live entities, so it must identify one type."""
    with pytest.raises(CombatProfileError) as caught:
        decode_combat_profiles([_RECORD, _RECORD])
    assert caught.value.code == "RW-COMBAT-001"


def test_blank_lines_are_skipped() -> None:
    """The dump is appended to, so a trailing newline is the normal case."""
    assert list(decode_combat_profiles(["", _RECORD, "   ", ""])) == ["turret"]


def test_a_missing_field_propagates_as_a_decode_error() -> None:
    with pytest.raises(DecodeError):
        decode_combat_profiles(['{"kind":"unitcombat","index":0,"name":"turret"}'])


def test_a_profile_round_trips_through_its_record() -> None:
    decoded = decode_combat_profiles([_RECORD])
    assert encode_combat_profile(decoded["turret"]) == _RECORD


def test_every_profile_in_the_real_dump_round_trips() -> None:
    """The claim is a round trip, not a round trip for the values seen so far."""
    profiles = decode_combat_profiles(_dump_lines())
    re_encoded = [encode_combat_profile(p) for p in profiles.values()]
    assert decode_combat_profiles(re_encoded) == profiles


def test_a_type_the_dump_does_not_carry_fails_loudly() -> None:
    """Indexed rather than defaulted: a miss is a stale dump, not an unarmed unit."""
    with pytest.raises(CombatProfileError) as caught:
        profile_of({}, "someModTank")
    assert caught.value.code == "RW-COMBAT-002"
    assert "someModTank" in caught.value.message


_TANK = profile("c_tank", 130.0)
_AA = profile("antiAirTurret", 250.0, land=False, air=True)
_SUB = profile("heavySub", 210.0, underwater=True, out_of_water=False)
_PROFILES = {
    "c_tank": _TANK,
    "antiAirTurret": _AA,
    "heavySub": _SUB,
    "builder": profile("builder", 0.0, land=False),
}


def test_reach_and_armament_read_off_the_same_record() -> None:
    tank = entity(1, "c_tank")
    assert reach_of(_PROFILES, tank) == 130.0
    assert is_armed(_PROFILES, tank) is True
    assert is_armed(_PROFILES, entity(2, "builder")) is False


def test_a_ground_weapon_reaches_a_ground_target() -> None:
    assert can_engage(_PROFILES, entity(1, "c_tank"), entity(9, "c_tank")) is True


def test_a_ground_weapon_does_not_reach_an_airborne_target() -> None:
    assert can_engage(_PROFILES, entity(1, "c_tank"), entity(9, "c_tank", flying=True)) is False


def test_an_air_weapon_reaches_an_airborne_target() -> None:
    flyer = entity(9, "helicopter", flying=True)
    assert can_engage(_PROFILES, entity(1, "antiAirTurret"), flyer) is True


def test_an_air_only_weapon_does_not_reach_the_ground() -> None:
    assert can_engage(_PROFILES, entity(1, "antiAirTurret"), entity(9, "c_tank")) is False


def test_a_submerged_target_needs_an_underwater_weapon() -> None:
    submerged = entity(9, "lightSub", submerged=True)
    assert can_engage(_PROFILES, entity(1, "c_tank"), submerged) is False
    assert can_engage(_PROFILES, entity(1, "heavySub"), submerged) is True


def test_airborne_is_tested_before_submerged() -> None:
    """The engine's order, kept so a unit that is somehow both is judged as it would be."""
    both = entity(9, "oddity", flying=True, submerged=True)
    assert can_engage(_PROFILES, entity(1, "heavySub"), both) is False


def test_a_torpedo_reaches_a_target_standing_in_water() -> None:
    wading = entity(9, "builder", touching_water=True)
    assert can_engage(_PROFILES, entity(1, "heavySub"), wading) is True


def test_a_torpedo_does_not_reach_a_target_clear_of_water() -> None:
    assert can_engage(_PROFILES, entity(1, "heavySub"), entity(9, "builder")) is False
