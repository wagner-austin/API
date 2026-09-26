"""Roundtrip and validation tests for the fleet report codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.bot.ai.types import AIConfigDict, make_default_ai_config
from tankpit_bot.bot.ai.types_codecs import decode_ai_config, encode_ai_config
from tankpit_bot.fleetshare.codecs import (
    decode_fleet_container_sighting,
    decode_fleet_enemy_sighting,
    decode_fleet_report,
    encode_fleet_report,
)
from tankpit_bot.fleetshare.types import (
    EngagementDoctrine,
    FleetContainerRemovalDict,
    FleetContainerSightingDict,
    FleetEnemySightingDict,
    FleetMineSightingDict,
    FleetReportDict,
    FleetRole,
    FleetScannedTileDict,
)


def _report() -> FleetReportDict:
    """Build a fully populated report for roundtrip pins."""
    return FleetReportDict(
        instance="arterial",
        team=2,
        room="6",
        tank_id=2731,
        role=FleetRole.GATHERER,
        x=100,
        y=120,
        war_ready=True,
        engaged_target_id=506,
        forage_goal_x=120,
        forage_goal_y=104,
        collect_claim_x=44,
        collect_claim_y=24,
        combat_consent_ids=[],
        written_ms=100000,
        enemies=[
            FleetEnemySightingDict(
                tank_id=506,
                name="red-7",
                team=0,
                rank=1,
                x=172,
                y=45,
                damage_state=1,
                observed_ms=99000,
            )
        ],
        containers=[
            FleetContainerSightingDict(x=50, y=60, is_fuel=True, volume=700, observed_ms=98000),
            FleetContainerSightingDict(x=51, y=61, is_fuel=False, volume=0, observed_ms=97000),
        ],
        removed=[FleetContainerRemovalDict(x=44, y=45, removed_ms=95000)],
        mines=[
            FleetMineSightingDict(x=101, y=100, mine_type=1, tank_id=709, team=1, observed_ms=96000)
        ],
        scanned=[FleetScannedTileDict(x=10, y=10, observed_ms=96000)],
    )


def test_report_roundtrips_through_encode_decode() -> None:
    """A full report survives encode -> decode unchanged."""
    report = _report()
    assert decode_fleet_report(encode_fleet_report(report)) == report


def test_decode_report_rejects_non_object() -> None:
    """A non-object report payload names the shape in the error."""
    with pytest.raises(JSONTypeError, match="fleet report must be an object"):
        decode_fleet_report([1, 2, 3])


def test_decode_enemy_sighting_rejects_non_object() -> None:
    """A non-object enemy row names the shape in the error."""
    with pytest.raises(JSONTypeError, match="enemy sighting must be an object"):
        decode_fleet_enemy_sighting("nope")


def test_decode_container_sighting_rejects_non_object() -> None:
    """A non-object container row names the shape in the error."""
    with pytest.raises(JSONTypeError, match="container sighting must be an object"):
        decode_fleet_container_sighting(7)


def test_decode_mine_sighting_rejects_non_object() -> None:
    """A non-object mine row names the shape in the error."""
    from tankpit_bot.fleetshare.codecs import decode_fleet_mine_sighting

    with pytest.raises(JSONTypeError, match="mine sighting must be an object"):
        decode_fleet_mine_sighting([1, 2])


def test_decode_report_rejects_missing_field() -> None:
    """A report missing a required key raises with the key named."""
    payload = encode_fleet_report(_report())
    del payload["tank_id"]
    with pytest.raises(JSONTypeError, match="tank_id"):
        decode_fleet_report(payload)


def test_decode_report_rejects_non_int_consent_id() -> None:
    """A non-int entry in ``combat_consent_ids`` raises with its index."""
    payload = encode_fleet_report(_report())
    payload["combat_consent_ids"] = [709, "1301"]
    with pytest.raises(JSONTypeError, match=r"combat_consent_ids\[1\] must be an int"):
        decode_fleet_report(payload)


@pytest.mark.parametrize("role", list(FleetRole))
def test_report_role_round_trips_for_every_role(role: FleetRole) -> None:
    """Both roles encode to their word and decode back to the member."""
    payload = encode_fleet_report(FleetReportDict(**{**_report(), "role": role}))
    assert payload["role"] == role.value
    assert decode_fleet_report(payload)["role"] is role


def test_report_rejects_unknown_role() -> None:
    """An unknown role raises with the word and the valid set named."""
    payload = encode_fleet_report(_report())
    payload["role"] = "medic"
    with pytest.raises(JSONTypeError, match="Invalid role 'medic': must be one of 'fighter'"):
        decode_fleet_report(payload)


def test_engagement_doctrine_vocabulary_order() -> None:
    """The doctrines in declaration order, skirmish (the default) first."""
    assert [doctrine.value for doctrine in EngagementDoctrine] == [
        "skirmish",
        "swarm",
        "duelist",
        "passive",
    ]


@pytest.mark.parametrize("doctrine", list(EngagementDoctrine))
def test_ai_config_doctrine_round_trips(doctrine: EngagementDoctrine) -> None:
    """Every doctrine survives the AI config codec, its only decoder."""
    config = AIConfigDict(**{**make_default_ai_config(), "doctrine": doctrine})
    assert decode_ai_config(encode_ai_config(config))["doctrine"] is doctrine


def test_ai_config_rejects_unknown_doctrine() -> None:
    """A doctrine outside the vocabulary raises rather than defaulting.

    A config that silently fell back to skirmish would have a bot
    fighting under a doctrine nobody chose, which is the whole reason
    the selector is validated instead of passed through.
    """
    payload = encode_ai_config(make_default_ai_config())
    payload["doctrine"] = "banzai"
    with pytest.raises(JSONTypeError, match="Invalid doctrine 'banzai'"):
        decode_ai_config(payload)


def test_decode_scanned_tile_rejects_non_object() -> None:
    """A non-object coverage row names the shape in the error."""
    from tankpit_bot.fleetshare.codecs import decode_fleet_scanned_tile

    with pytest.raises(JSONTypeError, match="scanned tile must be an object"):
        decode_fleet_scanned_tile([1])


def test_decode_container_removal_rejects_non_object() -> None:
    """A non-object removal row names the shape in the error."""
    from tankpit_bot.fleetshare.codecs import decode_fleet_container_removal

    with pytest.raises(JSONTypeError, match="container removal must be an object"):
        decode_fleet_container_removal("x")
