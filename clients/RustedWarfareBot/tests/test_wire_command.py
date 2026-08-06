"""Encoding orders into the exact lines the agent's parser accepts.

The agent rejects a malformed order rather than skipping it, so the contract
these tests hold is byte-level: the encoder must emit what the Java parser was
written against, including field order.
"""

from __future__ import annotations

import pytest

from rw_bot.wire.command import (
    CommandError,
    ability_order,
    attack_move_order,
    attack_order,
    build_order,
    encode_ability,
    encode_ack,
    encode_attack,
    encode_attack_move,
    encode_build,
    encode_move,
    encode_produce,
    encode_targeted_ability,
    move_order,
    produce_order,
    targeted_ability_order,
)
from rw_bot.wire.posture import encode_posture, posture_order


def test_move_encodes_to_the_exact_agent_format() -> None:
    line = encode_move(move_order(unit_id=214, x=4550.0, y=2610.5))
    assert line == '{"kind":"move","unit_id":214,"x":4550.0,"y":2610.5}'


def test_build_encodes_to_the_exact_agent_format() -> None:
    line = encode_build(build_order(unit_id=214, type_name="landFactory", x=4450.0, y=2730.0))
    assert line == ('{"kind":"build","unit_id":214,"x":4450.0,"y":2730.0,"type":"landFactory"}')


def test_produce_encodes_to_the_exact_agent_format() -> None:
    line = encode_produce(produce_order(unit_id=213, type_name="scout"))
    assert line == '{"kind":"produce","unit_id":213,"type":"scout"}'


def test_a_produce_order_carries_no_position() -> None:
    """A produced unit rolls out of the building that made it.

    The engine decides where it appears, so a coordinate here would be a number
    the planner invented and the agent would have to ignore. Its absence is the
    contract, which is why it is asserted rather than assumed.
    """
    line = encode_produce(produce_order(unit_id=213, type_name="builder"))
    assert '"x"' not in line
    assert '"y"' not in line


def test_a_blank_produce_type_is_rejected() -> None:
    with pytest.raises(CommandError) as caught:
        produce_order(unit_id=213, type_name="  ")
    assert caught.value.code == "RW-CMD-001"


def test_ability_encodes_to_the_exact_agent_format() -> None:
    """The tech verb: a unit and the engine's interned action key, nothing
    else -- the land factory's tier-two upgrade converts into no type, so
    neither produce nor build can name it. A key rather than the engine's
    per-action index, because the index is not a selector: every action on
    a unit answers the same figure, and dispatching by it resolved the
    rally point four probes running ([[mechanics-build-actions]])."""
    line = encode_ability(ability_order(unit_id=213, key="c_2"))
    assert line == '{"kind":"ability","unit_id":213,"key":"c_2"}'


def test_a_blank_ability_key_is_rejected() -> None:
    """A keyless action cannot be dispatched; the agent would refuse it."""
    with pytest.raises(CommandError) as caught:
        ability_order(unit_id=213, key="  ")
    assert caught.value.code == "RW-CMD-004"


@pytest.mark.parametrize("hostile", ['c"2', "c\\2", "c\n2", "c\r2"])
def test_an_ability_key_the_flat_format_cannot_carry_is_rejected(hostile: str) -> None:
    with pytest.raises(CommandError) as caught:
        ability_order(unit_id=213, key=hostile)
    assert caught.value.code == "RW-CMD-004"


def test_a_targeted_ability_encodes_to_the_exact_agent_format() -> None:
    """The finisher's verb: the same key dispatch as the plain ability with
    the ground point chosen by the planner. The nuke launch is declared
    ``fireTurretXAtGround``, so the point is the whole decision -- the plain
    verb sends the unit's own position, which the launch would aim at
    itself ([[mechanics-build-actions]])."""
    line = encode_targeted_ability(targeted_ability_order(unit_id=213, key="c_3", x=512.5, y=768.0))
    assert line == '{"kind":"ability_at","unit_id":213,"x":512.5,"y":768.0,"key":"c_3"}'


def test_a_blank_targeted_ability_key_is_rejected() -> None:
    with pytest.raises(CommandError) as caught:
        targeted_ability_order(unit_id=213, key="  ", x=0.0, y=0.0)
    assert caught.value.code == "RW-CMD-004"


@pytest.mark.parametrize("hostile", ['c"3', "c\\3", "c\n3", "c\r3"])
def test_a_targeted_ability_key_the_flat_format_cannot_carry_is_rejected(
    hostile: str,
) -> None:
    with pytest.raises(CommandError) as caught:
        targeted_ability_order(unit_id=213, key=hostile, x=0.0, y=0.0)
    assert caught.value.code == "RW-CMD-004"


@pytest.mark.parametrize("literal", ["nan", "inf", "-inf"])
def test_a_non_finite_targeted_ability_coordinate_is_rejected(literal: str) -> None:
    """The engine aims a turret at this point; NaN would fire at nothing
    and report nothing."""
    with pytest.raises(CommandError) as caught:
        targeted_ability_order(unit_id=213, key="c_3", x=float(literal), y=0.0)
    assert caught.value.code == "RW-CMD-002"


@pytest.mark.parametrize("hostile", ['sc"out', "sc\\out", "sc\nout", "sc\rout"])
def test_a_produce_type_the_flat_format_cannot_carry_is_rejected(hostile: str) -> None:
    with pytest.raises(CommandError) as caught:
        produce_order(unit_id=213, type_name=hostile)
    assert caught.value.code == "RW-CMD-001"


def test_attack_encodes_to_the_exact_agent_format() -> None:
    line = encode_attack(attack_order(unit_id=276, target_id=216))
    assert line == '{"kind":"attack","unit_id":276,"target_id":216}'


def test_an_attack_carries_no_position() -> None:
    """The target's identity is the whole of the order.

    A move sends a unit to where the target stood; naming the target is what
    makes the engine follow it as it moves, so a coordinate here would be a
    number nothing reads.
    """
    line = encode_attack(attack_order(unit_id=1, target_id=2))
    assert '"x"' not in line
    assert '"y"' not in line


def test_a_unit_cannot_be_ordered_to_attack_itself() -> None:
    """The engine accepts it and then cannot act on it."""
    with pytest.raises(CommandError) as caught:
        attack_order(unit_id=7, target_id=7)
    assert caught.value.code == "RW-CMD-003"


def test_the_ack_is_a_bare_verb() -> None:
    """It is about the exchange, not about a unit.

    In lockstep the agent holds the simulation after each sample until this
    arrives, so it carries no subject and needs none.
    """
    assert encode_ack() == '{"kind":"ack"}'


def test_a_move_never_carries_a_build_type() -> None:
    """The agent rejects a move that does, so the encoder must not emit one."""
    assert "type" not in encode_move(move_order(unit_id=1, x=0.0, y=0.0))


def test_no_encoded_order_contains_a_newline() -> None:
    """The agent splits on newlines before parsing."""
    lines = [
        encode_move(move_order(unit_id=1, x=-1.5, y=2.5)),
        encode_build(build_order(unit_id=2, type_name="extractorT1", x=3.0, y=4.0)),
    ]
    for line in lines:
        assert "\n" not in line
        assert "\r" not in line


def test_negative_coordinates_encode() -> None:
    line = encode_move(move_order(unit_id=7, x=-1000.0, y=-1000.0))
    assert line == '{"kind":"move","unit_id":7,"x":-1000.0,"y":-1000.0}'


def test_a_blank_build_type_is_rejected() -> None:
    with pytest.raises(CommandError) as caught:
        build_order(unit_id=1, type_name="   ", x=0.0, y=0.0)
    assert caught.value.code == "RW-CMD-001"


@pytest.mark.parametrize("literal", ["nan", "inf", "-inf"])
def test_a_non_finite_coordinate_is_rejected(literal: str) -> None:
    with pytest.raises(CommandError) as caught:
        move_order(unit_id=1, x=float(literal), y=0.0)
    assert caught.value.code == "RW-CMD-002"


@pytest.mark.parametrize("literal", ["nan", "inf"])
def test_a_non_finite_build_coordinate_is_rejected(literal: str) -> None:
    with pytest.raises(CommandError) as caught:
        build_order(unit_id=1, type_name="landFactory", x=0.0, y=float(literal))
    assert caught.value.code == "RW-CMD-002"


@pytest.mark.parametrize("hostile", ['land"Factory', "land\\Factory", "land\nFactory"])
def test_a_type_name_the_flat_format_cannot_carry_is_rejected(hostile: str) -> None:
    """Rejected at encode time rather than emitted for the agent to refuse."""
    with pytest.raises(CommandError) as caught:
        encode_build(build_order(unit_id=1, type_name=hostile, x=0.0, y=0.0))
    assert caught.value.code == "RW-CMD-001"


def test_attack_move_encodes_to_the_exact_agent_format() -> None:
    order = attack_move_order(unit_id=214, x=4550.0, y=2610.0)
    assert encode_attack_move(order) == '{"kind":"attack_move","unit_id":214,"x":4550.0,"y":2610.0}'


def test_an_attack_move_coordinate_must_be_finite() -> None:
    """The engine reads the point straight into a waypoint; NaN would walk
    nowhere and report nothing.
    """
    with pytest.raises(CommandError) as caught:
        attack_move_order(unit_id=214, x=float("nan"), y=0.0)
    assert caught.value.code == "RW-CMD-002"


def test_posture_encodes_to_the_exact_agent_format() -> None:
    """The reflex layer's table row: a type, its reach, its reflexes.

    Not an order to a unit -- the agent stores it and applies it between
    samples, at the engine's own pace ([[community-play-strategies]]).
    """
    line = encode_posture(
        posture_order(type_name="c_artillery", reach=290.0, speed=0.6, kite=True, hp_floor=30)
    )
    assert line == (
        '{"kind":"posture","type":"c_artillery","reach":290.0,"speed":0.6,"kite":1,"hp_floor":30}'
    )
    off = encode_posture(
        posture_order(type_name="c_tank", reach=130.0, speed=1.1, kite=False, hp_floor=0)
    )
    assert off == (
        '{"kind":"posture","type":"c_tank","reach":130.0,"speed":1.1,"kite":0,"hp_floor":0}'
    )


def test_a_posture_reach_must_be_finite_and_non_negative() -> None:
    with pytest.raises(CommandError) as caught:
        posture_order(type_name="c_tank", reach=float("nan"), speed=1.0, kite=False, hp_floor=0)
    assert caught.value.code == "RW-CMD-002"
    with pytest.raises(CommandError) as negative:
        posture_order(type_name="c_tank", reach=-1.0, speed=1.0, kite=False, hp_floor=0)
    assert negative.value.code == "RW-CMD-002"


@pytest.mark.parametrize("floor", [-1, 101])
def test_a_posture_floor_outside_percent_range_is_rejected(floor: int) -> None:
    with pytest.raises(CommandError) as caught:
        posture_order(type_name="c_tank", reach=130.0, speed=1.0, kite=False, hp_floor=floor)
    assert caught.value.code == "RW-CMD-005"


def test_a_posture_type_the_flat_format_cannot_carry_is_rejected() -> None:
    with pytest.raises(CommandError) as caught:
        posture_order(type_name='c"tank', reach=130.0, speed=1.0, kite=False, hp_floor=0)
    assert caught.value.code == "RW-CMD-001"
