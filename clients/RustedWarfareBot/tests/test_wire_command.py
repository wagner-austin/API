"""Encoding orders into the exact lines the agent's parser accepts.

The agent rejects a malformed order rather than skipping it, so the contract
these tests hold is byte-level: the encoder must emit what the Java parser was
written against, including field order.
"""

from __future__ import annotations

import pytest

from rw_bot.wire.command import (
    CommandError,
    build_order,
    encode_build,
    encode_move,
    encode_produce,
    move_order,
    produce_order,
)


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


@pytest.mark.parametrize("hostile", ['sc"out', "sc\\out", "sc\nout", "sc\rout"])
def test_a_produce_type_the_flat_format_cannot_carry_is_rejected(hostile: str) -> None:
    with pytest.raises(CommandError) as caught:
        produce_order(unit_id=213, type_name=hostile)
    assert caught.value.code == "RW-CMD-001"


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
