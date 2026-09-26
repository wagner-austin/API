"""Tests for the durable AI mode vocabulary and the mode/substate pairing rule."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.bot.ai.types_codecs import decode_ai_state, encode_ai_state
from tankpit_bot.types.modes import (
    COLLECT_MODE_STATES,
    HUNT_MODE_STATES,
    AIMode,
    AIModeState,
    is_valid_ai_mode_state,
)


def test_ai_modes_are_unset_hunt_and_collect() -> None:
    """The durable modes in declaration order, each carrying its own word."""
    assert [(mode.name, mode.value) for mode in AIMode] == [
        ("UNSET", "UNSET"),
        ("HUNT", "HUNT"),
        ("COLLECT", "COLLECT"),
    ]


def test_ai_mode_states_partition_into_hunt_and_collect() -> None:
    """Every substate but NONE belongs to exactly one mode, NONE to UNSET."""
    assert AIModeState.NONE.value == ""
    assert set(HUNT_MODE_STATES).isdisjoint(COLLECT_MODE_STATES)
    assert {AIModeState.NONE, *HUNT_MODE_STATES, *COLLECT_MODE_STATES} == set(AIModeState)


def test_ai_state_decodes_mode_and_substate_to_members() -> None:
    """The AI-state codec narrows both words to members."""
    encoded = encode_ai_state(make_initial_ai_state())
    encoded["mode"] = "HUNT"
    encoded["mode_state"] = "APPROACH"
    with pytest.raises(ValueError, match="mode_state 'APPROACH' is invalid for mode 'HUNT'"):
        decode_ai_state(encoded)
    encoded["mode_state"] = "ACQUIRE"
    decoded = decode_ai_state(encoded)
    assert decoded["mode"] is AIMode.HUNT
    assert decoded["mode_state"] is AIModeState.ACQUIRE


def test_ai_state_rejects_an_unknown_mode() -> None:
    """A mode outside the vocabulary is refused by name."""
    encoded = encode_ai_state(make_initial_ai_state())
    encoded["mode"] = "PATROL"
    with pytest.raises(JSONTypeError, match="Invalid mode 'PATROL'"):
        decode_ai_state(encoded)


def test_ai_state_rejects_an_unknown_substate() -> None:
    """A substate outside the vocabulary is refused by name."""
    encoded = encode_ai_state(make_initial_ai_state())
    encoded["mode_state"] = "PATROL"
    with pytest.raises(JSONTypeError, match="Invalid mode_state 'PATROL'"):
        decode_ai_state(encoded)


def test_validates_unset_mode_state_pair() -> None:
    """UNSET is valid only with the empty substate."""
    assert is_valid_ai_mode_state(AIMode.UNSET, AIModeState.NONE) is True
    assert is_valid_ai_mode_state(AIMode.UNSET, AIModeState.ACQUIRE) is False


def test_validates_hunt_mode_state_pair() -> None:
    """HUNT accepts only hunt substates."""
    assert is_valid_ai_mode_state(AIMode.HUNT, AIModeState.ACQUIRE) is True
    assert is_valid_ai_mode_state(AIMode.HUNT, AIModeState.APPROACH) is False


def test_validates_recovery_mode_state_pair() -> None:
    """COLLECT accepts only collect substates."""
    assert is_valid_ai_mode_state(AIMode.COLLECT, AIModeState.SEARCH) is True
    assert is_valid_ai_mode_state(AIMode.COLLECT, AIModeState.PICKUP) is True
    assert is_valid_ai_mode_state(AIMode.COLLECT, AIModeState.ENGAGE) is False
