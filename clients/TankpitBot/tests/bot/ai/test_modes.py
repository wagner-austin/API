"""Tests for durable AI mode literals and validation."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.types.modes import (
    AI_MODE_STATES,
    AI_MODES,
    is_valid_ai_mode_state,
    require_ai_mode,
    require_ai_mode_state,
)


def test_ai_modes_include_unset_and_refactor_targets() -> None:
    """Durable mode literals include unset plus the planned top-level modes."""
    assert AI_MODES == (
        "UNSET",
        "HUNT",
        "COLLECT",
    )


def test_ai_mode_states_include_hunt_and_recovery_states() -> None:
    """Durable mode substates include the expected migration vocabulary."""
    assert "" in AI_MODE_STATES
    assert "ACQUIRE" in AI_MODE_STATES
    assert "SEARCH" in AI_MODE_STATES
    assert "DONE" in AI_MODE_STATES


def test_require_ai_mode_accepts_valid_mode() -> None:
    """JSON mode validation accepts a supported durable mode."""
    data: JSONObject = {"mode": "HUNT"}
    assert require_ai_mode(data, "mode") == "HUNT"


def test_require_ai_mode_rejects_invalid_mode() -> None:
    """JSON mode validation rejects unsupported values."""
    data: JSONObject = {"mode": "PATROL"}
    with pytest.raises(ValueError, match="must be one of"):
        require_ai_mode(data, "mode")


def test_require_ai_mode_state_accepts_valid_state() -> None:
    """JSON substate validation accepts a supported durable substate."""
    data: JSONObject = {"mode_state": "APPROACH"}
    assert require_ai_mode_state(data, "mode_state") == "APPROACH"


def test_require_ai_mode_state_rejects_invalid_state() -> None:
    """JSON substate validation rejects unsupported values."""
    data: JSONObject = {"mode_state": "PATROL"}
    with pytest.raises(ValueError, match="must be one of"):
        require_ai_mode_state(data, "mode_state")


def test_validates_unset_mode_state_pair() -> None:
    """UNSET is valid only with the empty substate."""
    assert is_valid_ai_mode_state("UNSET", "") is True
    assert is_valid_ai_mode_state("UNSET", "ACQUIRE") is False


def test_validates_hunt_mode_state_pair() -> None:
    """HUNT accepts only hunt substates."""
    assert is_valid_ai_mode_state("HUNT", "ACQUIRE") is True
    assert is_valid_ai_mode_state("HUNT", "APPROACH") is False


def test_validates_recovery_mode_state_pair() -> None:
    """Recovery modes accept only recovery substates."""
    assert is_valid_ai_mode_state("COLLECT", "SEARCH") is True
    assert is_valid_ai_mode_state("COLLECT", "PICKUP") is True
    assert is_valid_ai_mode_state("COLLECT", "ENGAGE") is False
