"""Tests for the canonical account-identity model."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.state.types import (
    decode_self_account,
    encode_self_account,
    make_empty_self_account,
)


def test_empty_model_is_all_sentinels() -> None:
    """Never-observed fields read as sentinels, not fabricated values."""
    account = make_empty_self_account()

    assert account["name"] == ""
    assert account["persistent_tank_id"] == -1
    assert account["leaderboard_position"] == -1
    assert account["promotion_points"] == -1
    assert account["identity_observed_ms"] == 0
    assert account["stats_observed_ms"] == 0


def test_roundtrip_preserves_every_field() -> None:
    """encode -> decode is the identity."""
    account = make_empty_self_account()
    account["name"] = "Artax"
    account["persistent_tank_id"] = 62913
    account["decoration_state_hex"] = "1e000000"
    account["rank_name"] = "private"
    account["leaderboard_position"] = 25
    account["promotion_points"] = 381015
    account["destroyed_enemies"] = 1179
    account["deactivated_total"] = 3
    account["play_time_s"] = 265843
    account["identity_observed_ms"] = 111
    account["stats_observed_ms"] = 222

    assert decode_self_account(encode_self_account(account)) == account


def test_decode_rejects_missing_fields() -> None:
    """Strict decode: a partial object is an error, not a default."""
    with pytest.raises(JSONTypeError):
        decode_self_account({"name": "Artax"})
