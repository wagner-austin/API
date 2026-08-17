"""Tests for fleet role resolution and its behavior gates."""

from __future__ import annotations

import pytest

from tankpit_bot.fleetshare.role import resolve_fleet_role
from tests.conftest import FakeEnv


def test_unset_role_is_fighter() -> None:
    """The full doctrine is the primary configuration."""
    assert resolve_fleet_role() == "fighter"


def test_env_selects_the_gatherer(fake_env: FakeEnv) -> None:
    """``TANKPIT_ROLE=gatherer`` selects the scout role."""
    fake_env.set("TANKPIT_ROLE", "gatherer")
    assert resolve_fleet_role() == "gatherer"


def test_unknown_role_raises(fake_env: FakeEnv) -> None:
    """An unknown role names the valid set in the error."""
    fake_env.set("TANKPIT_ROLE", "medic")
    with pytest.raises(ValueError, match="TANKPIT_ROLE must be one of"):
        resolve_fleet_role()
