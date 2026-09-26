"""Tests for fleet role resolution and its behavior gates."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.fleetshare.role import resolve_engagement_doctrine, resolve_fleet_role
from tankpit_bot.fleetshare.types import EngagementDoctrine, FleetRole
from tests.conftest import FakeEnv


def test_unset_role_is_fighter() -> None:
    """The full doctrine is the primary configuration."""
    assert resolve_fleet_role() is FleetRole.FIGHTER


def test_env_selects_the_gatherer(fake_env: FakeEnv) -> None:
    """``TANKPIT_ROLE=gatherer`` selects the scout role."""
    fake_env.set("TANKPIT_ROLE", "gatherer")
    assert resolve_fleet_role() is FleetRole.GATHERER


def test_unknown_role_raises(fake_env: FakeEnv) -> None:
    """An unknown role names the variable, the word and the valid set."""
    fake_env.set("TANKPIT_ROLE", "medic")
    with pytest.raises(
        JSONTypeError, match="Invalid TANKPIT_ROLE 'medic': must be one of 'fighter', 'gatherer'"
    ):
        resolve_fleet_role()


class TestResolveEngagementDoctrine:
    """The doctrine resolver mirrors the role resolver's contract."""

    def test_unset_defaults_to_skirmish(self, fake_env: FakeEnv) -> None:
        """No TANKPIT_DOCTRINE means today's behavior."""
        del fake_env
        assert resolve_engagement_doctrine() is EngagementDoctrine.SKIRMISH

    def test_each_doctrine_resolves(self, fake_env: FakeEnv) -> None:
        """Every member of the vocabulary round-trips through the env."""
        for doctrine in EngagementDoctrine:
            fake_env.set("TANKPIT_DOCTRINE", doctrine.value)
            assert resolve_engagement_doctrine() is doctrine

    def test_unknown_doctrine_raises(self, fake_env: FakeEnv) -> None:
        """A typo'd doctrine fails loudly at launch, never best-effort."""
        fake_env.set("TANKPIT_DOCTRINE", "berserk")
        with pytest.raises(JSONTypeError, match="Invalid TANKPIT_DOCTRINE 'berserk'"):
            resolve_engagement_doctrine()
