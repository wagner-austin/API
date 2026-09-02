"""Tests for fleet role resolution and its behavior gates."""

from __future__ import annotations

import pytest

from tankpit_bot.fleetshare.role import resolve_engagement_doctrine, resolve_fleet_role
from tankpit_bot.fleetshare.types import ENGAGEMENT_DOCTRINES
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


class TestResolveEngagementDoctrine:
    """The doctrine resolver mirrors the role resolver's contract."""

    def test_unset_defaults_to_skirmish(self, fake_env: FakeEnv) -> None:
        """No TANKPIT_DOCTRINE means today's behavior."""
        del fake_env
        assert resolve_engagement_doctrine() == "skirmish"

    def test_each_doctrine_resolves(self, fake_env: FakeEnv) -> None:
        """Every member of the literal round-trips through the env."""
        for doctrine in ENGAGEMENT_DOCTRINES:
            fake_env.set("TANKPIT_DOCTRINE", doctrine)
            assert resolve_engagement_doctrine() == doctrine

    def test_unknown_doctrine_raises(self, fake_env: FakeEnv) -> None:
        """A typo'd doctrine fails loudly at launch, never best-effort."""
        fake_env.set("TANKPIT_DOCTRINE", "berserk")
        with pytest.raises(ValueError, match="TANKPIT_DOCTRINE must be one of"):
            resolve_engagement_doctrine()
