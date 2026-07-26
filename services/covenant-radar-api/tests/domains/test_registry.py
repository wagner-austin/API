"""Tests for domain registry."""

from __future__ import annotations

import pytest

from covenant_radar_api.domains.protocols import DomainProtocol
from covenant_radar_api.domains.registry import DomainRegistry
from tests.domains._test_domain_fixtures import make_fake_domain


class TestDomainRegistry:
    """Tests for DomainRegistry class."""

    def test_empty_list_names(self) -> None:
        """New registry returns empty tuple."""
        registry = DomainRegistry()
        assert registry.list_names() == ()

    def test_register_and_get(self) -> None:
        """Register a domain factory and retrieve the built domain by name."""
        registry = DomainRegistry()
        registry.register("weather", lambda: make_fake_domain("weather"))

        result = registry.get("weather")
        assert result.config["name"] == "weather"

    def test_factory_runs_only_when_the_domain_is_requested(self) -> None:
        """Registering must not build, or every deployment pays for every domain.

        Weather reads a fitted state and station map off disk to build. If
        registration constructed it, a deployment running only esports would
        fail at startup demanding files it never opens.
        """
        registry = DomainRegistry()
        builds: list[str] = []

        def _build() -> DomainProtocol:
            builds.append("weather")
            return make_fake_domain("weather")

        registry.register("weather", _build)
        assert builds == []

        registry.get("weather")
        assert builds == ["weather"]

    def test_register_and_list_names(self) -> None:
        """Register two domains and list returns sorted tuple."""
        registry = DomainRegistry()
        registry.register("weather", lambda: make_fake_domain("weather"))
        registry.register("covenant", lambda: make_fake_domain("covenant"))

        names = registry.list_names()
        assert names == ("covenant", "weather")

    def test_get_nonexistent_raises_key_error(self) -> None:
        """Get unknown name raises KeyError."""
        registry = DomainRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_key_error_message_lists_available(self) -> None:
        """KeyError message includes available domain names."""
        registry = DomainRegistry()
        registry.register("alpha", lambda: make_fake_domain("alpha"))
        registry.register("beta", lambda: make_fake_domain("beta"))

        with pytest.raises(KeyError, match="alpha, beta"):
            registry.get("missing")

    def test_register_duplicate_raises_value_error(self) -> None:
        """Registering same name twice raises ValueError."""
        registry = DomainRegistry()
        registry.register("weather", lambda: make_fake_domain("weather"))

        with pytest.raises(ValueError, match="Domain 'weather' already registered"):
            registry.register("weather", lambda: make_fake_domain("weather"))

    def test_list_names_sorted(self) -> None:
        """Names are returned in sorted order regardless of insertion."""
        registry = DomainRegistry()
        registry.register("zulu", lambda: make_fake_domain("zulu"))
        registry.register("alpha", lambda: make_fake_domain("alpha"))
        registry.register("mike", lambda: make_fake_domain("mike"))

        assert registry.list_names() == ("alpha", "mike", "zulu")

    def test_register_multiple_domains(self) -> None:
        """Register three domains and retrieve each."""
        registry = DomainRegistry()
        registry.register("a", lambda: make_fake_domain("a"))
        registry.register("b", lambda: make_fake_domain("b"))
        registry.register("c", lambda: make_fake_domain("c"))

        assert registry.get("a").config["name"] == "a"
        assert registry.get("b").config["name"] == "b"
        assert registry.get("c").config["name"] == "c"

    def test_get_empty_registry_error_message(self) -> None:
        """KeyError on empty registry shows empty available list."""
        registry = DomainRegistry()

        with pytest.raises(KeyError, match="Available: "):
            registry.get("anything")
