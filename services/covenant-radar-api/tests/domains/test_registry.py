"""Tests for domain registry."""

from __future__ import annotations

import pytest

from covenant_radar_api.domains.registry import DomainRegistry
from tests.domains.test_protocols import FakeDomain


class TestDomainRegistry:
    """Tests for DomainRegistry class."""

    def test_empty_list_names(self) -> None:
        """New registry returns empty tuple."""
        registry = DomainRegistry()
        assert registry.list_names() == ()

    def test_register_and_get(self) -> None:
        """Register a domain and retrieve it by name."""
        registry = DomainRegistry()
        domain = FakeDomain("weather")
        registry.register(domain)

        result = registry.get("weather")
        assert result.config["name"] == "weather"

    def test_register_and_list_names(self) -> None:
        """Register two domains and list returns sorted tuple."""
        registry = DomainRegistry()
        registry.register(FakeDomain("weather"))
        registry.register(FakeDomain("covenant"))

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
        registry.register(FakeDomain("alpha"))
        registry.register(FakeDomain("beta"))

        with pytest.raises(KeyError, match="alpha, beta"):
            registry.get("missing")

    def test_register_duplicate_raises_value_error(self) -> None:
        """Registering same name twice raises ValueError."""
        registry = DomainRegistry()
        registry.register(FakeDomain("weather"))

        with pytest.raises(ValueError, match="Domain 'weather' already registered"):
            registry.register(FakeDomain("weather"))

    def test_list_names_sorted(self) -> None:
        """Names are returned in sorted order regardless of insertion."""
        registry = DomainRegistry()
        registry.register(FakeDomain("zulu"))
        registry.register(FakeDomain("alpha"))
        registry.register(FakeDomain("mike"))

        assert registry.list_names() == ("alpha", "mike", "zulu")

    def test_register_multiple_domains(self) -> None:
        """Register three domains and retrieve each."""
        registry = DomainRegistry()
        registry.register(FakeDomain("a"))
        registry.register(FakeDomain("b"))
        registry.register(FakeDomain("c"))

        assert registry.get("a").config["name"] == "a"
        assert registry.get("b").config["name"] == "b"
        assert registry.get("c").config["name"] == "c"

    def test_get_empty_registry_error_message(self) -> None:
        """KeyError on empty registry shows empty available list."""
        registry = DomainRegistry()

        with pytest.raises(KeyError, match="Available: "):
            registry.get("anything")
