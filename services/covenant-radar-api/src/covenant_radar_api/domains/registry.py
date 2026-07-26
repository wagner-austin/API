"""Registry for pluggable domain implementations.

Domains are registered by name at startup and constructed on lookup. Holding
factories rather than instances matters once a domain needs configuration to
build: weather reads a fitted seasonal state and a station map off disk, so
registering it eagerly would demand those files from every deployment,
including one running only esports.

Thread-safe for reads after construction (no mutations after init).

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from typing import Protocol

from .protocols import DomainProtocol


class DomainFactory(Protocol):
    """Factory protocol to construct a domain implementation."""

    def __call__(self) -> DomainProtocol:
        """Build the domain.

        Returns:
            A domain satisfying DomainProtocol.
        """
        ...


class DomainRegistry:
    """Registry of available domain implementations.

    Domains are registered by name and built when requested, so registering
    a domain costs nothing until it is selected.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._factories: dict[str, DomainFactory] = {}

    def register(self, name: str, factory: DomainFactory) -> None:
        """Register a domain factory under a name.

        Args:
            name: Domain identifier callers select by.
            factory: Callable building the domain when it is requested.

        Raises:
            ValueError: If a domain with the same name is already registered.
        """
        if name in self._factories:
            raise ValueError(f"Domain '{name}' already registered")
        self._factories[name] = factory

    def get(self, name: str) -> DomainProtocol:
        """Build and return a registered domain by name.

        The factory runs here, so any configuration a domain needs is read
        only when that domain is the one being used.

        Args:
            name: Domain identifier.

        Returns:
            Domain satisfying DomainProtocol.

        Raises:
            KeyError: If the domain name is not registered.
        """
        if name not in self._factories:
            available: str = ", ".join(sorted(self._factories.keys()))
            raise KeyError(f"Domain '{name}' not found. Available: {available}")
        return self._factories[name]()

    def list_names(self) -> tuple[str, ...]:
        """List all registered domain names in sorted order.

        Returns:
            Sorted tuple of domain name strings.
        """
        return tuple(sorted(self._factories.keys()))


__all__ = [
    "DomainFactory",
    "DomainRegistry",
]
