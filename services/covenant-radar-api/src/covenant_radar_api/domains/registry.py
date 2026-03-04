"""Registry for pluggable domain implementations.

Domains are registered at startup and selected by configuration.
Thread-safe for reads after construction (no mutations after init).

Strict typing: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

from .protocols import DomainProtocol


class DomainRegistry:
    """Registry of available domain implementations.

    Domains are registered by name. After construction and registration,
    the registry is read-only and thread-safe for concurrent lookups.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._domains: dict[str, DomainProtocol] = {}

    def register(self, domain: DomainProtocol) -> None:
        """Register a domain implementation.

        Args:
            domain: Domain to register. Name is read from domain.config.

        Raises:
            ValueError: If a domain with the same name is already registered.
        """
        name: str = domain.config["name"]
        if name in self._domains:
            raise ValueError(f"Domain '{name}' already registered")
        self._domains[name] = domain

    def get(self, name: str) -> DomainProtocol:
        """Get a registered domain by name.

        Args:
            name: Domain identifier.

        Returns:
            Registered DomainProtocol implementation.

        Raises:
            KeyError: If domain name is not registered.
        """
        if name not in self._domains:
            available: str = ", ".join(sorted(self._domains.keys()))
            raise KeyError(f"Domain '{name}' not found. Available: {available}")
        return self._domains[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered domain names in sorted order.

        Returns:
            Sorted tuple of domain name strings.
        """
        return tuple(sorted(self._domains.keys()))


__all__ = [
    "DomainRegistry",
]
