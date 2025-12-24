"""Service container for grandma-api dependency injection.

Provides a centralized container for service dependencies with Protocol-based DI.
Production code sets hooks to real implementations at startup; tests set them to fakes.
"""

from __future__ import annotations

from typing import Protocol

from platform_stt import OpenAISttClient
from platform_stt._test_hooks import STTClientProtocol

from grandma_api.config import GrandmaApiSettings


class STTClientFactoryProtocol(Protocol):
    """Protocol for STT client factory function."""

    def __call__(self, api_key: str) -> STTClientProtocol:
        """Create STT client with given API key.

        Args:
            api_key: OpenAI API key.

        Returns:
            STT client instance.
        """
        ...


def _default_stt_client_factory(api_key: str) -> STTClientProtocol:
    """Production implementation - creates real OpenAI STT client.

    Args:
        api_key: OpenAI API key.

    Returns:
        OpenAISttClient instance.
    """
    client: STTClientProtocol = OpenAISttClient(api_key=api_key)
    return client


class ServiceContainer:
    """Dependency injection container for grandma-api services.

    Provides centralized access to all service dependencies.
    Use from_settings() to create with production defaults,
    or construct directly for testing with custom factories.

    Attributes:
        settings: Application configuration settings.
        stt_client_factory: Factory for creating STT clients.
    """

    __slots__ = ("settings", "stt_client_factory")

    settings: GrandmaApiSettings
    stt_client_factory: STTClientFactoryProtocol

    def __init__(
        self,
        settings: GrandmaApiSettings,
        stt_client_factory: STTClientFactoryProtocol,
    ) -> None:
        """Initialize the service container.

        Args:
            settings: Application configuration settings.
            stt_client_factory: Factory for creating STT clients.
        """
        self.settings = settings
        self.stt_client_factory = stt_client_factory

    @classmethod
    def from_settings(cls, settings: GrandmaApiSettings) -> ServiceContainer:
        """Create a ServiceContainer with production defaults.

        Args:
            settings: Application configuration settings.

        Returns:
            ServiceContainer configured for production use.
        """
        return cls(
            settings=settings,
            stt_client_factory=_default_stt_client_factory,
        )

    def get_stt_client(self) -> STTClientProtocol:
        """Get an STT client configured with the API key from settings.

        Returns:
            STT client instance ready for use.
        """
        return self.stt_client_factory(self.settings["openai_api_key"])


__all__ = [
    "STTClientFactoryProtocol",
    "ServiceContainer",
    "_default_stt_client_factory",
]
