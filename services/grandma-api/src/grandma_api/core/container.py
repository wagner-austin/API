"""Service container for grandma-api dependency injection.

Provides a centralized container for service dependencies with Protocol-based DI.
Production code sets hooks to real implementations at startup; tests set them to fakes.
"""

from __future__ import annotations

from typing import Protocol

from platform_langid import create_detector, default_detector_config
from platform_langid._test_hooks import SpokenLanguageDetectorProtocol
from platform_langid.types import DetectorConfig
from platform_stt import OpenAISttClient
from platform_stt._test_hooks import STTClientProtocol
from platform_translate import Translator, TranslatorConfig
from platform_translate.backends.protocol import TranslationBackendProtocol

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


class LangIdDetectorFactoryProtocol(Protocol):
    """Protocol for language ID detector factory function."""

    def __call__(self, config: DetectorConfig) -> SpokenLanguageDetectorProtocol:
        """Create language ID detector with given config.

        Args:
            config: Detector configuration.

        Returns:
            Detector instance.
        """
        ...


class TranslatorFactoryProtocol(Protocol):
    """Protocol for translator factory function."""

    def __call__(self, config: TranslatorConfig) -> TranslationBackendProtocol:
        """Create translator with given config.

        Args:
            config: Translator configuration.

        Returns:
            Translator instance.
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


def _default_langid_detector_factory(config: DetectorConfig) -> SpokenLanguageDetectorProtocol:
    """Production implementation - creates real language ID detector.

    Args:
        config: Detector configuration.

    Returns:
        SpokenLanguageDetector instance.
    """
    detector: SpokenLanguageDetectorProtocol = create_detector(config)
    return detector


def _default_translator_factory(config: TranslatorConfig) -> TranslationBackendProtocol:
    """Production implementation - creates real translator.

    Args:
        config: Translator configuration.

    Returns:
        Translator instance.
    """
    translator: TranslationBackendProtocol = Translator(config)
    return translator


class ServiceContainer:
    """Dependency injection container for grandma-api services.

    Provides centralized access to all service dependencies.
    Use from_settings() to create with production defaults,
    or construct directly for testing with custom factories.

    Attributes:
        settings: Application configuration settings.
        stt_client_factory: Factory for creating STT clients.
        langid_detector_factory: Factory for creating language ID detectors.
        translator_factory: Factory for creating translators.
    """

    __slots__ = (
        "langid_detector_factory",
        "settings",
        "stt_client_factory",
        "translator_factory",
    )

    settings: GrandmaApiSettings
    stt_client_factory: STTClientFactoryProtocol
    langid_detector_factory: LangIdDetectorFactoryProtocol
    translator_factory: TranslatorFactoryProtocol

    def __init__(
        self,
        settings: GrandmaApiSettings,
        stt_client_factory: STTClientFactoryProtocol,
        langid_detector_factory: LangIdDetectorFactoryProtocol,
        translator_factory: TranslatorFactoryProtocol,
    ) -> None:
        """Initialize the service container.

        Args:
            settings: Application configuration settings.
            stt_client_factory: Factory for creating STT clients.
            langid_detector_factory: Factory for creating language ID detectors.
            translator_factory: Factory for creating translators.
        """
        self.settings = settings
        self.stt_client_factory = stt_client_factory
        self.langid_detector_factory = langid_detector_factory
        self.translator_factory = translator_factory

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
            langid_detector_factory=_default_langid_detector_factory,
            translator_factory=_default_translator_factory,
        )

    def get_stt_client(self) -> STTClientProtocol:
        """Get an STT client configured with the API key from settings.

        Returns:
            STT client instance ready for use.
        """
        return self.stt_client_factory(self.settings["openai_api_key"])

    def get_langid_detector(self) -> SpokenLanguageDetectorProtocol:
        """Get a language ID detector with default configuration.

        Returns:
            Language ID detector instance ready for use.
        """
        config = default_detector_config()
        return self.langid_detector_factory(config)

    def get_translator(self) -> TranslationBackendProtocol:
        """Get a translator configured with the API key from settings.

        Returns:
            Translator instance ready for use.
        """
        config = TranslatorConfig(
            backend="openai",
            api_key=self.settings["openai_api_key"],
            model="gpt-4o-mini",
        )
        return self.translator_factory(config)


__all__ = [
    "LangIdDetectorFactoryProtocol",
    "STTClientFactoryProtocol",
    "ServiceContainer",
    "TranslatorFactoryProtocol",
    "_default_langid_detector_factory",
    "_default_stt_client_factory",
    "_default_translator_factory",
]
