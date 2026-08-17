"""Caption backend protocol and registry.

This module defines the protocol for caption backends and provides
a registry for managing different captioning implementations.
The design allows for easy addition of new backends like BLIP,
Gemini, GPT-4V, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from typing_extensions import TypedDict

CaptionBackendType = Literal["blip", "gemini", "openai"]


class CaptionBackend(Protocol):
    """Protocol for caption backends.

    All caption backends must implement this interface.
    """

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate caption for an image.

        Args:
            image_path: Path to the image file.
            trigger_word: Trigger word to prepend to caption.

        Returns:
            Generated caption with trigger word prefix.

        Raises:
            FileNotFoundError: If image_path does not exist.
            CaptionBackendError: If caption generation fails.
        """
        ...

    @property
    def backend_type(self) -> CaptionBackendType:
        """Get the backend type identifier.

        Returns:
            Backend type string.
        """
        ...


class CaptionBackendError(Exception):
    """Error from caption backend."""

    pass


class CaptionConfig(TypedDict, total=True):
    """Configuration for caption generation.

    Attributes:
        backend: Which caption backend to use.
        model_name: Model name/identifier for the backend.
        api_key: API key for API-based backends (Gemini, OpenAI).
            Empty string for local backends like BLIP.
    """

    backend: CaptionBackendType
    model_name: str
    api_key: str


class CaptionBackendFactoryProtocol(Protocol):
    """Protocol for a factory that builds one caption backend."""

    def __call__(self, config: CaptionConfig) -> CaptionBackend:
        """Build the backend the config names.

        Args:
            config: Caption configuration.

        Returns:
            Caption backend instance.
        """
        ...


def _create_blip_backend(config: CaptionConfig) -> CaptionBackend:
    """Create the BLIP backend.

    Args:
        config: Caption configuration; only model_name is used.

    Returns:
        BLIP caption backend.
    """
    from art_trainer.core.services.captioning.blip_model import BlipCaptioner

    return _BlipBackendAdapter(BlipCaptioner.get_instance(config["model_name"]))


def _create_gemini_backend(config: CaptionConfig) -> CaptionBackend:
    """Create the Gemini backend.

    Args:
        config: Caption configuration.

    Returns:
        Gemini caption backend.
    """
    from art_trainer.core.services.captioning.gemini_backend import GeminiCaptioner

    return GeminiCaptioner(config["model_name"], config["api_key"])


def _create_openai_backend(config: CaptionConfig) -> CaptionBackend:
    """Create the OpenAI backend.

    Args:
        config: Caption configuration.

    Returns:
        OpenAI caption backend.
    """
    from art_trainer.core.services.captioning.openai_backend import OpenAICaptioner

    return OpenAICaptioner(config["model_name"], config["api_key"])


class CaptionBackendRegistry:
    """Registry for caption backends.

    Holds one factory per backend type and caches what they build. A test
    substitutes a backend by registering its factory, which is why there is
    no hook here to short-circuit the lookup.
    """

    _backends: dict[str, CaptionBackend]
    _factories: dict[CaptionBackendType, CaptionBackendFactoryProtocol]

    def __init__(self) -> None:
        """Initialize with the real factory for every backend type."""
        self._backends = {}
        self._factories = {
            "blip": _create_blip_backend,
            "gemini": _create_gemini_backend,
            "openai": _create_openai_backend,
        }

    def register(
        self, backend_type: CaptionBackendType, factory: CaptionBackendFactoryProtocol
    ) -> None:
        """Replace the factory for one backend type.

        Any instance this registry already cached for that type is dropped,
        so the next get_backend call goes through the new factory.

        Args:
            backend_type: Type whose factory to replace.
            factory: Factory to use from now on.
        """
        self._factories[backend_type] = factory
        prefix = f"{backend_type}:"
        self._backends = {
            key: value for key, value in self._backends.items() if not key.startswith(prefix)
        }

    def get_backend(self, config: CaptionConfig) -> CaptionBackend:
        """Get or create a caption backend.

        Args:
            config: Caption configuration.

        Returns:
            Caption backend instance.
        """
        cache_key = f"{config['backend']}:{config['model_name']}"
        cached = self._backends.get(cache_key)
        if cached is not None:
            return cached
        backend = self._factories[config["backend"]](config)
        self._backends[cache_key] = backend
        return backend


class _BlipCaptionerProto(Protocol):
    """Protocol for BlipCaptioner interface."""

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate caption for an image.

        Args:
            image_path: Path to the image file.
            trigger_word: Trigger word to prepend to caption.

        Returns:
            Generated caption with trigger word prefix.
        """
        ...


class _BlipBackendAdapter:
    """Adapter to make BlipCaptioner conform to CaptionBackend protocol."""

    def __init__(self, captioner: _BlipCaptionerProto) -> None:
        """Initialize adapter.

        Args:
            captioner: BlipCaptioner instance.
        """
        self._captioner = captioner

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate caption for an image.

        Args:
            image_path: Path to the image file.
            trigger_word: Trigger word to prepend to caption.

        Returns:
            Generated caption with trigger word prefix.
        """
        return self._captioner.caption(image_path, trigger_word)

    @property
    def backend_type(self) -> CaptionBackendType:
        """Get the backend type identifier.

        Returns:
            Backend type string.
        """
        return "blip"


# Singleton registry instance
_registry: CaptionBackendRegistry | None = None


def get_caption_registry() -> CaptionBackendRegistry:
    """Get the global caption backend registry.

    Returns:
        Caption backend registry singleton.
    """
    global _registry
    if _registry is None:
        _registry = CaptionBackendRegistry()
    return _registry


def reset_caption_registry() -> None:
    """Reset the global caption backend registry for testing."""
    global _registry
    _registry = None


__all__ = [
    "CaptionBackend",
    "CaptionBackendError",
    "CaptionBackendFactoryProtocol",
    "CaptionBackendRegistry",
    "CaptionBackendType",
    "CaptionConfig",
    "get_caption_registry",
    "reset_caption_registry",
]
