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


class CaptionBackendRegistry:
    """Registry for caption backends.

    Manages creation and caching of caption backend instances.
    """

    _backends: dict[str, CaptionBackend]

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._backends = {}

    def get_backend(self, config: CaptionConfig) -> CaptionBackend:
        """Get or create a caption backend.

        Args:
            config: Caption configuration.

        Returns:
            Caption backend instance.

        Raises:
            ValueError: If backend type is not supported.
        """
        from art_trainer.core.services.captioning._test_hooks import Hooks

        backend_type = config["backend"]
        cache_key = f"{backend_type}:{config['model_name']}"

        if cache_key in self._backends:
            return self._backends[cache_key]

        backend: CaptionBackend

        # Check for test hook first
        if Hooks.caption_backend_factory is not None:
            from art_trainer.core.services.captioning._test_hooks import CaptionConfigDict

            hook_config: CaptionConfigDict = {
                "backend": config["backend"],
                "model_name": config["model_name"],
                "api_key": config["api_key"],
            }
            backend = Hooks.caption_backend_factory(hook_config)
        elif backend_type == "blip":
            backend = self._create_blip_backend(config["model_name"])
        elif backend_type == "gemini":
            backend = self._create_gemini_backend(config["model_name"], config["api_key"])
        else:
            backend = self._create_openai_backend(config["model_name"], config["api_key"])

        self._backends[cache_key] = backend
        return backend

    def _create_blip_backend(self, model_name: str) -> CaptionBackend:
        """Create BLIP backend.

        Args:
            model_name: HuggingFace model name.

        Returns:
            BLIP caption backend.
        """
        from art_trainer.core.services.captioning.blip_model import BlipCaptioner

        return _BlipBackendAdapter(BlipCaptioner.get_instance(model_name))

    def _create_gemini_backend(self, model_name: str, api_key: str) -> CaptionBackend:
        """Create Gemini backend.

        Args:
            model_name: Gemini model name (e.g., "gemini-pro-vision").
            api_key: Gemini API key.

        Returns:
            Gemini caption backend.
        """
        from art_trainer.core.services.captioning.gemini_backend import GeminiCaptioner

        return GeminiCaptioner(model_name, api_key)

    def _create_openai_backend(self, model_name: str, api_key: str) -> CaptionBackend:
        """Create OpenAI backend.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4-vision-preview").
            api_key: OpenAI API key.

        Returns:
            OpenAI caption backend.
        """
        from art_trainer.core.services.captioning.openai_backend import OpenAICaptioner

        return OpenAICaptioner(model_name, api_key)


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
    "CaptionBackendRegistry",
    "CaptionBackendType",
    "CaptionConfig",
    "get_caption_registry",
    "reset_caption_registry",
]
