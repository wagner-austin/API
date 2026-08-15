"""Test hooks for captioning service.

This module provides dependency injection hooks and unified Protocols for
captioning backends. Production code sets the hooks to real implementations
at startup. Tests set them to fakes.

The Protocols defined here are the single source of truth for the
OpenAI and Gemini client interfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from typing_extensions import TypedDict

# =============================================================================
# Caption Generator Protocol (for BLIP)
# =============================================================================


class CaptionGenerator(Protocol):
    """Protocol for image captioning."""

    def __call__(self, image_path: Path, trigger_word: str) -> str:
        """Generate a caption for an image.

        Args:
            image_path: Path to the image file.
            trigger_word: Trigger word to prepend to caption.

        Returns:
            Generated caption string.
        """
        ...


# =============================================================================
# OpenAI Protocols
# =============================================================================


class OpenAIImageUrlDict(TypedDict, total=True):
    """Image URL content for OpenAI vision."""

    url: str


class OpenAITextContentDict(TypedDict, total=True):
    """Text content item in OpenAI message."""

    type: str
    text: str


class OpenAIImageContentDict(TypedDict, total=True):
    """Image content item in OpenAI message."""

    type: str
    image_url: OpenAIImageUrlDict


class OpenAIMessageDict(TypedDict, total=True):
    """OpenAI message format."""

    role: str
    content: list[OpenAITextContentDict | OpenAIImageContentDict]


class OpenAIMessage(Protocol):
    """Protocol for OpenAI message object."""

    content: str | None


class OpenAIChoice(Protocol):
    """Protocol for OpenAI choice object."""

    message: OpenAIMessage


class OpenAICompletionResponse(Protocol):
    """Protocol for OpenAI completion response."""

    choices: list[OpenAIChoice]


class OpenAICompletions(Protocol):
    """Protocol for OpenAI completions interface."""

    def create(
        self,
        model: str,
        messages: list[OpenAIMessageDict],
        max_tokens: int,
    ) -> OpenAICompletionResponse:
        """Create a chat completion.

        Args:
            model: Model name.
            messages: List of message dicts.
            max_tokens: Maximum tokens to generate.

        Returns:
            Completion response.
        """
        ...


class OpenAIChat(Protocol):
    """Protocol for OpenAI chat interface."""

    completions: OpenAICompletions


class OpenAIClient(Protocol):
    """Protocol for OpenAI client."""

    chat: OpenAIChat


class OpenAIClientFactory(Protocol):
    """Protocol for OpenAI client factory."""

    def __call__(self, api_key: str) -> OpenAIClient:
        """Create OpenAI client.

        Args:
            api_key: API key.

        Returns:
            OpenAI client instance.
        """
        ...


class OpenAIModule(Protocol):
    """Protocol for openai module."""

    OpenAI: OpenAIClientFactory


# =============================================================================
# Gemini Protocols
# =============================================================================


class GeminiPart(Protocol):
    """Protocol for Gemini Part."""

    pass


class GeminiPartFactory(Protocol):
    """Protocol for Part factory."""

    def from_bytes(self, data: bytes, mime_type: str) -> GeminiPart:
        """Create part from bytes.

        Args:
            data: Image bytes.
            mime_type: MIME type.

        Returns:
            Part instance.
        """
        ...


class GeminiTypesModule(Protocol):
    """Protocol for google.genai.types module."""

    Part: GeminiPartFactory


class GeminiResponse(Protocol):
    """Protocol for Gemini response."""

    text: str


class GeminiModels(Protocol):
    """Protocol for Gemini models interface."""

    def generate_content(
        self,
        model: str,
        contents: list[str | GeminiPart],
    ) -> GeminiResponse:
        """Generate content.

        Args:
            model: Model name.
            contents: List of prompt and image parts.

        Returns:
            Generation response.
        """
        ...


class GeminiClient(Protocol):
    """Protocol for Gemini client."""

    models: GeminiModels


class GeminiClientFactory(Protocol):
    """Protocol for Gemini client factory."""

    def __call__(self, api_key: str) -> GeminiClient:
        """Create Gemini client.

        Args:
            api_key: API key.

        Returns:
            Gemini client instance.
        """
        ...


class GeminiModule(Protocol):
    """Protocol for google.genai module."""

    Client: GeminiClientFactory


# =============================================================================
# Caption Backend Protocol (for registry hook)
# =============================================================================


CaptionBackendTypeStr = Literal["blip", "gemini", "openai"]


class CaptionConfigDict(TypedDict, total=True):
    """Configuration for caption generation.

    Attributes:
        backend: Which caption backend to use.
        model_name: Model name/identifier for the backend.
        api_key: API key for API-based backends (Gemini, OpenAI).
            Empty string for local backends like BLIP.
    """

    backend: CaptionBackendTypeStr
    model_name: str
    api_key: str


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
        """
        ...

    @property
    def backend_type(self) -> CaptionBackendTypeStr:
        """Get the backend type identifier.

        Returns:
            Backend type string.
        """
        ...


class CaptionBackendFactory(Protocol):
    """Protocol for caption backend factory.

    Used by tests to inject fake backends into the registry.
    """

    def __call__(self, config: CaptionConfigDict) -> CaptionBackend:
        """Create a caption backend.

        Args:
            config: Caption configuration.

        Returns:
            Caption backend instance.
        """
        ...


# =============================================================================
# Hooks Container
# =============================================================================


# =============================================================================
# Production implementations
# =============================================================================


def _default_openai_client_factory(api_key: str) -> OpenAIClient:
    """Build the real OpenAI client.

    The module is imported here rather than at module scope so importing the
    hooks costs nothing until a caption is actually requested.

    Args:
        api_key: API key.

    Returns:
        OpenAI client instance.
    """
    openai_raw = __import__("openai", fromlist=["OpenAI"])
    openai_mod: OpenAIModule = openai_raw
    openai_cls: OpenAIClientFactory = openai_mod.OpenAI
    return openai_cls(api_key=api_key)


def _default_gemini_client_factory(api_key: str) -> GeminiClient:
    """Build the real Gemini client.

    Args:
        api_key: API key.

    Returns:
        Gemini client instance.
    """
    genai_raw = __import__("google.genai", fromlist=["Client"])
    genai_mod: GeminiModule = genai_raw
    client_cls: GeminiClientFactory = genai_mod.Client
    return client_cls(api_key=api_key)


class _DefaultGeminiPartFactory:
    """The real google.genai Part class, reached on each call.

    A class rather than a function because GeminiPartFactory is named by its
    from_bytes method, which is how google.genai spells it and how the
    captioner calls it.
    """

    def from_bytes(self, data: bytes, mime_type: str) -> GeminiPart:
        """Create part from bytes.

        Args:
            data: Image bytes.
            mime_type: MIME type.

        Returns:
            Part instance.
        """
        types_raw = __import__("google.genai.types", fromlist=["Part"])
        types_mod: GeminiTypesModule = types_raw
        return types_mod.Part.from_bytes(data=data, mime_type=mime_type)


class Hooks:
    """Container for captioning hooks.

    The Gemini and OpenAI hooks are bound to the real clients, so a caller
    reaches the vendor SDK without wiring anything; tests rebind them and
    reset_hooks() puts the real implementations back.

    Attributes:
        caption_generator: Hook for BLIP caption generation.
        caption_backend_factory: Hook for caption backend creation in registry.
        gemini_client_factory: Hook for Gemini client creation.
        gemini_part_factory: Hook for Gemini Part creation.
        openai_client_factory: Hook for OpenAI client creation.
    """

    caption_generator: CaptionGenerator | None = None
    caption_backend_factory: CaptionBackendFactory | None = None
    gemini_client_factory: GeminiClientFactory = _default_gemini_client_factory
    gemini_part_factory: GeminiPartFactory = _DefaultGeminiPartFactory()
    openai_client_factory: OpenAIClientFactory = _default_openai_client_factory


def reset_hooks() -> None:
    """Restore every hook to the implementation the container binds."""
    Hooks.caption_generator = None
    Hooks.caption_backend_factory = None
    Hooks.gemini_client_factory = _default_gemini_client_factory
    Hooks.gemini_part_factory = _DefaultGeminiPartFactory()
    Hooks.openai_client_factory = _default_openai_client_factory


__all__ = [
    "CaptionBackend",
    "CaptionBackendFactory",
    "CaptionBackendTypeStr",
    "CaptionConfigDict",
    "CaptionGenerator",
    "GeminiClient",
    "GeminiClientFactory",
    "GeminiModels",
    "GeminiModule",
    "GeminiPart",
    "GeminiPartFactory",
    "GeminiResponse",
    "GeminiTypesModule",
    "Hooks",
    "OpenAIChat",
    "OpenAIChoice",
    "OpenAIClient",
    "OpenAIClientFactory",
    "OpenAICompletionResponse",
    "OpenAICompletions",
    "OpenAIImageContentDict",
    "OpenAIImageUrlDict",
    "OpenAIMessage",
    "OpenAIMessageDict",
    "OpenAIModule",
    "OpenAITextContentDict",
    "reset_hooks",
]
