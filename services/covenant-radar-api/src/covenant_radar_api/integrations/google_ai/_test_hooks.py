"""Test hooks for Google AI (Gemini) integration.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Protocol

# =============================================================================
# Gemini Client Protocol
# =============================================================================


class GeminiClientProtocol(Protocol):
    """Protocol for Gemini text generation client.

    This protocol defines the interface for generating text using Google's
    Gemini LLM. Production code uses the real Google SDK; tests inject fakes.
    """

    def generate_content(
        self,
        model: str,
        contents: str,
    ) -> str:
        """Generate text content using Gemini.

        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash").
            contents: Text prompt to send to the model.

        Returns:
            Generated text response from the model.

        Raises:
            GeminiError: If the API call fails.
        """
        ...

    def count_tokens(
        self,
        model: str,
        contents: str,
    ) -> tuple[int, int]:
        """Count tokens for input content.

        Args:
            model: Gemini model name.
            contents: Text content to count tokens for.

        Returns:
            Tuple of (input_tokens, estimated_output_tokens).
            Note: output tokens are estimated as 0 for input-only counting.

        Raises:
            GeminiError: If the API call fails.
        """
        ...


# =============================================================================
# Real Implementation
# =============================================================================


class RealGeminiClient:
    """Real Gemini client using the google-genai SDK.

    This implementation uses the official Google Gen AI Python SDK to
    interact with Gemini models.

    Reference: https://github.com/googleapis/python-genai
    """

    def __init__(self, api_key: str) -> None:
        """Initialize the real Gemini client.

        Args:
            api_key: Google AI API key for authentication.
        """
        self._api_key = api_key
        self._client = _create_genai_client(api_key)

    def generate_content(
        self,
        model: str,
        contents: str,
    ) -> str:
        """Generate text content using Gemini.

        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash").
            contents: Text prompt to send to the model.

        Returns:
            Generated text response from the model.

        Raises:
            GeminiError: If the API call fails or response is empty.
        """
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
        )
        text = response.text
        if text is None:
            msg = "Gemini returned empty response"
            raise GeminiError(msg)
        return text

    def count_tokens(
        self,
        model: str,
        contents: str,
    ) -> tuple[int, int]:
        """Count tokens for input content.

        Args:
            model: Gemini model name.
            contents: Text content to count tokens for.

        Returns:
            Tuple of (input_tokens, 0).
            Output tokens are 0 for input-only counting.

        Raises:
            GeminiError: If the API call fails.
        """
        response = self._client.models.count_tokens(
            model=model,
            contents=contents,
        )
        input_tokens: int = response.total_tokens
        return (input_tokens, 0)


def _create_genai_client(api_key: str) -> _GenaiClientProtocol:
    """Create a google-genai Client instance.

    Uses dynamic import to avoid mypy errors from untyped google-genai package.

    Args:
        api_key: Google AI API key.

    Returns:
        Configured genai.Client instance.
    """
    genai_module = __import__("google.genai", fromlist=["Client"])
    client_class: type[_GenaiClientProtocol] = genai_module.Client
    client: _GenaiClientProtocol = client_class(api_key=api_key)
    return client


class _GenaiClientProtocol(Protocol):
    """Protocol for google.genai.Client to avoid Any types."""

    def __init__(self, api_key: str) -> None:
        """Initialize with API key."""
        ...

    @property
    def models(self) -> _ModelsProtocol:
        """Access models API."""
        ...


class _ModelsProtocol(Protocol):
    """Protocol for google.genai.Client.models."""

    def generate_content(
        self,
        model: str,
        contents: str,
    ) -> _GenerateContentResponseProtocol:
        """Generate content."""
        ...

    def count_tokens(
        self,
        model: str,
        contents: str,
    ) -> _CountTokensResponseProtocol:
        """Count tokens."""
        ...


class _GenerateContentResponseProtocol(Protocol):
    """Protocol for generate_content response."""

    @property
    def text(self) -> str | None:
        """Generated text."""
        ...


class _CountTokensResponseProtocol(Protocol):
    """Protocol for count_tokens response."""

    @property
    def total_tokens(self) -> int:
        """Total token count."""
        ...


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeGeminiClient:
    """Fake Gemini client for testing.

    Records all calls and returns configurable responses.
    """

    def __init__(self) -> None:
        """Initialize the fake client with empty call history."""
        self.generate_calls: list[tuple[str, str]] = []
        self.count_calls: list[tuple[str, str]] = []
        self.next_response: str = "Fake Gemini response"
        self.next_token_count: int = 100
        self.should_fail: bool = False
        self.fail_message: str = "Fake Gemini error"

    def generate_content(
        self,
        model: str,
        contents: str,
    ) -> str:
        """Record call and return configured response.

        Args:
            model: Gemini model name.
            contents: Text prompt.

        Returns:
            Configured fake response.

        Raises:
            GeminiError: If should_fail is True.
        """
        self.generate_calls.append((model, contents))
        if self.should_fail:
            raise GeminiError(self.fail_message)
        return self.next_response

    def count_tokens(
        self,
        model: str,
        contents: str,
    ) -> tuple[int, int]:
        """Record call and return configured token count.

        Args:
            model: Gemini model name.
            contents: Text content.

        Returns:
            Tuple of (configured_count, 0).

        Raises:
            GeminiError: If should_fail is True.
        """
        self.count_calls.append((model, contents))
        if self.should_fail:
            raise GeminiError(self.fail_message)
        return (self.next_token_count, 0)


# =============================================================================
# Error Types
# =============================================================================


class GeminiError(Exception):
    """Error from Gemini API call.

    Raised when the Gemini API returns an error or empty response.
    """

    pass


# =============================================================================
# Factory Hook
# =============================================================================


class GeminiClientFactory(Protocol):
    """Protocol for creating Gemini clients."""

    def __call__(self, api_key: str) -> GeminiClientProtocol:
        """Create a Gemini client.

        Args:
            api_key: Google AI API key.

        Returns:
            GeminiClientProtocol implementation.
        """
        ...


def _real_gemini_client_factory(api_key: str) -> GeminiClientProtocol:
    """Create a real Gemini client.

    Args:
        api_key: Google AI API key.

    Returns:
        RealGeminiClient instance.
    """
    return RealGeminiClient(api_key)


# Module-level injectable factory for testing.
# Production code calls this; tests override before calling create_gemini_client.
gemini_client_factory: GeminiClientFactory = _real_gemini_client_factory


def use_fake_gemini() -> FakeGeminiClient:
    """Set up fake Gemini client for testing.

    Call this in test setup to inject a fake client.

    Returns:
        The FakeGeminiClient that will be returned by the factory.

    Example:
        fake = use_fake_gemini()
        fake.next_response = "Custom response"
        client = create_gemini_client(config)
        # client will use fake internally
    """
    global gemini_client_factory
    fake = FakeGeminiClient()

    def fake_factory(api_key: str) -> GeminiClientProtocol:
        return fake

    gemini_client_factory = fake_factory
    return fake


def use_real_gemini() -> None:
    """Restore real Gemini client factory.

    Call this in test teardown to restore production behavior.
    """
    global gemini_client_factory
    gemini_client_factory = _real_gemini_client_factory


__all__ = [
    "FakeGeminiClient",
    "GeminiClientFactory",
    "GeminiClientProtocol",
    "GeminiError",
    "RealGeminiClient",
    "gemini_client_factory",
    "use_fake_gemini",
    "use_real_gemini",
]
