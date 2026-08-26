"""Tests for OpenAI captioning backend."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from art_trainer.core.services.captioning import _test_hooks
from art_trainer.core.services.captioning.backends import CaptionBackendError
from art_trainer.core.services.captioning.openai_backend import (
    DEFAULT_MODEL,
    SUPPORTED_FORMATS,
    OpenAICaptioner,
    _get_mime_type,
)


class FakeOpenAIMessage:
    """Fake OpenAI message for testing."""

    content: str | None

    def __init__(self, content: str | None) -> None:
        """Initialize fake message.

        Args:
            content: Message content.
        """
        self.content = content


class FakeOpenAIChoice:
    """Fake OpenAI choice for testing."""

    message: _test_hooks.OpenAIMessage

    def __init__(self, content: str | None) -> None:
        """Initialize fake choice.

        Args:
            content: Message content.
        """
        self.message = FakeOpenAIMessage(content)


class FakeOpenAICompletionResponse:
    """Fake OpenAI completion response for testing."""

    choices: list[_test_hooks.OpenAIChoice]

    def __init__(self, content: str | None) -> None:
        """Initialize fake response.

        Args:
            content: Message content.
        """
        self.choices = [FakeOpenAIChoice(content)]


class FakeOpenAICompletions:
    """Fake OpenAI completions interface for testing."""

    def __init__(self, response_content: str | None) -> None:
        """Initialize fake completions.

        Args:
            response_content: Content to return in response.
        """
        self._response_content = response_content

    def create(
        self,
        model: str,
        messages: list[_test_hooks.OpenAIMessageDict],
        max_tokens: int,
    ) -> _test_hooks.OpenAICompletionResponse:
        """Create fake completion.

        Args:
            model: Model name.
            messages: List of message dicts.
            max_tokens: Maximum tokens.

        Returns:
            Fake completion response.
        """
        return FakeOpenAICompletionResponse(self._response_content)


class FakeOpenAIChat:
    """Fake OpenAI chat interface for testing."""

    completions: _test_hooks.OpenAICompletions

    def __init__(self, response_content: str | None) -> None:
        """Initialize fake chat.

        Args:
            response_content: Content to return in response.
        """
        self.completions = FakeOpenAICompletions(response_content)


class FakeOpenAIClient:
    """Fake OpenAI client for testing."""

    chat: _test_hooks.OpenAIChat

    def __init__(self, response_content: str | None) -> None:
        """Initialize fake client.

        Args:
            response_content: Content to return in response.
        """
        self.chat = FakeOpenAIChat(response_content)


class FakeOpenAIClientFactory:
    """Fake OpenAI client factory for testing."""

    def __init__(self, response_content: str | None) -> None:
        """Initialize fake factory.

        Args:
            response_content: Content to return in response.
        """
        self._response_content = response_content

    def __call__(self, api_key: str) -> _test_hooks.OpenAIClient:
        """Create fake client.

        Args:
            api_key: API key (ignored in fake).

        Returns:
            Fake client instance.
        """
        return FakeOpenAIClient(self._response_content)


@pytest.fixture(autouse=True)
def reset_openai_hooks() -> None:
    """Reset hooks after each test."""
    _test_hooks.reset_hooks()


def test_openai_captioner_requires_api_key() -> None:
    """Test OpenAICaptioner raises ValueError for empty API key."""
    with pytest.raises(ValueError) as exc_info:
        OpenAICaptioner(model_name="gpt-4o", api_key="")

    assert "API key is required" in str(exc_info.value)


def test_openai_captioner_backend_type() -> None:
    """Test OpenAICaptioner backend_type property."""
    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")

    assert captioner.backend_type == "openai"


def test_openai_captioner_caption_file_not_found(tmp_path: Path) -> None:
    """Test OpenAICaptioner raises FileNotFoundError for missing file."""
    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")
    nonexistent = tmp_path / "nonexistent.png"

    with pytest.raises(FileNotFoundError) as exc_info:
        captioner.caption(nonexistent, "trigger")

    assert "nonexistent.png" in str(exc_info.value)


def test_openai_captioner_caption_unsupported_format(tmp_path: Path) -> None:
    """Test OpenAICaptioner raises error for unsupported format."""
    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")
    unsupported = tmp_path / "image.bmp"
    unsupported.write_bytes(b"fake bmp content")

    with pytest.raises(CaptionBackendError) as exc_info:
        captioner.caption(unsupported, "trigger")

    assert "Unsupported image format" in str(exc_info.value)


def test_get_mime_type_jpg() -> None:
    """Test _get_mime_type returns correct type for jpg."""
    assert _get_mime_type(".jpg") == "image/jpeg"


def test_get_mime_type_jpeg() -> None:
    """Test _get_mime_type returns correct type for jpeg."""
    assert _get_mime_type(".jpeg") == "image/jpeg"


def test_get_mime_type_png() -> None:
    """Test _get_mime_type returns correct type for png."""
    assert _get_mime_type(".png") == "image/png"


def test_get_mime_type_webp() -> None:
    """Test _get_mime_type returns correct type for webp."""
    assert _get_mime_type(".webp") == "image/webp"


def test_get_mime_type_gif() -> None:
    """Test _get_mime_type returns correct type for gif."""
    assert _get_mime_type(".gif") == "image/gif"


def test_get_mime_type_unknown_defaults_to_jpeg() -> None:
    """Test _get_mime_type defaults to jpeg for unknown."""
    assert _get_mime_type(".unknown") == "image/jpeg"


def test_supported_formats_contains_expected() -> None:
    """Test SUPPORTED_FORMATS contains expected formats."""
    assert ".jpg" in SUPPORTED_FORMATS
    assert ".jpeg" in SUPPORTED_FORMATS
    assert ".png" in SUPPORTED_FORMATS
    assert ".webp" in SUPPORTED_FORMATS
    assert ".gif" in SUPPORTED_FORMATS


def test_default_model_is_set() -> None:
    """Test DEFAULT_MODEL is set to a reasonable value."""
    assert DEFAULT_MODEL == "gpt-4o"


def test_openai_captioner_client_not_initialized() -> None:
    """Test client is None before first use."""
    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")

    assert captioner._client is None


def _create_test_image(tmp_path: Path) -> Path:
    """Create a valid test image.

    Args:
        tmp_path: Temporary directory.

    Returns:
        Path to test image.
    """
    image_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_path.write_bytes(buffer.getvalue())
    return image_path


def test_openai_captioner_caption_with_hook(tmp_path: Path) -> None:
    """Test OpenAICaptioner caption method using hooks."""
    _test_hooks.Hooks.openai_client_factory = FakeOpenAIClientFactory("A gray square image")

    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")
    image_path = _create_test_image(tmp_path)

    caption = captioner.caption(image_path, "test_trigger")

    assert caption == "test_trigger, A gray square image"


def test_openai_captioner_ensures_client_once(tmp_path: Path) -> None:
    """Test client is cached after first use."""
    _test_hooks.Hooks.openai_client_factory = FakeOpenAIClientFactory("Description")

    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")
    image_path = _create_test_image(tmp_path)

    # First call creates client
    captioner.caption(image_path, "trigger1")
    client1 = captioner._client

    # Second call reuses client
    captioner.caption(image_path, "trigger2")
    client2 = captioner._client

    assert client1 is client2


def test_openai_captioner_caption_strips_whitespace(tmp_path: Path) -> None:
    """Test caption strips whitespace from response."""
    _test_hooks.Hooks.openai_client_factory = FakeOpenAIClientFactory("  response with spaces  ")

    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")
    image_path = _create_test_image(tmp_path)

    caption = captioner.caption(image_path, "trigger")

    assert caption == "trigger, response with spaces"


def test_openai_captioner_caption_empty_response_raises(tmp_path: Path) -> None:
    """Test caption raises error when response is empty."""
    _test_hooks.Hooks.openai_client_factory = FakeOpenAIClientFactory(None)

    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")
    image_path = _create_test_image(tmp_path)

    with pytest.raises(CaptionBackendError) as exc_info:
        captioner.caption(image_path, "trigger")

    assert "empty response" in str(exc_info.value)


def test_openai_captioner_caption_jpg_image(tmp_path: Path) -> None:
    """Test caption works with JPG images."""
    _test_hooks.Hooks.openai_client_factory = FakeOpenAIClientFactory("JPG image")

    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")

    image_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    img.save(image_path, format="JPEG")

    caption = captioner.caption(image_path, "trigger")

    assert caption == "trigger, JPG image"


def test_openai_captioner_caption_webp_image(tmp_path: Path) -> None:
    """Test caption works with WebP images."""
    _test_hooks.Hooks.openai_client_factory = FakeOpenAIClientFactory("WebP image")

    captioner = OpenAICaptioner(model_name="gpt-4o", api_key="test-key")

    image_path = tmp_path / "test_image.webp"
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    img.save(image_path, format="WEBP")

    caption = captioner.caption(image_path, "trigger")

    assert caption == "trigger, WebP image"


def test_openai_captioner_integration_real_api(tmp_path: Path) -> None:
    """Integration test: OpenAICaptioner with real OpenAI API.

    This test calls the real OpenAI API without hooks.
    Requires OPENAI_API_KEY in settings.
    """
    from platform_core.config.art_trainer import load_settings

    settings = load_settings()
    api_key = settings["app"]["openai_api_key"]
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in settings")

    # No hooks set - uses real API
    captioner = OpenAICaptioner(model_name="gpt-4o-mini", api_key=api_key)

    # Create test image
    image_path = tmp_path / "test_real.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_path.write_bytes(buffer.getvalue())

    caption = captioner.caption(image_path, "test_trigger")

    # Verify caption has expected format
    assert caption.startswith("test_trigger, ")
    assert len(caption) > len("test_trigger, ")
