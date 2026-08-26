"""Tests for Gemini captioning backend."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from art_trainer.core.services.captioning import _test_hooks
from art_trainer.core.services.captioning.backends import CaptionBackendError
from art_trainer.core.services.captioning.gemini_backend import (
    DEFAULT_MODEL,
    SUPPORTED_FORMATS,
    GeminiCaptioner,
    _get_mime_type,
)


class FakeGeminiPart:
    """Fake Gemini Part for testing."""

    pass


class FakeGeminiPartFactory:
    """Fake Gemini Part factory for testing."""

    def from_bytes(self, data: bytes, mime_type: str) -> _test_hooks.GeminiPart:
        """Create fake part from bytes.

        Args:
            data: Image bytes.
            mime_type: MIME type.

        Returns:
            Fake part instance.
        """
        return FakeGeminiPart()


class FakeGeminiResponse:
    """Fake Gemini response for testing."""

    text: str

    def __init__(self, text: str) -> None:
        """Initialize fake response.

        Args:
            text: Response text.
        """
        self.text = text


class FakeGeminiModels:
    """Fake Gemini models interface for testing."""

    def __init__(self, response_text: str) -> None:
        """Initialize fake models.

        Args:
            response_text: Text to return from generate_content.
        """
        self._response_text = response_text

    def generate_content(
        self,
        model: str,
        contents: list[str | _test_hooks.GeminiPart],
    ) -> _test_hooks.GeminiResponse:
        """Generate fake content.

        Args:
            model: Model name.
            contents: List of prompt and image parts.

        Returns:
            Fake response.
        """
        return FakeGeminiResponse(self._response_text)


class FakeGeminiClient:
    """Fake Gemini client for testing."""

    models: _test_hooks.GeminiModels

    def __init__(self, response_text: str) -> None:
        """Initialize fake client.

        Args:
            response_text: Text to return from generate_content.
        """
        self.models = FakeGeminiModels(response_text)


class FakeGeminiClientFactory:
    """Fake Gemini client factory for testing."""

    def __init__(self, response_text: str) -> None:
        """Initialize fake factory.

        Args:
            response_text: Text to return from generate_content.
        """
        self._response_text = response_text

    def __call__(self, api_key: str) -> _test_hooks.GeminiClient:
        """Create fake client.

        Args:
            api_key: API key (ignored in fake).

        Returns:
            Fake client instance.
        """
        return FakeGeminiClient(self._response_text)


@pytest.fixture(autouse=True)
def reset_gemini_hooks() -> None:
    """Reset hooks after each test."""
    _test_hooks.reset_hooks()


def test_gemini_captioner_requires_api_key() -> None:
    """Test GeminiCaptioner raises ValueError for empty API key."""
    with pytest.raises(ValueError) as exc_info:
        GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="")

    assert "API key is required" in str(exc_info.value)


def test_gemini_captioner_backend_type() -> None:
    """Test GeminiCaptioner backend_type property."""
    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")

    assert captioner.backend_type == "gemini"


def test_gemini_captioner_caption_file_not_found(tmp_path: Path) -> None:
    """Test GeminiCaptioner raises FileNotFoundError for missing file."""
    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")
    nonexistent = tmp_path / "nonexistent.png"

    with pytest.raises(FileNotFoundError) as exc_info:
        captioner.caption(nonexistent, "trigger")

    assert "nonexistent.png" in str(exc_info.value)


def test_gemini_captioner_caption_unsupported_format(tmp_path: Path) -> None:
    """Test GeminiCaptioner raises error for unsupported format."""
    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")
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
    assert DEFAULT_MODEL == "gemini-2.5-flash"


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


def test_gemini_captioner_client_not_initialized() -> None:
    """Test client is None before first use."""
    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")

    assert captioner._client is None


def test_gemini_captioner_caption_with_hook(tmp_path: Path) -> None:
    """Test GeminiCaptioner caption method using hooks."""
    # Set up hooks
    _test_hooks.Hooks.gemini_client_factory = FakeGeminiClientFactory("A gray square image")
    _test_hooks.Hooks.gemini_part_factory = FakeGeminiPartFactory()

    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")
    image_path = _create_test_image(tmp_path)

    caption = captioner.caption(image_path, "test_trigger")

    assert caption == "test_trigger, A gray square image"


def test_gemini_captioner_ensures_client_once(tmp_path: Path) -> None:
    """Test client is cached after first use."""
    _test_hooks.Hooks.gemini_client_factory = FakeGeminiClientFactory("Description")
    _test_hooks.Hooks.gemini_part_factory = FakeGeminiPartFactory()

    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")
    image_path = _create_test_image(tmp_path)

    # First call creates client
    captioner.caption(image_path, "trigger1")
    client1 = captioner._client

    # Second call reuses client
    captioner.caption(image_path, "trigger2")
    client2 = captioner._client

    assert client1 is client2


def test_gemini_captioner_caption_strips_whitespace(tmp_path: Path) -> None:
    """Test caption strips whitespace from response."""
    _test_hooks.Hooks.gemini_client_factory = FakeGeminiClientFactory("  response with spaces  ")
    _test_hooks.Hooks.gemini_part_factory = FakeGeminiPartFactory()

    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")
    image_path = _create_test_image(tmp_path)

    caption = captioner.caption(image_path, "trigger")

    assert caption == "trigger, response with spaces"


def test_gemini_captioner_caption_jpg_image(tmp_path: Path) -> None:
    """Test caption works with JPG images."""
    _test_hooks.Hooks.gemini_client_factory = FakeGeminiClientFactory("JPG image")
    _test_hooks.Hooks.gemini_part_factory = FakeGeminiPartFactory()

    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")

    image_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    img.save(image_path, format="JPEG")

    caption = captioner.caption(image_path, "trigger")

    assert caption == "trigger, JPG image"


def test_gemini_captioner_caption_webp_image(tmp_path: Path) -> None:
    """Test caption works with WebP images."""
    _test_hooks.Hooks.gemini_client_factory = FakeGeminiClientFactory("WebP image")
    _test_hooks.Hooks.gemini_part_factory = FakeGeminiPartFactory()

    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="test-key")

    image_path = tmp_path / "test_image.webp"
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    img.save(image_path, format="WEBP")

    caption = captioner.caption(image_path, "trigger")

    assert caption == "trigger, WebP image"


def test_gemini_captioner_real_imports_client(tmp_path: Path) -> None:
    """Test GeminiCaptioner imports real google.genai.Client without hooks.

    This test verifies the import path for the real Gemini client works.
    Uses an invalid API key so the API call fails after imports succeed.
    The google-genai library raises an error for invalid API keys.
    """
    # No hooks set - uses real imports
    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key="invalid-test-key")

    # Create test image
    image_path = _create_test_image(tmp_path)

    # Call caption - imports succeed but API call fails with invalid key
    # google.genai raises google.genai.errors.ClientError for invalid API key
    with pytest.raises(Exception) as exc_info:
        captioner.caption(image_path, "test_trigger")

    # Verify error is from the API client (not import error)
    # Import errors would be ModuleNotFoundError/ImportError
    error_type = type(exc_info.value).__name__
    assert error_type not in ("ModuleNotFoundError", "ImportError")


def test_gemini_captioner_integration_real_api(tmp_path: Path) -> None:
    """Integration test: GeminiCaptioner with real Gemini API.

    This test calls the real Gemini API without hooks.
    Requires GEMINI_API_KEY in settings.
    """
    from platform_core.config.art_trainer import load_settings

    settings = load_settings()
    api_key = settings["app"]["gemini_api_key"]
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set in settings")

    # No hooks set - uses real API
    captioner = GeminiCaptioner(model_name=DEFAULT_MODEL, api_key=api_key)

    # Create test image
    image_path = tmp_path / "test_real.png"
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_path.write_bytes(buffer.getvalue())

    caption = captioner.caption(image_path, "test_trigger")

    # Verify caption has expected format
    assert caption.startswith("test_trigger, ")
    assert len(caption) > len("test_trigger, ")
