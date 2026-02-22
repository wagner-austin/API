"""Tests for grandma_api.api.routes.translate module."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_stt import VerboseResponse, VerboseSegment

from grandma_api.api.main import create_app
from grandma_api.config import GrandmaApiSettings

from .conftest import generate_test_wav, make_test_container, set_fake_env


def _make_test_settings() -> GrandmaApiSettings:
    """Create test settings."""
    return GrandmaApiSettings(
        openai_api_key="sk-test",
        api_token="test-token",
        port=8080,
        log_level="INFO",
        log_format="json",
    )


def test_translate_success() -> None:
    """Test successful translation with real WAV audio."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    source_text = "Xin chào bà"
    translated_text = "Hello grandmother"
    fake_response = VerboseResponse(
        text=source_text,
        language="vi",
        segments=[VerboseSegment(text=source_text, start=0.0, end=2.0)],
    )
    settings = _make_test_settings()
    container, fake_client, _, _ = make_test_container(
        settings, fake_response, translated_text=translated_text
    )

    app = create_app(settings, container=container)
    client = TestClient(app)

    audio_content = generate_test_wav()
    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": ("test.wav", audio_content, "audio/wav")},
    )

    assert response.status_code == 200
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("text") == translated_text
    assert fake_client.call_count == 1


def test_translate_invalid_token() -> None:
    """Test translation with invalid token returns 401."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, fake_client, _, _ = make_test_container(settings)

    app = create_app(settings, container=container)
    client = TestClient(app)

    audio_content = generate_test_wav()
    response = client.post(
        "/translate",
        data={"token": "wrong-token"},
        files={"audio": ("test.wav", audio_content, "audio/wav")},
    )

    assert response.status_code == 401
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("message") == "Invalid token"
    assert body.get("code") == "UNAUTHORIZED"
    assert fake_client.call_count == 0


def test_translate_empty_audio() -> None:
    """Test translation with empty audio returns 400."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, fake_client, _, _ = make_test_container(settings)

    app = create_app(settings, container=container)
    client = TestClient(app)

    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": ("test.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("message") == "No audio file provided"
    assert body.get("code") == "INVALID_INPUT"
    assert fake_client.call_count == 0


def test_translate_no_filename() -> None:
    """Test translation works when filename is not provided."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, fake_client, _, _ = make_test_container(settings)

    app = create_app(settings, container=container)
    client = TestClient(app)

    audio_content = generate_test_wav()
    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": audio_content},
    )

    assert response.status_code == 200
    assert fake_client.call_count == 1


def test_translate_english_skips_translation() -> None:
    """Test that English audio skips translation step.

    When Whisper detects English, the translation step is skipped.
    Langid detector is not used - Whisper handles language detection.
    """
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    english_text = "Hello grandmother"
    fake_response = VerboseResponse(
        text=english_text,
        language="en",
        segments=[VerboseSegment(text=english_text, start=0.0, end=2.0)],
    )
    settings = _make_test_settings()
    container, fake_client, _, fake_translator = make_test_container(settings, fake_response)

    app = create_app(settings, container=container)
    client = TestClient(app)

    audio_content = generate_test_wav()
    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": ("test.wav", audio_content, "audio/wav")},
    )

    assert response.status_code == 200
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("text") == english_text
    assert body.get("detected_language") == "en"
    assert body.get("source_text") == english_text
    assert fake_client.call_count == 1
    # Translator not called for English source
    assert fake_translator.call_count == 0


def test_translate_silent_audio_returns_empty() -> None:
    """Test that silent/empty transcription returns empty response.

    When Whisper detects no speech in the audio, an empty response is returned
    with confidence 0.0 and detected_language 'unknown'.
    """
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    # Whisper returns empty text when no speech detected
    fake_response = VerboseResponse(
        text="",
        language=None,
        segments=[],
    )
    settings = _make_test_settings()
    container, fake_client, _, fake_translator = make_test_container(settings, fake_response)

    app = create_app(settings, container=container)
    client = TestClient(app)

    audio_content = generate_test_wav()
    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": ("test.wav", audio_content, "audio/wav")},
    )

    assert response.status_code == 200
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("text") == ""
    assert body.get("detected_language") == "unknown"
    assert body.get("source_text") == ""
    assert body.get("confidence") == 0.0
    assert fake_client.call_count == 1
    # Translator not called for empty transcription
    assert fake_translator.call_count == 0
