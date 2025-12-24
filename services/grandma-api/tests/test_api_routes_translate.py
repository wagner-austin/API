"""Tests for grandma_api.api.routes.translate module."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_stt import VerboseResponse, VerboseSegment

from grandma_api.api.main import create_app
from grandma_api.config import GrandmaApiSettings

from .conftest import make_test_container, set_fake_env


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
    """Test successful translation."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    response_text = "Hello, how are you grandmother?"
    fake_response = VerboseResponse(
        text=response_text,
        segments=[VerboseSegment(text=response_text, start=0.0, end=2.0)],
    )
    settings = _make_test_settings()
    container, fake_client = make_test_container(settings, fake_response)

    app = create_app(settings, container=container)
    client = TestClient(app)

    audio_content = b"fake audio bytes"
    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": ("test.webm", audio_content, "audio/webm")},
    )

    assert response.status_code == 200
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("text") == response_text
    assert fake_client.call_count == 1


def test_translate_invalid_token() -> None:
    """Test translation with invalid token returns 401."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, fake_client = make_test_container(settings)

    app = create_app(settings, container=container)
    client = TestClient(app)

    audio_content = b"fake audio bytes"
    response = client.post(
        "/translate",
        data={"token": "wrong-token"},
        files={"audio": ("test.webm", audio_content, "audio/webm")},
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
    container, fake_client = make_test_container(settings)

    app = create_app(settings, container=container)
    client = TestClient(app)

    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": ("test.webm", b"", "audio/webm")},
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
    container, fake_client = make_test_container(settings)

    app = create_app(settings, container=container)
    client = TestClient(app)

    # Send without explicit filename
    audio_content = b"fake audio bytes"
    response = client.post(
        "/translate",
        data={"token": "test-token"},
        files={"audio": audio_content},
    )

    assert response.status_code == 200
    assert fake_client.call_count == 1
