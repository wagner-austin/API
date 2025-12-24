"""Test fixtures for grandma-api."""

from __future__ import annotations

from collections.abc import Generator
from typing import BinaryIO

import pytest
from platform_core.config import config_test_hooks
from platform_stt import VerboseResponse, VerboseSegment
from platform_stt._test_hooks import STTClientProtocol

from grandma_api.api import _test_hooks as api_hooks
from grandma_api.config import GrandmaApiSettings
from grandma_api.core.container import ServiceContainer, STTClientFactoryProtocol


class FakeSTTClient:
    """Fake STT client for testing.

    Returns configurable responses without making real API calls.
    """

    __slots__ = ("_response", "call_count", "last_file")

    def __init__(self, response: VerboseResponse | None = None) -> None:
        """Initialize fake client.

        Args:
            response: Response to return from translate().
        """
        default = VerboseResponse(
            text="Hello from grandmother",
            segments=[VerboseSegment(text="Hello", start=0.0, end=1.0)],
        )
        self._response = response if response is not None else default
        self.call_count = 0
        self.last_file: bytes | None = None

    def transcribe(
        self,
        *,
        file: BinaryIO,
        language: str | None = None,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Fake transcription.

        Args:
            file: Binary file-like object.
            language: Optional language hint.
            timeout: Optional timeout.

        Returns:
            Configured response.
        """
        _ = (language, timeout)
        self.last_file = file.read()
        file.seek(0)
        self.call_count += 1
        return self._response

    def translate(
        self,
        *,
        file: BinaryIO,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Fake translation.

        Args:
            file: Binary file-like object.
            timeout: Optional timeout.

        Returns:
            Configured response.
        """
        _ = timeout
        self.last_file = file.read()
        file.seek(0)
        self.call_count += 1
        return self._response


def make_fake_stt_client(
    response: VerboseResponse | None = None,
) -> tuple[FakeSTTClient, STTClientFactoryProtocol]:
    """Create fake STT client and factory.

    Args:
        response: Response to return from client.

    Returns:
        Tuple of (fake_client, factory_function).
    """
    client = FakeSTTClient(response)

    def factory(api_key: str) -> STTClientProtocol:
        del api_key  # unused
        return client

    return client, factory


def make_test_container(
    settings: GrandmaApiSettings,
    response: VerboseResponse | None = None,
) -> tuple[ServiceContainer, FakeSTTClient]:
    """Create ServiceContainer with fake STT client for testing.

    Args:
        settings: Test settings.
        response: Response to return from fake client.

    Returns:
        Tuple of (ServiceContainer, FakeSTTClient).
    """
    client, factory = make_fake_stt_client(response)
    container = ServiceContainer(settings=settings, stt_client_factory=factory)
    return container, client


def set_fake_env(env: dict[str, str]) -> None:
    """Set fake environment variables for testing.

    Args:
        env: Dictionary of environment variable values.
    """

    def _fake_env(key: str) -> str | None:
        return env.get(key)

    config_test_hooks.get_env = _fake_env


@pytest.fixture(autouse=True)
def _restore_config_hooks() -> Generator[None, None, None]:
    """Restore config hooks after each test."""
    original_get_env = config_test_hooks.get_env
    yield
    config_test_hooks.get_env = original_get_env


@pytest.fixture(autouse=True)
def _restore_api_hooks() -> Generator[None, None, None]:
    """Restore API hooks after each test."""
    original_stt_factory = api_hooks.stt_client_factory
    yield
    api_hooks.stt_client_factory = original_stt_factory
