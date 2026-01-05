"""Tests for Google AI integration test hooks."""

from __future__ import annotations

import pytest

from covenant_radar_api.integrations.google_ai._test_hooks import (
    FakeGeminiClient,
    GeminiClientFactory,
    GeminiClientProtocol,
    GeminiError,
    gemini_client_factory,
    use_fake_gemini,
    use_real_gemini,
)


class TestGeminiClientProtocol:
    """Tests for GeminiClientProtocol."""

    def test_protocol_defines_generate_content(self) -> None:
        """Test that protocol defines generate_content method."""

        class FakeClient:
            def generate_content(
                self,
                model: str,
                contents: str,
            ) -> str:
                return "generated text"

            def count_tokens(
                self,
                model: str,
                contents: str,
            ) -> tuple[int, int]:
                return (10, 0)

        client: GeminiClientProtocol = FakeClient()
        result = client.generate_content("gemini-2.5-flash", "test prompt")
        assert result == "generated text"

    def test_protocol_defines_count_tokens(self) -> None:
        """Test that protocol defines count_tokens method."""

        class FakeClient:
            def generate_content(
                self,
                model: str,
                contents: str,
            ) -> str:
                return "text"

            def count_tokens(
                self,
                model: str,
                contents: str,
            ) -> tuple[int, int]:
                return (42, 0)

        client: GeminiClientProtocol = FakeClient()
        result = client.count_tokens("gemini-2.5-flash", "test content")
        assert result == (42, 0)


class TestGeminiClientFactory:
    """Tests for GeminiClientFactory protocol."""

    def test_factory_protocol(self) -> None:
        """Test factory protocol signature."""

        def fake_factory(api_key: str) -> GeminiClientProtocol:
            class FakeClient:
                def generate_content(
                    self,
                    model: str,
                    contents: str,
                ) -> str:
                    return f"response for {api_key}"

                def count_tokens(
                    self,
                    model: str,
                    contents: str,
                ) -> tuple[int, int]:
                    return (10, 0)

            return FakeClient()

        factory: GeminiClientFactory = fake_factory
        client = factory("test-api-key")
        result = client.generate_content("gemini-2.5-flash", "prompt")
        assert result == "response for test-api-key"


class TestFakeGeminiClient:
    """Tests for FakeGeminiClient."""

    def test_init_creates_empty_call_lists(self) -> None:
        """Test that init creates empty call history."""
        fake = FakeGeminiClient()
        assert fake.generate_calls == []
        assert fake.count_calls == []

    def test_init_sets_default_response(self) -> None:
        """Test that init sets default fake response."""
        fake = FakeGeminiClient()
        assert fake.next_response == "Fake Gemini response"

    def test_init_sets_default_token_count(self) -> None:
        """Test that init sets default token count."""
        fake = FakeGeminiClient()
        assert fake.next_token_count == 100

    def test_generate_content_records_call(self) -> None:
        """Test that generate_content records the call."""
        fake = FakeGeminiClient()
        fake.generate_content("gemini-2.5-flash", "test prompt")
        assert fake.generate_calls == [("gemini-2.5-flash", "test prompt")]

    def test_generate_content_returns_configured_response(self) -> None:
        """Test that generate_content returns configured response."""
        fake = FakeGeminiClient()
        fake.next_response = "custom response"
        result = fake.generate_content("gemini-2.5-flash", "prompt")
        assert result == "custom response"

    def test_generate_content_raises_on_failure(self) -> None:
        """Test that generate_content raises when should_fail is True."""
        fake = FakeGeminiClient()
        fake.should_fail = True
        fake.fail_message = "API error"
        with pytest.raises(GeminiError, match="API error"):
            fake.generate_content("gemini-2.5-flash", "prompt")

    def test_count_tokens_records_call(self) -> None:
        """Test that count_tokens records the call."""
        fake = FakeGeminiClient()
        fake.count_tokens("gemini-2.5-flash", "test content")
        assert fake.count_calls == [("gemini-2.5-flash", "test content")]

    def test_count_tokens_returns_configured_count(self) -> None:
        """Test that count_tokens returns configured count."""
        fake = FakeGeminiClient()
        fake.next_token_count = 250
        result = fake.count_tokens("gemini-2.5-flash", "content")
        assert result == (250, 0)

    def test_count_tokens_raises_on_failure(self) -> None:
        """Test that count_tokens raises when should_fail is True."""
        fake = FakeGeminiClient()
        fake.should_fail = True
        fake.fail_message = "Count error"
        with pytest.raises(GeminiError, match="Count error"):
            fake.count_tokens("gemini-2.5-flash", "content")

    def test_multiple_generate_calls_recorded(self) -> None:
        """Test that multiple calls are recorded."""
        fake = FakeGeminiClient()
        fake.generate_content("model1", "prompt1")
        fake.generate_content("model2", "prompt2")
        assert len(fake.generate_calls) == 2
        assert fake.generate_calls[0] == ("model1", "prompt1")
        assert fake.generate_calls[1] == ("model2", "prompt2")


class TestGeminiError:
    """Tests for GeminiError exception."""

    def test_gemini_error_can_be_raised_and_caught(self) -> None:
        """Test that GeminiError can be raised and caught."""
        with pytest.raises(GeminiError, match="test error"):
            raise GeminiError("test error")

    def test_gemini_error_message(self) -> None:
        """Test that error message is preserved."""
        err = GeminiError("API failed")
        assert str(err) == "API failed"


class TestHookSwitching:
    """Tests for hook switching functions."""

    def test_use_fake_gemini_returns_fake_client(self) -> None:
        """Test that use_fake_gemini returns a FakeGeminiClient."""
        fake = use_fake_gemini()
        # Verify it's a FakeGeminiClient by checking its behavior
        assert fake.generate_calls == []
        assert fake.count_calls == []
        assert fake.next_response == "Fake Gemini response"
        assert fake.next_token_count == 100

    def test_use_fake_gemini_sets_factory(self) -> None:
        """Test that use_fake_gemini sets the factory hook."""
        fake = use_fake_gemini()
        fake.next_response = "hooked response"

        # Factory should now return the fake
        from covenant_radar_api.integrations.google_ai import _test_hooks

        client = _test_hooks.gemini_client_factory("any-key")
        result = client.generate_content("model", "prompt")
        assert result == "hooked response"

    def test_use_real_gemini_restores_factory(self) -> None:
        """Test that use_real_gemini restores the real factory."""
        # First switch to fake
        use_fake_gemini()

        # Then restore
        use_real_gemini()

        # Verify factory is callable (we can't actually call the real API)
        from covenant_radar_api.integrations.google_ai import _test_hooks

        assert callable(_test_hooks.gemini_client_factory)


class TestDefaultHooks:
    """Tests for default hook values."""

    def test_gemini_client_factory_is_callable(self) -> None:
        """Test that default gemini_client_factory is callable."""
        assert callable(gemini_client_factory)


def _get_gemini_api_key() -> str:
    """Get Gemini API key from environment using the hook."""
    from platform_core.config import _test_hooks

    result = _test_hooks.get_env("GEMINI_API_KEY")
    if result is None:
        return ""
    return result


def _skip_if_no_api_key() -> None:
    """Skip test if GEMINI_API_KEY is not set."""
    if not _get_gemini_api_key():
        pytest.skip("GEMINI_API_KEY not set")


class TestRealGeminiClientEmptyResponse:
    """Tests for RealGeminiClient empty response handling."""

    def test_generate_content_raises_on_empty_response(self) -> None:
        """Test that generate_content raises GeminiError on empty response."""
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            GeminiError,
            RealGeminiClient,
        )

        # Create a RealGeminiClient but replace its inner client with a fake
        # that returns None for text
        class FakeResponse:
            @property
            def text(self) -> None:
                return None

        class FakeModels:
            def generate_content(self, model: str, contents: str) -> FakeResponse:
                return FakeResponse()

            def count_tokens(self, model: str, contents: str) -> FakeCountTokensResponse:
                return FakeCountTokensResponse()

        class FakeCountTokensResponse:
            @property
            def total_tokens(self) -> int:
                return 10

        class FakeInnerClient:
            @property
            def models(self) -> FakeModels:
                return FakeModels()

        # Create client with a dummy key - inner client is replaced so real API
        # is never called
        client = RealGeminiClient("dummy-api-key-for-test")
        # Replace inner client with our fake before any API call
        client._client = FakeInnerClient()

        with pytest.raises(GeminiError, match="Gemini returned empty response"):
            client.generate_content("gemini-2.0-flash", "test")


class TestRealGeminiClientCountTokensWithFake:
    """Tests for RealGeminiClient.count_tokens using fake inner client."""

    def test_count_tokens_returns_token_tuple(self) -> None:
        """Test that count_tokens returns (input_tokens, 0) tuple."""
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            RealGeminiClient,
        )

        class FakeCountTokensResponse:
            @property
            def total_tokens(self) -> int:
                return 42

        class FakeGenerateResponse:
            @property
            def text(self) -> str:
                return "unused"

        class FakeModels:
            def generate_content(self, model: str, contents: str) -> FakeGenerateResponse:
                raise AssertionError("Should not be called")

            def count_tokens(self, model: str, contents: str) -> FakeCountTokensResponse:
                return FakeCountTokensResponse()

        class FakeInnerClient:
            @property
            def models(self) -> FakeModels:
                return FakeModels()

        client = RealGeminiClient("dummy-api-key-for-test")
        client._client = FakeInnerClient()

        result = client.count_tokens("gemini-2.0-flash", "test content")

        assert result == (42, 0)
        assert result[0] == 42  # Input tokens
        assert result[1] == 0  # Output tokens always 0


class TestRealGeminiClient:
    """Tests for RealGeminiClient instantiation (no API calls)."""

    def test_instantiation_stores_api_key(self) -> None:
        """Test that RealGeminiClient stores the API key."""
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            RealGeminiClient,
        )

        api_key = "test-api-key-12345"
        client = RealGeminiClient(api_key)
        assert client._api_key == api_key

    def test_instantiation_creates_inner_client(self) -> None:
        """Test that RealGeminiClient creates an inner client."""
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            RealGeminiClient,
        )

        api_key = "test-api-key-12345"
        client = RealGeminiClient(api_key)
        # Verify client has models attribute (the API interface)
        models = client._client.models
        assert models.__class__.__name__ == "Models"

    def test_generate_content_returns_text(self) -> None:
        """Test that generate_content returns generated text with real API."""
        _skip_if_no_api_key()
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            RealGeminiClient,
        )

        api_key = _get_gemini_api_key()
        client = RealGeminiClient(api_key)
        result = client.generate_content("gemini-2.0-flash", "Say hello")
        assert result  # Non-empty response

    def test_count_tokens_returns_tuple(self) -> None:
        """Test that count_tokens returns token count tuple with real API."""
        _skip_if_no_api_key()
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            RealGeminiClient,
        )

        api_key = _get_gemini_api_key()
        client = RealGeminiClient(api_key)
        result = client.count_tokens("gemini-2.0-flash", "Hello world")
        assert result[0] > 0  # Input tokens
        assert result[1] == 0  # Output tokens always 0


class TestCreateGenaiClient:
    """Tests for _create_genai_client function."""

    def test_creates_genai_client(self) -> None:
        """Test that _create_genai_client creates a client."""
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            _create_genai_client,
        )

        api_key = "test-api-key-12345"
        client = _create_genai_client(api_key)
        # Verify client has models attribute
        assert client.models.__class__.__name__ == "Models"


class TestRealGeminiClientFactory:
    """Tests for _real_gemini_client_factory function."""

    def test_creates_real_gemini_client(self) -> None:
        """Test that factory creates a RealGeminiClient."""
        from covenant_radar_api.integrations.google_ai._test_hooks import (
            _real_gemini_client_factory,
        )

        api_key = "test-api-key-12345"
        client = _real_gemini_client_factory(api_key)
        assert client.__class__.__name__ == "RealGeminiClient"
