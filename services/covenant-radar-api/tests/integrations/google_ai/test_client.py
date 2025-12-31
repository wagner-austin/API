"""Tests for Google AI (Gemini) client."""

from __future__ import annotations

import pytest

from covenant_radar_api.integrations.google_ai import _test_hooks
from covenant_radar_api.integrations.google_ai._test_hooks import (
    FakeGeminiClient,
    GeminiError,
)
from covenant_radar_api.integrations.google_ai.client import (
    ALERT_PROMPT_TEMPLATE,
    GeminiClient,
    GeminiConfig,
    _build_prompt,
    create_gemini_client,
    make_default_gemini_config,
)
from covenant_radar_api.integrations.google_ai.schemas import (
    AlertContext,
    make_alert_context,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def make_test_alert_context() -> AlertContext:
    """Create a test AlertContext."""
    return make_alert_context(
        deal_id="deal-001",
        deal_name="Acme Corp Loan",
        borrower_name="Acme Corporation",
        sector="Technology",
        risk_probability=0.85,
        risk_tier="CRITICAL",
        evaluation_status="BREACH",
        breaches_count=2,
        covenants_evaluated=5,
        period_start="2024-01-01",
        period_end="2024-03-31",
    )


def make_test_config() -> GeminiConfig:
    """Create a test GeminiConfig."""
    return {
        "api_key": "test-api-key",
        "model": "gemini-2.5-flash",
    }


def setup_fake_gemini() -> FakeGeminiClient:
    """Set up fake Gemini client for testing."""
    fake = FakeGeminiClient()

    def fake_factory(api_key: str) -> FakeGeminiClient:
        return fake

    _test_hooks.gemini_client_factory = fake_factory
    return fake


# =============================================================================
# GeminiConfig Tests
# =============================================================================


class TestMakeDefaultGeminiConfig:
    """Tests for make_default_gemini_config."""

    def test_returns_empty_api_key(self) -> None:
        """Test that default config has empty API key."""
        config = make_default_gemini_config()
        assert config["api_key"] == ""

    def test_uses_flash_model(self) -> None:
        """Test that default config uses gemini-2.5-flash."""
        config = make_default_gemini_config()
        assert config["model"] == "gemini-2.5-flash"


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestBuildPrompt:
    """Tests for _build_prompt helper."""

    def test_includes_deal_id(self) -> None:
        """Test that prompt includes deal ID."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "deal-001" in prompt

    def test_includes_deal_name(self) -> None:
        """Test that prompt includes deal name."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "Acme Corp Loan" in prompt

    def test_includes_borrower_name(self) -> None:
        """Test that prompt includes borrower name."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "Acme Corporation" in prompt

    def test_includes_sector(self) -> None:
        """Test that prompt includes sector."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "Technology" in prompt

    def test_includes_risk_probability_formatted(self) -> None:
        """Test that prompt includes formatted risk probability."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        # 0.85 formatted as 85.0%
        assert "85.0%" in prompt

    def test_includes_risk_tier(self) -> None:
        """Test that prompt includes risk tier."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "CRITICAL" in prompt

    def test_includes_evaluation_status(self) -> None:
        """Test that prompt includes evaluation status."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "BREACH" in prompt

    def test_includes_breaches_count(self) -> None:
        """Test that prompt includes breaches count."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "2" in prompt

    def test_includes_covenants_evaluated(self) -> None:
        """Test that prompt includes covenants evaluated."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "5" in prompt

    def test_includes_period_dates(self) -> None:
        """Test that prompt includes period dates."""
        context = make_test_alert_context()
        prompt = _build_prompt(context)
        assert "2024-01-01" in prompt
        assert "2024-03-31" in prompt


class TestAlertPromptTemplate:
    """Tests for ALERT_PROMPT_TEMPLATE."""

    def test_template_is_non_empty(self) -> None:
        """Test that template is a non-empty string."""
        # Verify by checking string operations work
        assert "You are a financial risk analyst" in ALERT_PROMPT_TEMPLATE

    def test_template_has_placeholders(self) -> None:
        """Test that template has required placeholders."""
        assert "{deal_id}" in ALERT_PROMPT_TEMPLATE
        assert "{deal_name}" in ALERT_PROMPT_TEMPLATE
        assert "{borrower_name}" in ALERT_PROMPT_TEMPLATE
        assert "{sector}" in ALERT_PROMPT_TEMPLATE
        assert "{risk_probability" in ALERT_PROMPT_TEMPLATE
        assert "{risk_tier}" in ALERT_PROMPT_TEMPLATE
        assert "{evaluation_status}" in ALERT_PROMPT_TEMPLATE
        assert "{breaches_count}" in ALERT_PROMPT_TEMPLATE
        assert "{covenants_evaluated}" in ALERT_PROMPT_TEMPLATE
        assert "{period_start}" in ALERT_PROMPT_TEMPLATE
        assert "{period_end}" in ALERT_PROMPT_TEMPLATE


# =============================================================================
# GeminiClient Tests
# =============================================================================


class TestGeminiClientInit:
    """Tests for GeminiClient initialization."""

    def test_stores_model(self) -> None:
        """Test that client stores model name."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-flash")
        assert client.model == "gemini-2.5-flash"

    def test_model_property(self) -> None:
        """Test model property accessor."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-pro")
        assert client.model == "gemini-2.5-pro"


class TestGeminiClientGenerateAlertSummary:
    """Tests for GeminiClient.generate_alert_summary."""

    def test_calls_count_tokens(self) -> None:
        """Test that count_tokens is called."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        client.generate_alert_summary(context)

        assert len(fake.count_calls) == 1
        assert fake.count_calls[0][0] == "gemini-2.5-flash"

    def test_calls_generate_content(self) -> None:
        """Test that generate_content is called."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        client.generate_alert_summary(context)

        assert len(fake.generate_calls) == 1
        assert fake.generate_calls[0][0] == "gemini-2.5-flash"

    def test_passes_built_prompt(self) -> None:
        """Test that built prompt is passed to generate_content."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        client.generate_alert_summary(context)

        # Check that prompt contains context data
        prompt = fake.generate_calls[0][1]
        assert "deal-001" in prompt
        assert "Acme Corp Loan" in prompt

    def test_returns_response_with_summary(self) -> None:
        """Test that response contains generated summary."""
        fake = FakeGeminiClient()
        fake.next_response = "Critical alert for Acme Corp."
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        response = client.generate_alert_summary(context)

        assert response["summary"] == "Critical alert for Acme Corp."

    def test_returns_response_with_input_tokens(self) -> None:
        """Test that response contains input token count."""
        fake = FakeGeminiClient()
        fake.next_token_count = 150
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        response = client.generate_alert_summary(context)

        assert response["input_tokens"] == 150

    def test_returns_response_with_estimated_output_tokens(self) -> None:
        """Test that response contains estimated output tokens."""
        fake = FakeGeminiClient()
        fake.next_response = "Short."  # 6 chars -> ~1-2 tokens
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        response = client.generate_alert_summary(context)

        # Output tokens = len(summary) // 4
        assert response["output_tokens"] == 6 // 4

    def test_returns_response_with_model(self) -> None:
        """Test that response contains model name."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-pro")
        context = make_test_alert_context()

        response = client.generate_alert_summary(context)

        assert response["model"] == "gemini-2.5-pro"

    def test_returns_response_with_latency(self) -> None:
        """Test that response contains latency measurement."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        response = client.generate_alert_summary(context)

        # Latency should be non-negative
        assert response["latency_ms"] >= 0

    def test_propagates_gemini_error(self) -> None:
        """Test that GeminiError is propagated."""
        fake = FakeGeminiClient()
        fake.should_fail = True
        fake.fail_message = "API quota exceeded"
        client = GeminiClient(fake, "gemini-2.5-flash")
        context = make_test_alert_context()

        with pytest.raises(GeminiError, match="API quota exceeded"):
            client.generate_alert_summary(context)


class TestGeminiClientGenerateText:
    """Tests for GeminiClient.generate_text."""

    def test_calls_generate_content_with_prompt(self) -> None:
        """Test that generate_content is called with prompt."""
        fake = FakeGeminiClient()
        client = GeminiClient(fake, "gemini-2.5-flash")

        client.generate_text("Custom prompt")

        assert len(fake.generate_calls) == 1
        assert fake.generate_calls[0][0] == "gemini-2.5-flash"
        assert fake.generate_calls[0][1] == "Custom prompt"

    def test_returns_generated_text(self) -> None:
        """Test that generated text is returned."""
        fake = FakeGeminiClient()
        fake.next_response = "Generated response"
        client = GeminiClient(fake, "gemini-2.5-flash")

        result = client.generate_text("Prompt")

        assert result == "Generated response"

    def test_propagates_gemini_error(self) -> None:
        """Test that GeminiError is propagated."""
        fake = FakeGeminiClient()
        fake.should_fail = True
        fake.fail_message = "Model error"
        client = GeminiClient(fake, "gemini-2.5-flash")

        with pytest.raises(GeminiError, match="Model error"):
            client.generate_text("Prompt")


# =============================================================================
# Factory Tests
# =============================================================================


class TestCreateGeminiClient:
    """Tests for create_gemini_client factory."""

    def test_creates_client_with_config(self) -> None:
        """Test that factory creates client with config."""
        setup_fake_gemini()
        config = make_test_config()

        client = create_gemini_client(config)

        assert client.model == "gemini-2.5-flash"

    def test_uses_hook_factory(self) -> None:
        """Test that factory uses the hook factory."""
        fake = setup_fake_gemini()
        fake.next_response = "Hooked response"
        config = make_test_config()

        client = create_gemini_client(config)
        result = client.generate_text("Test")

        assert result == "Hooked response"

    def test_client_uses_configured_model(self) -> None:
        """Test that created client uses configured model."""
        setup_fake_gemini()
        config: GeminiConfig = {
            "api_key": "key",
            "model": "gemini-2.5-pro",
        }

        client = create_gemini_client(config)

        assert client.model == "gemini-2.5-pro"

    def test_generated_calls_use_correct_model(self) -> None:
        """Test that API calls use the configured model."""
        fake = setup_fake_gemini()
        config: GeminiConfig = {
            "api_key": "key",
            "model": "gemini-2.5-pro",
        }

        client = create_gemini_client(config)
        client.generate_text("Test")

        assert fake.generate_calls[0][0] == "gemini-2.5-pro"
