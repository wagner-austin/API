"""Gemini client for alert text generation.

This module provides a typed wrapper around the Gemini API for generating
human-readable alert summaries for covenant breaches and high-risk predictions.

Usage:
    from covenant_radar_api.integrations.google_ai import (
        create_gemini_client,
        GeminiConfig,
        make_alert_context,
    )

    config: GeminiConfig = {
        "api_key": "your-api-key",
        "model": "gemini-2.5-flash",
    }
    client = create_gemini_client(config)

    context = make_alert_context(
        deal_id="deal-001",
        deal_name="Acme Corp Loan",
        ...
    )
    response = client.generate_alert_summary(context)
    summary = response["summary"]  # Access the generated text

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time
from typing import TypedDict

from . import _test_hooks
from .schemas import (
    AlertContext,
    GenerateAlertResponse,
    make_generate_alert_response,
)

# =============================================================================
# Configuration
# =============================================================================


class GeminiConfig(TypedDict, total=True):
    """Configuration for Gemini client.

    Fields:
        api_key: Google AI API key for authentication.
        model: Gemini model name (e.g., "gemini-2.5-flash").
    """

    api_key: str
    model: str


def make_default_gemini_config() -> GeminiConfig:
    """Create default Gemini configuration.

    Uses gemini-2.5-flash as the default model for fast, low-cost generation.

    Returns:
        GeminiConfig with sensible defaults (api_key must be overridden).
    """
    return {
        "api_key": "",
        "model": "gemini-2.5-flash",
    }


# =============================================================================
# Prompt Template
# =============================================================================


ALERT_PROMPT_TEMPLATE = (
    "You are a financial risk analyst assistant. "
    "Generate a concise, professional alert summary for the following situation.\n"
    "\n"
    "Deal Information:\n"
    "- Deal ID: {deal_id}\n"
    "- Deal Name: {deal_name}\n"
    "- Borrower: {borrower_name}\n"
    "- Sector: {sector}\n"
    "- Period: {period_start} to {period_end}\n"
    "\n"
    "Risk Assessment:\n"
    "- ML Risk Probability: {risk_probability:.1%}\n"
    "- Risk Tier: {risk_tier}\n"
    "- Evaluation Status: {evaluation_status}\n"
    "- Covenants Evaluated: {covenants_evaluated}\n"
    "- Breaches Detected: {breaches_count}\n"
    "\n"
    "Write a single paragraph (2-3 sentences) summarizing this alert for a credit "
    "risk officer. Focus on the key risk factors and recommended immediate actions. "
    "Be specific about the severity and urgency."
)


def _build_prompt(context: AlertContext) -> str:
    """Build the Gemini prompt from alert context.

    Args:
        context: Alert context with deal and risk information.

    Returns:
        Formatted prompt string for Gemini.
    """
    return ALERT_PROMPT_TEMPLATE.format(
        deal_id=context["deal_id"],
        deal_name=context["deal_name"],
        borrower_name=context["borrower_name"],
        sector=context["sector"],
        period_start=context["period_start"],
        period_end=context["period_end"],
        risk_probability=context["risk_probability"],
        risk_tier=context["risk_tier"],
        evaluation_status=context["evaluation_status"],
        covenants_evaluated=context["covenants_evaluated"],
        breaches_count=context["breaches_count"],
    )


# =============================================================================
# Client
# =============================================================================


class GeminiClient:
    """Typed client for generating alert summaries with Gemini.

    This client wraps the Gemini API with a domain-specific interface
    for generating human-readable alert summaries.

    Example:
        config: GeminiConfig = {
            "api_key": os.environ["GEMINI_API_KEY"],
            "model": "gemini-2.5-flash",
        }
        client = create_gemini_client(config)
        response = client.generate_alert_summary(context)
    """

    def __init__(
        self,
        inner: _test_hooks.GeminiClientProtocol,
        model: str,
    ) -> None:
        """Initialize the Gemini client.

        Args:
            inner: Underlying Gemini client (real or fake).
            model: Gemini model name to use for generation.
        """
        self._inner = inner
        self._model = model

    @property
    def model(self) -> str:
        """Get the configured model name."""
        return self._model

    def generate_alert_summary(
        self,
        context: AlertContext,
    ) -> GenerateAlertResponse:
        """Generate a human-readable alert summary.

        Calls Gemini to generate a professional alert summary based on
        the provided deal and risk context.

        Args:
            context: Alert context with deal and risk information.

        Returns:
            GenerateAlertResponse with summary text and usage metrics.

        Raises:
            GeminiError: If the API call fails.
        """
        prompt = _build_prompt(context)

        # Count input tokens for metrics
        input_tokens, _ = self._inner.count_tokens(self._model, prompt)

        # Generate summary with timing
        start_time = time.perf_counter()
        summary = self._inner.generate_content(self._model, prompt)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Estimate output tokens (rough approximation: ~4 chars per token)
        output_tokens = len(summary) // 4

        return make_generate_alert_response(
            summary=summary,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
            latency_ms=latency_ms,
        )

    def generate_text(
        self,
        prompt: str,
    ) -> str:
        """Generate text from a raw prompt.

        Lower-level method for custom prompts.

        Args:
            prompt: Text prompt to send to Gemini.

        Returns:
            Generated text response.

        Raises:
            GeminiError: If the API call fails.
        """
        return self._inner.generate_content(self._model, prompt)


# =============================================================================
# Factory
# =============================================================================


def create_gemini_client(config: GeminiConfig) -> GeminiClient:
    """Create a Gemini client with the given configuration.

    Uses the injectable gemini_client_factory from _test_hooks.
    Production code uses RealGeminiClient; tests inject FakeGeminiClient.

    Args:
        config: Gemini configuration with API key and model.

    Returns:
        GeminiClient instance.
    """
    inner = _test_hooks.gemini_client_factory(config["api_key"])
    return GeminiClient(inner, config["model"])


__all__ = [
    "ALERT_PROMPT_TEMPLATE",
    "GeminiClient",
    "GeminiConfig",
    "create_gemini_client",
    "make_default_gemini_config",
]
