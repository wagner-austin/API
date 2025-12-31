"""Google AI (Gemini) integration for alert text generation.

This module provides a typed client for generating human-readable alert
summaries using Google's Gemini LLM.

Usage:
    from covenant_radar_api.integrations.google_ai import (
        create_gemini_client,
        GeminiConfig,
    )

    config: GeminiConfig = {
        "api_key": "your-api-key",
        "model": "gemini-2.5-flash",
    }
    client = create_gemini_client(config)
    summary = client.generate_alert_summary(context)
"""

from __future__ import annotations

from .client import (
    GeminiClient,
    GeminiConfig,
    create_gemini_client,
    make_default_gemini_config,
)
from .schemas import (
    AlertContext,
    GenerateAlertRequest,
    GenerateAlertResponse,
    make_alert_context,
    make_generate_alert_request,
)

__all__ = [
    "AlertContext",
    "GeminiClient",
    "GeminiConfig",
    "GenerateAlertRequest",
    "GenerateAlertResponse",
    "create_gemini_client",
    "make_alert_context",
    "make_default_gemini_config",
    "make_generate_alert_request",
]
