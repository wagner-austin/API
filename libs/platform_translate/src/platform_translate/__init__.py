"""Platform Translate - Text translation with pluggable backends.

This library provides text translation using pluggable backends including
Anthropic Claude, with support for additional backends like DeepL and NLLB.

Usage:
    from platform_translate import translate_text, TranslatorConfig

    config = TranslatorConfig(
        backend="anthropic",
        api_key="sk-...",
        model="claude-3-haiku-20240307",
    )

    result = translate_text("Xin chào", "vi", "en", config)
    result["text"]  # "Hello"

    # For repeated translations, create a translator instance
    from platform_translate import create_translator

    translator = create_translator(config)
    result1 = translator.translate("Xin chào", "vi", "en")
    result2 = translator.translate("Cảm ơn", "vi", "en")
"""

from platform_translate.backends import AnthropicBackend, TranslationBackendProtocol
from platform_translate.translator import Translator, create_translator, translate_text
from platform_translate.types import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    TranslationRequest,
    TranslationResult,
    TranslatorConfig,
    decode_translation_request,
    decode_translation_result,
    decode_translator_config,
    default_translator_config,
    encode_translation_request,
    encode_translation_result,
    encode_translator_config,
    require_translation_request,
    require_translation_result,
    require_translator_config,
)

__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_MODEL",
    "AnthropicBackend",
    "TranslationBackendProtocol",
    "TranslationRequest",
    "TranslationResult",
    "Translator",
    "TranslatorConfig",
    "create_translator",
    "decode_translation_request",
    "decode_translation_result",
    "decode_translator_config",
    "default_translator_config",
    "encode_translation_request",
    "encode_translation_result",
    "encode_translator_config",
    "require_translation_request",
    "require_translation_result",
    "require_translator_config",
    "translate_text",
]
