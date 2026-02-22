"""Type definitions for platform_translate.

Provides TypedDict schemas with encode/decode/require_* validation for
text translation requests and results.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_str,
)
from typing_extensions import TypedDict


class TranslationRequest(TypedDict):
    """Request for text translation.

    Attributes:
        text: Source text to translate.
        source_language: ISO 639-1 source language code (e.g., "vi", "es").
        target_language: ISO 639-1 target language code (e.g., "en").
    """

    text: str
    source_language: str
    target_language: str


def encode_translation_request(request: TranslationRequest) -> JSONObject:
    """Encode TranslationRequest to JSON-compatible dict.

    Args:
        request: The request to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "text": request["text"],
        "source_language": request["source_language"],
        "target_language": request["target_language"],
    }


def decode_translation_request(obj: JSONObject) -> TranslationRequest:
    """Decode JSON object to TranslationRequest with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TranslationRequest.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If text is empty.
    """
    text = require_str(obj, "text")
    source_language = require_str(obj, "source_language")
    target_language = require_str(obj, "target_language")

    if len(text.strip()) == 0:
        raise ValueError("text cannot be empty")

    return TranslationRequest(
        text=text,
        source_language=source_language,
        target_language=target_language,
    )


def require_translation_request(obj: JSONValue) -> TranslationRequest:
    """Validate and convert JSONValue to TranslationRequest.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated TranslationRequest.

    Raises:
        JSONTypeError: If validation fails.
        ValueError: If semantic validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_translation_request(obj)


class TranslationResult(TypedDict):
    """Result from text translation.

    Attributes:
        text: Translated text in target language.
        source_language: Source language code used.
        target_language: Target language code used.
        backend: Backend identifier used for translation.
    """

    text: str
    source_language: str
    target_language: str
    backend: str


def encode_translation_result(result: TranslationResult) -> JSONObject:
    """Encode TranslationResult to JSON-compatible dict.

    Args:
        result: The result to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "text": result["text"],
        "source_language": result["source_language"],
        "target_language": result["target_language"],
        "backend": result["backend"],
    }


def decode_translation_result(obj: JSONObject) -> TranslationResult:
    """Decode JSON object to TranslationResult with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TranslationResult.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    text = require_str(obj, "text")
    source_language = require_str(obj, "source_language")
    target_language = require_str(obj, "target_language")
    backend = require_str(obj, "backend")

    return TranslationResult(
        text=text,
        source_language=source_language,
        target_language=target_language,
        backend=backend,
    )


def require_translation_result(obj: JSONValue) -> TranslationResult:
    """Validate and convert JSONValue to TranslationResult.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated TranslationResult.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_translation_result(obj)


class TranslatorConfig(TypedDict):
    """Configuration for translator service.

    Attributes:
        backend: Backend identifier ("anthropic", "deepl", "nllb").
        api_key: API key for the backend service.
        model: Model identifier for the backend (e.g., "claude-3-haiku-20240307").
    """

    backend: str
    api_key: str
    model: str


def encode_translator_config(config: TranslatorConfig) -> JSONObject:
    """Encode TranslatorConfig to JSON-compatible dict.

    Args:
        config: The config to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "backend": config["backend"],
        "api_key": config["api_key"],
        "model": config["model"],
    }


def decode_translator_config(obj: JSONObject) -> TranslatorConfig:
    """Decode JSON object to TranslatorConfig with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TranslatorConfig.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If backend is not supported.
    """
    backend = require_str(obj, "backend")
    api_key = require_str(obj, "api_key")
    model = require_str(obj, "model")

    supported_backends = {"anthropic", "openai", "deepl", "nllb"}
    if backend not in supported_backends:
        raise ValueError(f"backend must be one of {supported_backends}, got '{backend}'")

    return TranslatorConfig(
        backend=backend,
        api_key=api_key,
        model=model,
    )


def require_translator_config(obj: JSONValue) -> TranslatorConfig:
    """Validate and convert JSONValue to TranslatorConfig.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated TranslatorConfig.

    Raises:
        JSONTypeError: If validation fails.
        ValueError: If semantic validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_translator_config(obj)


# Default configuration values
DEFAULT_BACKEND = "anthropic"
DEFAULT_MODEL = "claude-3-haiku-20240307"


def default_translator_config(api_key: str) -> TranslatorConfig:
    """Create default translator configuration.

    Args:
        api_key: API key for the backend service.

    Returns:
        TranslatorConfig with default values.
    """
    return TranslatorConfig(
        backend=DEFAULT_BACKEND,
        api_key=api_key,
        model=DEFAULT_MODEL,
    )


__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_MODEL",
    "TranslationRequest",
    "TranslationResult",
    "TranslatorConfig",
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
]
