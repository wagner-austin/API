"""types: WhisperTask and related definitions."""

from __future__ import annotations

from enum import StrEnum

from platform_core.members import find_member


class WhisperTask(StrEnum):
    """What Whisper does with the audio: transcribe it, or translate it into English."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


# Supported languages for Whisper (ISO 639-1 codes)
# Full list at: https://platform.openai.com/docs/guides/speech-to-text
WHISPER_SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "af",
        "ar",
        "hy",
        "az",
        "be",
        "bs",
        "bg",
        "ca",
        "zh",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "gl",
        "de",
        "el",
        "he",
        "hi",
        "hu",
        "is",
        "id",
        "it",
        "ja",
        "kn",
        "kk",
        "ko",
        "lv",
        "lt",
        "mk",
        "ms",
        "mr",
        "mi",
        "ne",
        "no",
        "fa",
        "pl",
        "pt",
        "ro",
        "ru",
        "sr",
        "sk",
        "sl",
        "es",
        "sw",
        "sv",
        "tl",
        "ta",
        "th",
        "tr",
        "uk",
        "ur",
        "vi",
        "cy",
    }
)


def validate_whisper_language(lang: str) -> str:
    """Validate that a language code is supported by Whisper.

    Args:
        lang: ISO 639-1 language code.

    Returns:
        The validated language code.

    Raises:
        ValueError: If the language is not supported.
    """
    if lang not in WHISPER_SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported Whisper language: {lang}")
    return lang


def validate_whisper_task(task: str) -> WhisperTask:
    """Validate that a task is a valid Whisper task.

    Args:
        task: Task string to validate.

    Returns:
        The member whose value is ``task``.

    Raises:
        ValueError: If the task is not valid.
    """
    member = find_member(task, WhisperTask)
    if member is None:
        raise ValueError(f"Invalid Whisper task: {task}")
    return member


# =============================================================================
# Transcript Segment
# =============================================================================
