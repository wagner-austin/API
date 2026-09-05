"""Language identification using FastText models.

Provides language detection on text using the NLLB LID-218e or FastText
lid.176 models. Useful for auto-detecting source language before transcription.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from platform_core.langid_models import langid_model_file

from . import _test_hooks
from .types import LanguageDetectionResult

# ISO 639-3 to 639-1 mappings for common languages
_LANG_MAPPING: dict[str, str] = {
    # Vietnamese
    "vie": "vi",
    "vi": "vi",
    # English
    "eng": "en",
    "en": "en",
    # Common Turkic languages
    "kaz": "kk",
    "kir": "ky",
    "tur": "tr",
    "uzn": "uz",
    "uzs": "uz",
    "uig": "ug",
    "aze": "az",
    "azj": "az",
    "azb": "az",
    "kk": "kk",
    "ky": "ky",
    "tr": "tr",
    "uz": "uz",
    "ug": "ug",
    "az": "az",
    # Other common languages
    "fin": "fi",
    "fi": "fi",
    "jpn": "ja",
    "ja": "ja",
    "kor": "ko",
    "ko": "ko",
    "cmn": "zh",
    "zho": "zh",
    "zh": "zh",
    "spa": "es",
    "es": "es",
    "fra": "fr",
    "fr": "fr",
    "deu": "de",
    "de": "de",
    "por": "pt",
    "pt": "pt",
    "rus": "ru",
    "ru": "ru",
    "ara": "ar",
    "ar": "ar",
    "hin": "hi",
    "hi": "hi",
    "tha": "th",
    "th": "th",
}


def ensure_model_path(data_dir: str, prefer_218e: bool = True) -> Path:
    """Ensure model file exists, downloading if necessary.

    Uses _test_hooks.langid_download for the actual download operation.

    Args:
        data_dir: Base directory for model storage.
        prefer_218e: If True, use LID-218e model; otherwise use lid.176.

    Returns:
        Path to the model file.
    """
    wanted = langid_model_file(data_dir, prefer_218e=prefer_218e)
    if not wanted["path"].exists():
        _test_hooks.langid_download(wanted["url"], wanted["path"])
    return wanted["path"]


def _parse_label(raw: str) -> tuple[str, str | None]:
    """Parse a FastText label into language code and script.

    Maps ISO 639-3 codes to ISO 639-1 for common languages.

    Args:
        raw: Raw FastText label (e.g., "__label__vie_Latn").

    Returns:
        Tuple of (language_code, script_or_none).
    """
    label = raw.replace("__label__", "")
    if "_" in label:
        lang_part, script = label.split("_", 1)
    else:
        lang_part, script = label, None
    return _LANG_MAPPING.get(lang_part, lang_part), script


def _extract_prob(probs: NDArray[np.float64]) -> float:
    """Extract the first probability as a float, or 0.0 if empty.

    Args:
        probs: Probability array from FastText prediction.

    Returns:
        First probability value or 0.0.
    """
    if len(probs) == 0:
        return 0.0
    return float(probs.item(0))


def load_langid_model(data_dir: str, prefer_218e: bool = True) -> _test_hooks.LangIdModelProtocol:
    """Load a language-ID model from local cache, downloading if missing.

    The underlying implementation uses FastText at runtime without exposing
    untyped imports to the type checker.

    Args:
        data_dir: Base directory for model storage.
        prefer_218e: If True, use LID-218e model; otherwise use lid.176.

    Returns:
        Loaded language identification model.
    """
    model_path = ensure_model_path(data_dir, prefer_218e=prefer_218e)
    factory = _test_hooks.langid_get_fasttext_factory()
    return factory(model_path=str(model_path))


def detect_language(
    text: str,
    model: _test_hooks.LangIdModelProtocol,
    threshold: float = 0.0,
) -> LanguageDetectionResult:
    """Detect language of given text.

    Args:
        text: Text to analyze.
        model: Loaded language identification model.
        threshold: Minimum confidence threshold (0.0-1.0).

    Returns:
        LanguageDetectionResult with detected language, confidence, and script.
    """
    # Normalize text for prediction (remove newlines)
    normalized = text.replace("\n", " ").strip()
    if not normalized:
        return LanguageDetectionResult(
            language="und",  # Undetermined
            confidence=0.0,
            script=None,
        )

    labels, probs = model.predict(normalized, k=1)
    label: str = labels[0] if labels else ""
    prob = _extract_prob(probs)
    lang, script = _parse_label(label)

    # Apply threshold
    if prob < threshold:
        return LanguageDetectionResult(
            language="und",
            confidence=prob,
            script=None,
        )

    return LanguageDetectionResult(
        language=lang,
        confidence=prob,
        script=script,
    )


def is_language(
    text: str,
    target_lang: str,
    model: _test_hooks.LangIdModelProtocol,
    threshold: float = 0.5,
) -> bool:
    """Check if text is in the target language with confidence threshold.

    Args:
        text: Text to analyze.
        target_lang: Target language code (ISO 639-1).
        model: Loaded language identification model.
        threshold: Minimum confidence threshold.

    Returns:
        True if text is detected as target language with sufficient confidence.
    """
    result = detect_language(text, model, threshold=0.0)
    return result["language"] == target_lang and result["confidence"] >= threshold


__all__ = [
    "detect_language",
    "ensure_model_path",
    "is_language",
    "load_langid_model",
]
