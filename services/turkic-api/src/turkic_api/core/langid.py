from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from turkic_api import _test_hooks

_MODEL_DIRNAME: Final[str] = "models"
_URL_218E: Final[str] = "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"
_URL_176: Final[str] = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def ensure_model_path(data_dir: str, prefer_218e: bool = True) -> Path:
    """Ensure model file exists, downloading if necessary.

    Uses _test_hooks.langid_download for the actual download operation.
    """
    base = Path(data_dir) / _MODEL_DIRNAME
    path_218e = base / "lid218e.bin"
    path_176 = base / "lid.176.bin"
    if prefer_218e:
        if not path_218e.exists():
            _test_hooks.langid_download(_URL_218E, path_218e)
        return path_218e
    if not path_176.exists():
        _test_hooks.langid_download(_URL_176, path_176)
    return path_176


def _parse_label(raw: str) -> tuple[str, str | None]:
    """Return (lang, script) parsed from a fastText label.

    Maps 639-3 to 639-1 for Turkic languages we support. Script is returned
    as-is (e.g., "Cyrl", "Latn") when present, otherwise None.
    """
    label = raw.replace("__label__", "")
    if "_" in label:
        lang_part, script = label.split("_", 1)
    else:
        lang_part, script = label, None
    mapping: dict[str, str] = {
        "kaz": "kk",
        "kir": "ky",
        "tur": "tr",
        "uzn": "uz",
        "uzs": "uz",
        "uig": "ug",
        "fin": "fi",
        "aze": "az",
        # NLLB LID-218e splits Azerbaijani into azj (North) and azb (South).
        # We normalize both to a single pipeline code 'az'. Script gating may
        # still be applied by callers when needed (e.g., enforce Latn).
        "azj": "az",
        "azb": "az",
        "kk": "kk",
        "ky": "ky",
        "tr": "tr",
        "uz": "uz",
        "ug": "ug",
        "fi": "fi",
        "az": "az",
        # ISO 639-3 to 639-1 for common languages
        "eng": "en",
        "en": "en",
    }
    return mapping.get(lang_part, lang_part), script


class LangIdModel(Protocol):
    """Protocol for language identification model with predict method."""

    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        """Predict language labels and probabilities for the given text."""
        ...


def _get_fasttext_model_factory() -> _test_hooks.LangIdModelFactoryProtocol:
    """Get the FastText model constructor without triggering deprecation warnings.

    The fasttext.load_model wrapper prints a warning to stderr on every call.
    We bypass it by directly accessing the underlying _FastText class from
    fasttext.FastText module, which accepts model_path as a keyword argument.

    Uses _test_hooks.langid_get_fasttext_factory for the actual factory lookup.
    """
    return _test_hooks.langid_get_fasttext_factory()


def load_langid_model(data_dir: str, prefer_218e: bool = True) -> LangIdModel:
    """Load a language-id model from local cache, downloading if missing.

    The underlying implementation uses fastText at runtime without exposing
    untyped imports to the type checker. Uses the internal _FastText constructor
    directly to avoid the deprecation warning from fasttext.load_model.

    Uses _test_hooks.langid_ensure_model_path for path resolution.
    """
    model_path = _test_hooks.langid_ensure_model_path(data_dir, prefer_218e=prefer_218e)
    factory = _get_fasttext_model_factory()
    return factory(model_path=str(model_path))


def _extract_prob(probs: NDArray[np.float64]) -> float:
    """Extract the first probability as a float, or 0.0 if empty."""
    if len(probs) == 0:
        return 0.0
    return float(probs.item(0))


def build_lang_filter(
    target_lang: str, threshold: float, model: LangIdModel
) -> Callable[[str], bool]:
    """Return a predicate that keeps sentences matching target_lang above threshold.

    Loads a FastText language-ID model from $data_dir/models, downloading it
    if necessary. The filter returns True if the predicted language equals
    target_lang and the probability >= threshold.
    """

    def _keep(text: str) -> bool:
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        label: str = labels[0] if labels else ""
        prob = _extract_prob(probs)
        lang, _script = _parse_label(label)
        return lang == target_lang and prob >= threshold

    return _keep


def build_lang_script_filter(
    *, target_lang: str, script: str | None, threshold: float, model: LangIdModel
) -> Callable[[str], bool]:
    """Return a predicate for language + optional script with probability threshold.

    If script is provided, the sentence must match both target_lang and script;
    otherwise only target_lang is enforced. Probability must be >= threshold.
    """
    script_norm = script
    if script_norm is not None:
        # Normalize to canonical capitalization like "Latn", "Cyrl"
        script_norm = script_norm.strip()
        if not script_norm:
            script_norm = None
        else:
            script_norm = script_norm[0:1].upper() + script_norm[1:].lower()

    def _keep(text: str) -> bool:
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        label: str = labels[0] if labels else ""
        prob = _extract_prob(probs)
        lang, script_pred = _parse_label(label)
        if lang != target_lang:
            return False
        if script_norm is not None and script_pred != script_norm:
            return False
        return prob >= threshold

    return _keep
