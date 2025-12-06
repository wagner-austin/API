from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol

import requests

_MODEL_DIRNAME: Final[str] = "models"
_URL_218E: Final[str] = "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"
_URL_176: Final[str] = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def ensure_model_path(data_dir: str, prefer_218e: bool = True) -> Path:
    base = Path(data_dir) / _MODEL_DIRNAME
    path_218e = base / "lid218e.bin"
    path_176 = base / "lid.176.bin"
    if prefer_218e:
        if not path_218e.exists():
            _download(_URL_218E, path_218e)
        return path_218e
    if not path_176.exists():
        _download(_URL_176, path_176)
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
    }
    return mapping.get(lang_part, lang_part), script


class LangIdModel(Protocol):
    def predict(self, text: str, k: int = 1) -> tuple[list[str], list[float]]: ...


class _FastTextModule(Protocol):
    def load_model(self, path: str) -> LangIdModel: ...


def _get_fasttext() -> _FastTextModule:
    import importlib

    return importlib.import_module("fasttext")


def load_langid_model(data_dir: str, prefer_218e: bool = True) -> LangIdModel:
    """Load a language-id model from local cache, downloading if missing.

    The underlying implementation uses fastText at runtime without exposing
    untyped imports to the type checker.
    """
    model_path = ensure_model_path(data_dir, prefer_218e=prefer_218e)
    ft = _get_fasttext()
    return ft.load_model(str(model_path))


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
        label = labels[0] if labels else ""
        prob = probs[0] if probs else 0.0
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
        label = labels[0] if labels else ""
        prob = probs[0] if probs else 0.0
        lang, script_pred = _parse_label(label)
        if lang != target_lang:
            return False
        if script_norm is not None and script_pred != script_norm:
            return False
        return prob >= threshold

    return _keep
