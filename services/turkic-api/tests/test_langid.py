"""Tests for langid module."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tests.conftest import make_probs
from turkic_api import _test_hooks
from turkic_api.core import langid as lid


def test_build_lang_filter_with_threshold() -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            # Return variants to hit mapping logic: __label__kk and kaz_Cyrl
            if "cyrl" in text.lower():
                return (("__label__kaz_Cyrl",), make_probs(0.95))
            return (("__label__kk",), make_probs(0.80))

    model = _Model()
    keep = lid.build_lang_filter(target_lang="kk", threshold=0.90, model=model)
    assert keep("foo cyrl") is True  # kaz_Cyrl maps to kk
    assert keep("bar") is False  # prob below threshold


def test_build_lang_script_filter_match_and_mismatch() -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            t = text.lower()
            if "latn" in t:
                return (("__label__kaz_Latn",), make_probs(0.99))
            if "cyrl" in t:
                return (("__label__kaz_Cyrl",), make_probs(0.99))
            return (("__label__eng",), make_probs(0.99))

    model = _Model()
    # Script normalized from lower-case
    keep = lid.build_lang_script_filter(target_lang="kk", script="latn", threshold=0.5, model=model)
    assert keep("text latn") is True
    assert keep("text cyrl") is False  # script mismatch -> return False
    # Lang mismatch -> return False
    assert keep("english") is False
    # No script filter
    keep2 = lid.build_lang_script_filter(target_lang="kk", script=None, threshold=0.5, model=model)
    assert keep2("text latn") is True


def test_build_lang_script_filter_blank_script_treated_as_none() -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__kaz_Latn",), make_probs(0.99))

    model = _Model()
    keep = lid.build_lang_script_filter(target_lang="kk", script="   ", threshold=0.5, model=model)
    # Blank script should be treated as None (no script gating)
    assert keep("anything") is True


def test_get_fasttext_model_factory() -> None:
    """Test that _get_fasttext_model_factory returns the _FastText class via hook."""
    orig_factory = _test_hooks.langid_get_fasttext_factory

    class _FakeModel:
        def __init__(self, *, model_path: str) -> None:
            self._model_path = model_path

        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__en",), make_probs(0.99))

    def _fake_make(*, model_path: str) -> _FakeModel:
        return _FakeModel(model_path=model_path)

    def _fake_factory() -> _test_hooks.LangIdModelFactoryProtocol:
        return _fake_make

    _test_hooks.langid_get_fasttext_factory = _fake_factory
    try:
        factory = lid._get_fasttext_model_factory()
        model = factory(model_path="/fake/path.bin")
        # Verify factory works by calling predict
        labels, _probs = model.predict("test", k=1)
        assert labels == ("__label__en",)
    finally:
        _test_hooks.langid_get_fasttext_factory = orig_factory


def test_extract_prob_empty_array() -> None:
    """Test _extract_prob returns 0.0 for empty arrays."""
    empty_list: list[np.float64] = []
    empty_probs: NDArray[np.float64] = np.array(empty_list, dtype=np.float64)
    assert lid._extract_prob(empty_probs) == 0.0


def test_extract_prob_with_value() -> None:
    """Test _extract_prob returns the first value for non-empty arrays."""
    assert lid._extract_prob(make_probs(0.95, 0.05)) == 0.95
