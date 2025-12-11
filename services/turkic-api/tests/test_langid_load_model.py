"""Tests for langid module model loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tests.conftest import make_probs
from turkic_api import _test_hooks
from turkic_api.core import langid as lid


def test_load_langid_model_calls_fasttext_factory(tmp_path: Path) -> None:
    """Test that load_langid_model uses the factory to construct the model."""
    orig_ensure_model_path = _test_hooks.langid_ensure_model_path
    orig_get_fasttext_factory = _test_hooks.langid_get_fasttext_factory

    # Force ensure_model_path to return a fake path without disk IO
    def _ensure(data_dir: str, prefer_218e: bool = True) -> Path:
        return tmp_path / "model.bin"

    _test_hooks.langid_ensure_model_path = _ensure

    loaded_paths: list[str] = []

    class _FakeModel:
        def __init__(self, *, model_path: str) -> None:
            loaded_paths.append(model_path)

        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__en",), make_probs(0.99))

    def _fake_factory(*, model_path: str) -> _FakeModel:
        return _FakeModel(model_path=model_path)

    def _get_factory() -> _test_hooks.LangIdModelFactoryProtocol:
        return _fake_factory

    _test_hooks.langid_get_fasttext_factory = _get_factory
    try:
        model = lid.load_langid_model(str(tmp_path))
        # Verify model was loaded correctly by calling predict
        labels, _probs = model.predict("test", k=1)
        assert labels == ("__label__en",)
        assert loaded_paths
        first = loaded_paths[0]
        assert first.endswith("model.bin")
    finally:
        _test_hooks.langid_ensure_model_path = orig_ensure_model_path
        _test_hooks.langid_get_fasttext_factory = orig_get_fasttext_factory


def test_get_fasttext_model_factory_returns_hook_value() -> None:
    """Test that _get_fasttext_model_factory returns value from hook."""
    orig_get_fasttext_factory = _test_hooks.langid_get_fasttext_factory

    class _FakeModel:
        def __init__(self, *, model_path: str) -> None:
            self._model_path = model_path

        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__en",), make_probs(0.99))

    def _fake_factory(*, model_path: str) -> _FakeModel:
        return _FakeModel(model_path=model_path)

    def _get_factory() -> _test_hooks.LangIdModelFactoryProtocol:
        return _fake_factory

    _test_hooks.langid_get_fasttext_factory = _get_factory
    try:
        factory = lid._get_fasttext_model_factory()
        # Verify factory works by calling predict
        model = factory(model_path="/test/path.bin")
        labels, _probs = model.predict("test", k=1)
        assert labels == ("__label__en",)
    finally:
        _test_hooks.langid_get_fasttext_factory = orig_get_fasttext_factory
