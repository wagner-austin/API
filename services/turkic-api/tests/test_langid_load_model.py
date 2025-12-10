from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.conftest import make_probs

import turkic_api.core.langid as lid


def test_load_langid_model_calls_fasttext_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that load_langid_model uses the factory to construct the model."""

    # Force ensure_model_path to return a fake path without disk IO
    def _ensure(d: str, prefer_218e: bool = True) -> Path:
        return tmp_path / "model.bin"

    monkeypatch.setattr(lid, "ensure_model_path", _ensure)

    loaded_paths: list[str] = []

    class _FakeModel:
        def __init__(self, *, model_path: str) -> None:
            loaded_paths.append(model_path)

        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__en",), make_probs(0.99))

    def _fake_factory(*, model_path: str) -> _FakeModel:
        return _FakeModel(model_path=model_path)

    def _get_factory() -> lid._FastTextModelFactory:
        return _fake_factory

    monkeypatch.setattr(lid, "_get_fasttext_model_factory", _get_factory)
    model = lid.load_langid_model(str(tmp_path))
    # Verify model was loaded correctly by calling predict
    labels, _probs = model.predict("test", k=1)
    assert labels == ("__label__en",)
    assert loaded_paths
    first = loaded_paths[0]
    assert first.endswith("model.bin")


def test_get_fasttext_model_factory_imports_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_fasttext_model_factory imports fasttext.FastText._FastText."""
    imported: list[tuple[str, list[str] | None]] = []

    class _FakeModel:
        def __init__(self, *, model_path: str) -> None:
            self._model_path = model_path

        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__en",), make_probs(0.99))

    class _FakeFastTextModule:
        _FastText = _FakeModel

    def fake_import(
        name: str,
        globals_: dict[str, str] | None = None,
        locals_: dict[str, str] | None = None,
        fromlist: list[str] | None = None,
        level: int = 0,
    ) -> _FakeFastTextModule:
        imported.append((name, list(fromlist) if fromlist else None))
        return _FakeFastTextModule()

    monkeypatch.setattr("builtins.__import__", fake_import)
    factory = lid._get_fasttext_model_factory()
    assert imported == [("fasttext.FastText", ["_FastText"])]
    # Verify factory works by calling predict
    model = factory(model_path="/test/path.bin")
    labels, _probs = model.predict("test", k=1)
    assert labels == ("__label__en",)
