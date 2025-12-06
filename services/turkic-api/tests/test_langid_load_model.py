from __future__ import annotations

from pathlib import Path

import pytest

import turkic_api.core.langid as lid


def test_load_langid_model_calls_fasttext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force ensure_model_path to return a fake path without disk IO
    def _ensure(d: str, prefer_218e: bool = True) -> Path:
        return tmp_path / "model.bin"

    monkeypatch.setattr(lid, "ensure_model_path", _ensure)

    class _FT:
        def __init__(self) -> None:
            self.loaded: list[str] = []

        def load_model(self, p: str) -> str:
            self.loaded.append(p)
            return "model"

    ft = _FT()

    def _get() -> _FT:
        return ft

    monkeypatch.setattr(lid, "_get_fasttext", _get)
    model = lid.load_langid_model(str(tmp_path))
    if model is None:
        pytest.fail("expected model")
    assert ft.loaded
    first = ft.loaded[0]
    assert first.endswith("model.bin")


def test_get_fasttext_imports_module(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []

    class _Mod:
        pass

    def fake_import(name: str) -> _Mod:
        imported.append(name)
        return _Mod()

    monkeypatch.setattr("importlib.import_module", fake_import)
    mod = lid._get_fasttext()
    assert imported == ["fasttext"]
    assert type(mod).__name__ == "_Mod"
