from __future__ import annotations

import shutil

import pytest

from model_trainer.core.services.container import _create_tokenizer_registry


def test_tokenizer_registry_without_sentencepiece(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which_none(name: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _which_none)
    reg = _create_tokenizer_registry()
    assert "bpe" in reg.backends and "sentencepiece" not in reg.backends


def test_tokenizer_registry_with_sentencepiece(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which_path(name: str) -> str:
        return "/bin/spm"

    monkeypatch.setattr(shutil, "which", _which_path)
    reg = _create_tokenizer_registry()
    assert "sentencepiece" in reg.backends
