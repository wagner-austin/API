from __future__ import annotations

from model_trainer.core import _test_hooks
from model_trainer.core.services.container import _create_tokenizer_registry


def test_tokenizer_registry_without_sentencepiece() -> None:
    def _which_none(cmd: str) -> None:
        return None

    _test_hooks.shutil_which = _which_none
    reg = _create_tokenizer_registry()
    assert "bpe" in reg.backends and "sentencepiece" not in reg.backends


def test_tokenizer_registry_with_sentencepiece() -> None:
    def _which_path(cmd: str) -> str:
        return "/bin/spm"

    _test_hooks.shutil_which = _which_path
    reg = _create_tokenizer_registry()
    assert "sentencepiece" in reg.backends
