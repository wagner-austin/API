from __future__ import annotations

from pathlib import Path

from model_trainer.core import _test_hooks
from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend


def test_sentencepiece_adapter_encode_decode(tmp_path: Path) -> None:
    """Test that SPMAdapter encode/decode methods work through hooks."""
    # Provide model + vocab files
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"")
    (tmp_path / "tokenizer.vocab").write_text("[PAD]\n[UNK]\n[EOS]\nfoo\nbar\n", encoding="utf-8")

    # Set up hooks to return fixed outputs for encode/decode
    def _fake_encode(model_path: str, text: str) -> list[int]:
        _ = model_path
        return [1, 2, 3]

    def _fake_decode(model_path: str, ids: list[int]) -> str:
        _ = model_path, ids
        return "hello"

    _test_hooks.spm_encode_ids = _fake_encode
    _test_hooks.spm_decode_ids = _fake_decode

    handle = SentencePieceBackend().load(str(model_path))
    ids = SentencePieceBackend().encode(handle, "foo bar")
    text = SentencePieceBackend().decode(handle, ids)
    assert ids == [1, 2, 3]
    assert text == "hello"
