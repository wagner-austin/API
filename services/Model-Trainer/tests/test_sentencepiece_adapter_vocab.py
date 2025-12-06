from __future__ import annotations

from pathlib import Path

from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend


def test_sentencepiece_adapter_vocab_read(tmp_path: Path) -> None:
    # Create dummy model path and vocab file
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"")
    vocab_path = tmp_path / "tokenizer.vocab"
    vocab_path.write_text("[PAD]\n\n[UNK]\n[EOS]\nfoo\nbar\n", encoding="utf-8")
    # Load adapter and check token_to_id and vocab size
    handle = SentencePieceBackend().load(str(model_path))
    # token_to_id depends on line order
    foo_id = handle.token_to_id("foo")
    assert foo_id is not None and foo_id >= 0
    vocab_size = handle.get_vocab_size()
    assert vocab_size >= 2
