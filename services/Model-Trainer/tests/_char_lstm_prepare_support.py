"""Shared fakes for the char_lstm prepare/train/eval tests."""

from __future__ import annotations

from pathlib import Path

from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.char_backend import CharBackend


def _write_tiny_corpus(root: Path) -> str:
    out_dir = root / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "tiny.txt"
    # Expanded corpus for meaningful training - original 10 bytes was too small
    # Need multiple batches to show loss reduction across training steps
    corpus_lines = ["aba", "abbaba", "abaaba", "babbab", "ababab", "bababa"]
    corpus_text = "\n".join(corpus_lines * 10) + "\n"  # ~300 bytes
    fp.write_text(corpus_text, encoding="utf-8")
    return str(out_dir)


def _train_char_tokenizer(root: Path, corpus_path: str) -> tuple[str, str]:
    tok_out = root / "artifacts" / "tokenizers" / "tok1"
    cfg = TokenizerTrainConfig(
        method="char",
        vocab_size=0,
        min_frequency=1,
        corpus_path=corpus_path,
        holdout_fraction=0.05,
        seed=42,
        out_dir=str(tok_out),
    )
    stats = CharBackend().train(cfg)
    assert stats.token_count >= 4
    return "tok1", str(tok_out)


def _noop(_: float) -> None:
    return None


def _never() -> bool:
    return False


class _FakeTokHandle:
    """Fake tokenizer handle for testing tokenizer_id None case."""

    def encode(self: _FakeTokHandle, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self: _FakeTokHandle, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def token_to_id(self: _FakeTokHandle, token: str) -> int | None:
        if token == "[EOS]":
            return 0
        if token == "[PAD]":
            return 1
        return None

    def get_vocab_size(self: _FakeTokHandle) -> int:
        return 256
