"""Tests for platform_ml.sentencepiece module."""

from __future__ import annotations

from pathlib import Path

from platform_ml import sentencepiece as spm


def test_get_trainer_can_be_called() -> None:
    """Test get_trainer returns a trainer that can be invoked."""
    trainer = spm.get_trainer()
    # Access static train method directly - will raise AttributeError if missing
    _ = trainer.train


def test_get_processor_ctor_creates_processor() -> None:
    """Test get_processor_ctor returns callable that creates processor."""
    ctor = spm.get_processor_ctor()
    processor = ctor()
    # Access methods directly - will raise AttributeError if missing
    _ = processor.load
    _ = processor.encode_as_ids
    _ = processor.decode_ids


def test_require_module_succeeds() -> None:
    """Test require_module succeeds when sentencepiece is installed."""
    spm.require_module()


def _build_tokenizer(tmp_path: Path, vocab_size: int = 50) -> str:
    """Helper to build a tokenizer and return model path."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nthis is a test\n" * 100, encoding="utf-8")
    model_prefix = str(tmp_path / "tokenizer")
    spm.train([str(corpus)], model_prefix=model_prefix, vocab_size=vocab_size)
    return str(tmp_path / "tokenizer.model")


def test_spm_creates_model_files(tmp_path: Path) -> None:
    """Test spm module creates .model and .vocab files."""
    _build_tokenizer(tmp_path)
    assert (tmp_path / "tokenizer.model").exists()
    assert (tmp_path / "tokenizer.vocab").exists()


def test_spm_encode_decode_roundtrip(tmp_path: Path) -> None:
    """Test encode_ids and decode_ids roundtrip."""
    model_path = _build_tokenizer(tmp_path)

    text = "hello world"
    ids = spm.encode_ids(model_path, text)

    # Access first element - raises IndexError if empty
    first_id = ids[0]
    assert first_id >= 0

    decoded = spm.decode_ids(model_path, ids)
    assert decoded == text
