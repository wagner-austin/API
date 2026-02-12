"""Tests for tokenizer loader module."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from model_trainer.core.services.tokenizer.loader import (
    detect_tokenizer_kind,
    load_tokenizer_from_dir,
    load_tokenizer_from_path,
)


def _make_bpe_token(token_id: int, content: str) -> dict[str, bool | int | str]:
    """Create a BPE added_token entry."""
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


class TestDetectTokenizerKind:
    """Tests for detect_tokenizer_kind function."""

    def test_detects_bpe_from_tokenizer_json(self, tmp_path: Path) -> None:
        """Detect BPE tokenizer from tokenizer.json without kind field."""
        tok_json = tmp_path / "tokenizer.json"
        tok_json.write_text(dump_json_str({"vocab": {"a": 0, "b": 1}}))

        kind = detect_tokenizer_kind(str(tmp_path))

        assert kind == "bpe"

    def test_detects_char_from_tokenizer_json_with_kind(self, tmp_path: Path) -> None:
        """Detect char tokenizer from tokenizer.json with kind=char."""
        tok_json = tmp_path / "tokenizer.json"
        tok_json.write_text(dump_json_str({"kind": "char", "vocab": {"a": 0}}))

        kind = detect_tokenizer_kind(str(tmp_path))

        assert kind == "char"

    def test_detects_sentencepiece_from_model_file(self, tmp_path: Path) -> None:
        """Detect SentencePiece from tokenizer.model file."""
        tok_model = tmp_path / "tokenizer.model"
        tok_model.write_bytes(b"fake spm model")

        kind = detect_tokenizer_kind(str(tmp_path))

        assert kind == "sentencepiece"

    def test_raises_when_no_artifacts_found(self, tmp_path: Path) -> None:
        """Raise AppError when no tokenizer artifacts exist."""
        with pytest.raises(AppError, match="No tokenizer artifacts found in"):
            detect_tokenizer_kind(str(tmp_path))


class TestLoadTokenizerFromDir:
    """Tests for load_tokenizer_from_dir function."""

    def test_loads_bpe_tokenizer(self, tmp_path: Path) -> None:
        """Load BPE tokenizer from directory."""
        tok_json = tmp_path / "tokenizer.json"
        tok_data = {
            "model": {"type": "BPE", "vocab": {}, "merges": []},
            "added_tokens": [
                _make_bpe_token(0, "[PAD]"),
                _make_bpe_token(1, "[UNK]"),
            ],
        }
        tok_json.write_text(dump_json_str(tok_data))

        handle = load_tokenizer_from_dir(str(tmp_path))

        assert handle.get_vocab_size() >= 0

    def test_loads_char_tokenizer(self, tmp_path: Path) -> None:
        """Load char tokenizer from directory."""
        tok_json = tmp_path / "tokenizer.json"
        tok_data = {
            "kind": "char",
            "specials": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            "vocab": {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "b": 5},
        }
        tok_json.write_text(dump_json_str(tok_data))

        handle = load_tokenizer_from_dir(str(tmp_path))

        assert handle.get_vocab_size() == 6

    def test_raises_when_no_artifacts(self, tmp_path: Path) -> None:
        """Raise AppError when directory has no tokenizer artifacts."""
        with pytest.raises(AppError, match="No tokenizer artifacts found in"):
            load_tokenizer_from_dir(str(tmp_path))

    def test_loads_sentencepiece_tokenizer(self, tmp_path: Path) -> None:
        """Load SentencePiece tokenizer from directory."""
        tok_model = tmp_path / "tokenizer.model"
        tok_model.write_bytes(b"fake spm model")
        tok_vocab = tmp_path / "tokenizer.vocab"
        tok_vocab.write_text("<unk>\t0\n<s>\t0\n</s>\t0\nhello\t-1\n", encoding="utf-8")

        handle = load_tokenizer_from_dir(str(tmp_path))

        assert handle.get_vocab_size() == 4


class TestLoadTokenizerFromPath:
    """Tests for load_tokenizer_from_path function."""

    def test_delegates_to_from_dir_for_directory_path(self, tmp_path: Path) -> None:
        """Delegate to load_tokenizer_from_dir when path is a directory."""
        tok_json = tmp_path / "tokenizer.json"
        tok_data = {
            "kind": "char",
            "specials": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            "vocab": {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "x": 4},
        }
        tok_json.write_text(dump_json_str(tok_data))

        handle = load_tokenizer_from_path(str(tmp_path))

        assert handle.get_vocab_size() == 5

    def test_loads_bpe_from_json_file_path(self, tmp_path: Path) -> None:
        """Load BPE tokenizer from tokenizer.json file path."""
        tok_json = tmp_path / "tokenizer.json"
        tok_data = {
            "model": {"type": "BPE", "vocab": {}, "merges": []},
            "added_tokens": [_make_bpe_token(0, "[PAD]")],
        }
        tok_json.write_text(dump_json_str(tok_data))

        handle = load_tokenizer_from_path(str(tok_json))

        assert handle.get_vocab_size() >= 0

    def test_loads_char_from_json_file_path(self, tmp_path: Path) -> None:
        """Load char tokenizer from tokenizer.json file path with kind=char."""
        tok_json = tmp_path / "tokenizer.json"
        tok_data = {
            "kind": "char",
            "specials": ["[PAD]", "[UNK]"],
            "vocab": {"[PAD]": 0, "[UNK]": 1, "c": 2},
        }
        tok_json.write_text(dump_json_str(tok_data))

        handle = load_tokenizer_from_path(str(tok_json))

        assert handle.get_vocab_size() == 3

    def test_loads_sentencepiece_from_model_file_path(self, tmp_path: Path) -> None:
        """Load SentencePiece tokenizer from tokenizer.model file path."""
        tok_model = tmp_path / "tokenizer.model"
        tok_model.write_bytes(b"fake spm model")
        tok_vocab = tmp_path / "tokenizer.vocab"
        tok_vocab.write_text("<unk>\t0\n<s>\t0\n</s>\t0\n", encoding="utf-8")

        handle = load_tokenizer_from_path(str(tok_model))

        assert handle.get_vocab_size() == 3

    def test_raises_for_unrecognized_file_extension(self, tmp_path: Path) -> None:
        """Raise AppError for unrecognized file extension."""
        unknown_file = tmp_path / "tokenizer.txt"
        unknown_file.write_text("some content")

        with pytest.raises(AppError, match="Unrecognized tokenizer artifact"):
            load_tokenizer_from_path(str(unknown_file))
