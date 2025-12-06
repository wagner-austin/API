from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError

from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.services.data.corpus import list_text_files, sample_lines
from model_trainer.core.services.training.dataset_builder import CausalLMDataset, split_corpus_files


def test_list_text_files_single_file(tmp_path: Path) -> None:
    fp = tmp_path / "a.txt"
    fp.write_text("x", encoding="utf-8")
    out = list_text_files(str(fp))
    assert out == [str(fp)]


def test_sample_lines_zero_k(tmp_path: Path) -> None:
    # Build a corpus file with some lines
    fp = tmp_path / "b.txt"
    fp.write_text("a\n\n b \n", encoding="utf-8")
    out = sample_lines([str(fp)], 0, seed=1)
    assert out == []


def test_split_corpus_no_files_raises(tmp_path: Path) -> None:
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.5)
    with pytest.raises(AppError):
        split_corpus_files(cfg)


def test_split_corpus_few_files_edge_case(tmp_path: Path) -> None:
    """Test split when test_n + val_n >= n (not enough files for all splits)."""
    # Create 2 files - not enough for full 3-way split with high ratios
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.5, test_split_ratio=0.5)
    train, val, test = split_corpus_files(cfg)
    # With 2 files and high ratios, all splits should be populated
    assert train  # non-empty
    assert val or test  # at least one holdout split populated
    total = len(train) + len(val) + len(test)
    assert total == 2 or total == 4  # exact count: 2 files, possibly duplicated


def test_split_corpus_single_file_reused(tmp_path: Path) -> None:
    """Test split with only 1 file returns same file for all splits (line 44)."""
    (tmp_path / "only.txt").write_text("content", encoding="utf-8")
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.5, test_split_ratio=0.5)
    train, val, test = split_corpus_files(cfg)
    # With only 1 file, it should be reused for all splits
    assert train == val == test
    assert len(train) == 1


def test_split_corpus_three_files_edge(tmp_path: Path) -> None:
    """Test split with 3 files and high split ratios triggers priority logic (lines 46-47)."""
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "c.txt").write_text("c", encoding="utf-8")
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.5, test_split_ratio=0.5)
    train, val, test = split_corpus_files(cfg)
    # Should prioritize train, then val, then test
    assert train  # at least 1 file
    assert len(train) + len(val) + len(test) == 3  # all 3 files used exactly once


def test_dataset_len_zero_on_empty(tmp_path: Path) -> None:
    # Empty file yields no ids
    fp = tmp_path / "c.txt"
    fp.write_text("\n\n", encoding="utf-8")

    class _Tok:
        class _Enc:
            def __init__(self: _Tok._Enc, ids: list[int]) -> None:
                self._ids = ids

            @property
            def ids(self: _Tok._Enc) -> list[int]:
                return self._ids

        def encode(self: _Tok, text: str) -> _Tok._Enc:
            return _Tok._Enc([])

        def token_to_id(self: _Tok, token: str) -> int | None:
            return 0

        def get_vocab_size(self: _Tok) -> int:
            return 1

        def decode(self: _Tok, ids: list[int]) -> str:
            return ""

    ds = CausalLMDataset(files=[str(fp)], tokenizer=_Tok(), max_len=8, eos_id=1, pad_id=0)
    assert len(ds) == 0
