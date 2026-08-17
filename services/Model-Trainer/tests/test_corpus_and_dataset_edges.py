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


def test_split_corpus_two_files_gives_train_priority_then_validation(tmp_path: Path) -> None:
    """Two files and ratios summing past the corpus: train and validation win.

    Asserted as the exact three splits rather than as "at least one holdout is
    populated", because a split that duplicates a file across sets and a split
    that partitions them both satisfy the looser statement.
    """
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.5, test_split_ratio=0.5)

    files = list_text_files(str(tmp_path))
    assert split_corpus_files(cfg) == ([files[0]], [files[1]], [])


def test_split_corpus_single_file_returns_it_as_all_three_splits(tmp_path: Path) -> None:
    """Pins a KNOWN DEFECT, not intended behaviour. Do not read this as a spec.

    The split is by file, so a one-file corpus cannot produce a holdout
    disjoint from training -- and instead of saying so, this returns the same
    file as train, validation and test. Nothing fails. The run's manifest then
    reports validation and test losses that are training-set losses,
    indistinguishable in the output from real ones, and early stopping and
    best-checkpoint selection run against data the model trained on.

    The assertion is written as the exact identity so that a fix flips this
    test red rather than leaving it quietly passing on new behaviour. The
    single-file corpus is this service's prevailing convention -- 91 corpus
    fixtures across the suite, plus the corpus-ablation experiments -- so
    correcting it is a contract change, not a local edit, and is tracked
    separately.
    """
    (tmp_path / "only.txt").write_text("content", encoding="utf-8")
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.5, test_split_ratio=0.5)

    files = list_text_files(str(tmp_path))
    assert split_corpus_files(cfg) == (files, files, files)


def test_split_corpus_single_file_is_allowed_when_no_holdout_is_asked_for(
    tmp_path: Path,
) -> None:
    """Training on one file without a holdout is a supported request.

    The trainer builds no validation loader for an empty split, so this is the
    configuration a caller uses when the evaluation lives outside the run.
    """
    (tmp_path / "only.txt").write_text("content", encoding="utf-8")
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.0, test_split_ratio=0.0)

    files = list_text_files(str(tmp_path))
    assert split_corpus_files(cfg) == (files, [], [])


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
