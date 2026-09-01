from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.services.data.corpus import list_text_files, sample_lines
from model_trainer.core.services.training.dataset_builder import (
    CausalLMDataset,
    read_corpus_lines,
    split_corpus,
)


class _Tok:
    """Encoder that emits one id per character, so token counts are readable."""

    class _Enc:
        def __init__(self: _Tok._Enc, ids: list[int]) -> None:
            self._ids = ids

        @property
        def ids(self: _Tok._Enc) -> list[int]:
            return self._ids

    def encode(self: _Tok, text: str) -> _Tok._Enc:
        return _Tok._Enc([ord(ch) for ch in text])

    def token_to_id(self: _Tok, token: str) -> int | None:
        return 0

    def get_vocab_size(self: _Tok) -> int:
        return 1

    def decode(self: _Tok, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class _EmptyTok(_Tok):
    """Encoder that emits no ids at all, to exercise the empty-corpus paths."""

    def encode(self: _EmptyTok, text: str) -> _Tok._Enc:
        return _Tok._Enc([])


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


def test_read_corpus_lines_concatenates_in_file_order_and_drops_blanks(
    tmp_path: Path,
) -> None:
    """Blank lines are dropped at read time so split fractions cover real content.

    Asserted as the exact tuple rather than as a length, because a reader that
    kept blank lines and a reader that dropped them agree on nothing else but
    would both satisfy a looser check.
    """
    (tmp_path / "a.txt").write_text("one\n\n  two  \n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("\nthree\n", encoding="utf-8")

    assert read_corpus_lines(list_text_files(str(tmp_path))) == ("one", "two", "three")


def test_split_corpus_no_files_raises_corpus_empty(tmp_path: Path) -> None:
    cfg = DatasetConfig(corpus_path=str(tmp_path), corpus_format="lines", holdout_fraction=0.5)
    with pytest.raises(AppError) as exc:
        split_corpus(cfg)
    assert exc.value.code is ModelTrainerErrorCode.CORPUS_EMPTY


def test_split_corpus_blank_only_files_raise_corpus_empty(tmp_path: Path) -> None:
    """A file of blank lines is an empty corpus, and says so rather than splitting nothing.

    This is a distinct path from "no files at all": the directory listing
    succeeds and the emptiness only appears after reading.
    """
    (tmp_path / "blank.txt").write_text("\n\n   \n", encoding="utf-8")
    cfg = DatasetConfig(
        corpus_path=str(tmp_path), corpus_format="lines", holdout_fraction=0.1, test_split_ratio=0.1
    )

    with pytest.raises(AppError) as exc:
        split_corpus(cfg)
    assert exc.value.code is ModelTrainerErrorCode.CORPUS_EMPTY


def test_split_corpus_single_file_partitions_its_lines_disjointly(tmp_path: Path) -> None:
    """The headline fix: one file yields a real holdout, not itself three times.

    Ten lines at 0.1/0.2 gives one validation line and two test lines, leaving
    seven to train on. Asserted as the exact three tuples, because the defect
    this replaces -- returning the whole corpus as all three partitions -- also
    satisfies "validation is non-empty".
    """
    lines = [f"line-{i}" for i in range(10)]
    (tmp_path / "only.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = DatasetConfig(
        corpus_path=str(tmp_path), corpus_format="lines", holdout_fraction=0.1, test_split_ratio=0.2
    )

    split = split_corpus(cfg)

    assert split["train"] == tuple(lines[:7])
    assert split["validation"] == (lines[7],)
    assert split["test"] == (lines[8], lines[9])


def test_split_corpus_partitions_share_no_line(tmp_path: Path) -> None:
    """Disjointness is the property the defect violated, so it is asserted directly."""
    lines = [f"line-{i}" for i in range(100)]
    (tmp_path / "only.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = DatasetConfig(
        corpus_path=str(tmp_path),
        corpus_format="lines",
        holdout_fraction=0.05,
        test_split_ratio=0.15,
    )

    split = split_corpus(cfg)

    train, validation, test = split["train"], split["validation"], split["test"]
    assert set(train) & set(validation) == set()
    assert set(train) & set(test) == set()
    assert set(validation) & set(test) == set()
    assert len(train) + len(validation) + len(test) == len(lines)


def test_split_corpus_spans_multiple_files_as_one_corpus(tmp_path: Path) -> None:
    """A corpus is one body of text however many files it occupies.

    Three files of four lines each split exactly as one twelve-line file would,
    which is what makes the fraction mean the same thing in both layouts.
    """
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("\n".join(f"{name}-{i}" for i in range(4)) + "\n", "utf-8")
    cfg = DatasetConfig(
        corpus_path=str(tmp_path),
        corpus_format="lines",
        holdout_fraction=0.25,
        test_split_ratio=0.25,
    )

    split = split_corpus(cfg)

    assert len(split["train"]) == 6
    assert len(split["validation"]) == 3
    assert len(split["test"]) == 3


def test_split_corpus_rounds_a_wanted_partition_up_to_one_line(tmp_path: Path) -> None:
    """A requested holdout too small to round to a whole line still gets one.

    Ten lines at 0.01 is 0.1 lines. Truncating to zero would silently grant the
    caller no validation set while reporting success, which is the class of
    failure this whole change exists to remove.
    """
    lines = [f"line-{i}" for i in range(10)]
    (tmp_path / "only.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = DatasetConfig(
        corpus_path=str(tmp_path),
        corpus_format="lines",
        holdout_fraction=0.01,
        test_split_ratio=0.0,
    )

    split = split_corpus(cfg)

    assert split["validation"] == (lines[9],)
    assert split["train"] == tuple(lines[:9])
    assert split["test"] == ()


def test_split_corpus_raises_when_the_holdout_would_leave_no_training_lines(
    tmp_path: Path,
) -> None:
    """A corpus too small to split is an error with its own code, not a silent overlap.

    One line cannot yield a validation line disjoint from a training line. The
    old behaviour returned that one line as all three partitions.
    """
    (tmp_path / "only.txt").write_text("just one line\n", encoding="utf-8")
    cfg = DatasetConfig(
        corpus_path=str(tmp_path), corpus_format="lines", holdout_fraction=0.5, test_split_ratio=0.5
    )

    with pytest.raises(AppError) as exc:
        split_corpus(cfg)

    assert exc.value.code is ModelTrainerErrorCode.CORPUS_HOLDOUT_UNSATISFIABLE
    assert "1 line(s)" in exc.value.message


def test_split_corpus_single_file_is_allowed_when_no_holdout_is_asked_for(
    tmp_path: Path,
) -> None:
    """Training on one file without a holdout is a supported request.

    The trainer builds no validation loader for an empty split, so this is the
    configuration a caller uses when the evaluation lives outside the run.
    """
    (tmp_path / "only.txt").write_text("content\n", encoding="utf-8")
    cfg = DatasetConfig(
        corpus_path=str(tmp_path), corpus_format="lines", holdout_fraction=0.0, test_split_ratio=0.0
    )

    split = split_corpus(cfg)

    assert split["train"] == ("content",)
    assert split["validation"] == ()
    assert split["test"] == ()


def test_dataset_len_zero_on_empty_lines() -> None:
    ds = CausalLMDataset(lines=(), tokenizer=_EmptyTok(), max_len=8, eos_id=1, pad_id=0)
    assert len(ds) == 0


def test_dataset_packs_the_lines_it_is_given_and_nothing_else() -> None:
    """The dataset tokenizes exactly its partition, which is how the holdout stays held out.

    Two one-character lines with an eos each is four ids, so a max_len of 4
    packs to exactly one full block with no padding.
    """
    ds = CausalLMDataset(lines=("a", "b"), tokenizer=_Tok(), max_len=4, eos_id=1, pad_id=0)

    assert len(ds) == 1
    input_ids, labels = ds[0]
    expected = [ord("a"), 1, ord("b"), 1]
    assert [int(input_ids[i].item()) for i in range(4)] == expected
    assert [int(labels[i].item()) for i in range(4)] == expected
