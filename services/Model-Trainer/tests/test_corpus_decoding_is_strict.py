from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.services.data.corpus import iter_lines, open_corpus
from model_trainer.core.services.training.dataset_builder import read_corpus_lines

# 0x93 and 0x94 are cp1252 smart quotes. They are not valid UTF-8 anywhere, so
# a corpus saved from Word or Notepad on a Windows box carries them. Read with
# errors="ignore" they simply vanish and the surrounding text trains as though
# the document had always been written without them.
CP1252_SMART_QUOTES = b"the model said \x93hello\x94 to the world\n"

# A three-byte sequence cut after its first byte, which is what a corpus
# truncated by a failed copy or a full disk looks like at the tail.
TRUNCATED_MULTIBYTE = b"complete line\nincomplete " + b"\xe2\x80"


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_open_corpus_refuses_cp1252_bytes(tmp_path: Path) -> None:
    """A cp1252 corpus must fail, not lose its punctuation silently."""
    path = _write_bytes(tmp_path / "smart_quotes.txt", CP1252_SMART_QUOTES)

    with pytest.raises(AppError) as raised, open_corpus(str(path)) as handle:
        handle.read()

    err: AppError[ModelTrainerErrorCode] = raised.value
    assert err.code == ModelTrainerErrorCode.CORPUS_NOT_DECODABLE


def test_open_corpus_error_names_the_file(tmp_path: Path) -> None:
    """The message has to be traceable back to a submission from a worker log.

    A bare UnicodeDecodeError says only that some byte somewhere failed, and it
    is raised deep inside a worker. The file name is what lets the next person
    find the corpus. The decoder's own reason travels with it; the offset it
    reports is relative to the buffered read, not to the file, so it is not
    claimed as a file position.
    """
    path = _write_bytes(tmp_path / "smart_quotes.txt", CP1252_SMART_QUOTES)

    with pytest.raises(AppError) as raised, open_corpus(str(path)) as handle:
        handle.read()

    err: AppError[ModelTrainerErrorCode] = raised.value
    message = str(err.message)
    assert "smart_quotes.txt" in message
    assert "not valid UTF-8" in message


def test_open_corpus_refuses_a_truncated_multibyte_sequence(tmp_path: Path) -> None:
    """Truncation mid-character is the failure the manifest sha256 cannot see."""
    path = _write_bytes(tmp_path / "truncated.txt", TRUNCATED_MULTIBYTE)

    with pytest.raises(AppError) as raised, open_corpus(str(path)) as handle:
        handle.read()

    err: AppError[ModelTrainerErrorCode] = raised.value
    assert err.code == ModelTrainerErrorCode.CORPUS_NOT_DECODABLE


def test_open_corpus_reads_valid_utf8_including_non_ascii(tmp_path: Path) -> None:
    """Strict decoding must not reject legitimate multibyte text.

    The measured corpora are 755 MB of scraped web text carrying accents, CJK
    and emoji. Refusing those would be a worse defect than the one being fixed.
    """
    path = _write_bytes(
        tmp_path / "valid.txt",
        "café naïve\n漢字\n\U0001f600\n".encode(),
    )

    with open_corpus(str(path)) as handle:
        assert handle.read().splitlines() == ["café naïve", "漢字", "😀"]


def test_read_corpus_lines_propagates_the_refusal(tmp_path: Path) -> None:
    """The training read path must not soften what open_corpus refused."""
    path = _write_bytes(tmp_path / "smart_quotes.txt", CP1252_SMART_QUOTES)

    with pytest.raises(AppError) as raised:
        read_corpus_lines([str(path)])

    err: AppError[ModelTrainerErrorCode] = raised.value
    assert err.code == ModelTrainerErrorCode.CORPUS_NOT_DECODABLE


def test_iter_lines_propagates_the_refusal(tmp_path: Path) -> None:
    """So must the tokenizer read path, which uses a different reader."""
    path = _write_bytes(tmp_path / "smart_quotes.txt", CP1252_SMART_QUOTES)

    with pytest.raises(AppError) as raised:
        list(iter_lines([str(path)]))

    err: AppError[ModelTrainerErrorCode] = raised.value
    assert err.code == ModelTrainerErrorCode.CORPUS_NOT_DECODABLE
