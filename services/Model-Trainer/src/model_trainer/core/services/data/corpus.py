from __future__ import annotations

import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final, TextIO

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import InvalidJsonError, JSONValue, load_json_str


@contextmanager
def open_corpus(path: str) -> Generator[TextIO, None, None]:
    """Open a corpus file, refusing bytes that are not UTF-8.

    Every corpus read in this service goes through here. The readers used to
    pass ``errors="ignore"``, which DROPS each byte that will not decode: a
    corpus saved as cp1252 or UTF-16, or truncated mid-sequence, was read as
    mangled text and trained on with nothing in the run saying so. The manifest
    recorded the sha256, so the bytes were traceable, but nothing recorded that
    the bytes had not decoded.

    The failure is translated where it surfaces rather than by validating the
    file up front: these corpora run to hundreds of megabytes and a validation
    pass would read every one of them twice.

    Args:
        path: Corpus file to open.

    Yields:
        A text handle reading strict UTF-8.

    Raises:
        AppError: ``CORPUS_NOT_DECODABLE`` when the file is not UTF-8, naming
            the file, because a bare UnicodeDecodeError raised inside a worker
            is hard to trace back to the submission that carried the file.
        OSError: When the file cannot be opened at all.
    """
    with open(path, encoding="utf-8", errors="strict") as handle:
        try:
            yield handle
        except UnicodeDecodeError as undecodable:
            raise AppError(
                ModelTrainerErrorCode.CORPUS_NOT_DECODABLE,
                f"corpus file is not valid UTF-8: {path} ({undecodable.reason})",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_DECODABLE),
            ) from undecodable


def list_text_files(root: str) -> list[str]:
    p = Path(root)
    if p.is_file():
        return [str(p)]
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".txt", ".text")):
                paths.append(str(Path(dirpath) / name))
    return paths


#: Suffix a document-mode corpus file carries.
DOCUMENT_SUFFIXES: Final[tuple[str, ...]] = (".jsonl",)


def list_document_files(root: str) -> list[str]:
    """List the JSONL files a document-mode corpus path names, in sorted order.

    Kept separate from :func:`list_text_files` rather than widening that
    function's suffix tuple. The two formats are read by different readers,
    and a directory holding both would otherwise be handed to whichever
    reader the format field happened to name -- reading half a corpus and
    reporting a full one.

    The listing is SORTED, which :func:`list_text_files` does not do. File
    order decides the train/validation/test partition, because the split is
    taken over the units in the order they are read; ``os.walk`` order is a
    filesystem detail, so an unsorted listing can partition the same corpus
    differently on two machines. Sorting is safe to introduce here because
    no run has used this path yet.

    Args:
        root: Corpus file, or a directory to walk for corpus files.

    Returns:
        Every ``.jsonl`` file under root in sorted order, or root itself
        when root names a file directly.
    """
    p = Path(root)
    if p.is_file():
        return [str(p)]
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(DOCUMENT_SUFFIXES):
                paths.append(str(Path(dirpath) / name))
    return sorted(paths)


def _malformed(path: str, number: int, detail: str) -> AppError[ModelTrainerErrorCode]:
    """Build the error a bad JSONL record raises, naming where it is.

    Args:
        path: File the record came from.
        number: 1-based line number of the record within that file.
        detail: What is wrong with it.

    Returns:
        The AppError to raise, carrying the file and line in its message.
    """
    return AppError(
        ModelTrainerErrorCode.CORPUS_MALFORMED_RECORD,
        f"{path} line {number}: {detail}",
        model_trainer_status_for(ModelTrainerErrorCode.CORPUS_MALFORMED_RECORD),
    )


def _json_type_name(value: JSONValue) -> str:
    """Name a decoded JSON value's type as JSON names it.

    Reports the JSON type rather than the Python one, because the reader of
    this message is looking at a ``.jsonl`` file, not at a traceback.

    Args:
        value: The decoded value to name.

    Returns:
        The JSON type name.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    return "null"


def _record_text(path: str, number: int, raw: str) -> str:
    """Decode one JSONL line into the document text it carries.

    THE ONE ``except`` HERE TRANSLATES, IT DOES NOT RECOVER. A parse failure
    still ends the read; what the catch adds is the file and line, which
    :func:`platform_core.json_utils.load_json_str` cannot know. This is the
    same reason :func:`open_corpus` translates ``UnicodeDecodeError`` into
    ``CORPUS_NOT_DECODABLE``: a bare decode error raised inside a worker,
    against a corpus of thousands of records, names nothing the operator can
    act on. The two structural checks below need no ``except`` at all --
    ``JSONValue`` is a closed union, so ``isinstance`` narrows it directly.

    Args:
        path: File the line came from, for the error message.
        number: 1-based line number within that file.
        raw: The line's text, already decoded as UTF-8 by ``open_corpus``.

    Returns:
        The record's ``text`` field, verbatim, newlines and indentation intact.

    Raises:
        AppError: ``CORPUS_MALFORMED_RECORD`` when the line is not valid JSON,
            is not a JSON object, carries no string ``text``, or carries an
            empty one.
    """
    try:
        record: JSONValue = load_json_str(raw)
    except InvalidJsonError as invalid:
        raise _malformed(path, number, f"is not valid JSON ({invalid})") from invalid
    if not isinstance(record, dict):
        raise _malformed(path, number, f"is a JSON {_json_type_name(record)}, not an object")
    text: JSONValue | None = record.get("text")
    if text is None:
        raise _malformed(path, number, "carries no 'text' field")
    if not isinstance(text, str):
        raise _malformed(path, number, f"has a JSON {_json_type_name(text)} 'text', not a string")
    if not text:
        raise _malformed(
            path,
            number,
            "carries an empty 'text'. An empty document contributes nothing but "
            "an end-of-sequence token, and the emitter excludes empty files, so "
            "a corpus holding one was not built by it",
        )
    return text


def read_corpus_documents(files: Sequence[str]) -> tuple[str, ...]:
    """Read every JSONL record's text, in file order, with bytes intact.

    The line reader strips each line and drops the blank ones, which is
    correct for prose -- a paragraph's surrounding whitespace carries
    nothing -- and fatal for source code, where indentation IS syntax. A
    Python file read as stripped lines does not parse, so a model trained
    on it is being shown text that could never have come from the corpus
    it is supposed to be learning. Document mode therefore takes each
    record's text exactly as the emitter wrote it.

    Blank separator lines BETWEEN records are skipped rather than refused,
    because that is a property of the file's framing rather than of any
    record. A blank ``text`` INSIDE a record is refused; see
    :func:`_record_text`.

    Args:
        files: JSONL paths to read, in the order their records concatenate.

    Returns:
        The ``text`` of every record, in order.

    Raises:
        AppError: ``CORPUS_MALFORMED_RECORD`` when a record is not a JSON
            object carrying a non-empty string ``text``, naming the file
            and 1-based line so the emitter's output can be corrected.
            ``CORPUS_NOT_DECODABLE`` when a file is not UTF-8.
    """
    documents: list[str] = []
    for path in files:
        with open_corpus(path) as handle:
            for number, raw in enumerate(handle, start=1):
                if raw.strip():
                    documents.append(_record_text(path, number, raw))
    return tuple(documents)


def iter_lines(files: Sequence[str]) -> Generator[str, None, None]:
    for fp in files:
        with open_corpus(fp) as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s


def count_lines(files: Sequence[str]) -> int:
    n = 0
    for _ in iter_lines(files):
        n += 1
    return n


def sample_lines(files: Sequence[str], k: int, *, seed: int) -> list[str]:
    from model_trainer.core import _test_hooks

    if k <= 0:
        return []
    rng = _test_hooks.random_factory(seed)
    reservoir: list[str] = []
    for i, s in enumerate(iter_lines(files), start=1):
        if len(reservoir) < k:
            reservoir.append(s)
        else:
            j = rng.randint(1, i)
            if j <= k:
                reservoir[j - 1] = s
    return reservoir
