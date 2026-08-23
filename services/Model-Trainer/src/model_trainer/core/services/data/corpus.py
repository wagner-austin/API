from __future__ import annotations

import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for


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
