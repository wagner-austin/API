"""Turn selected files into corpus documents, a split, and a manifest.

Two decisions here are consequences of how the corpus will be consumed.

*The holdout split is by file, never by line or block.* The guard-pass
evaluation builds its prompts from held-out files, and a split that let half
of a file into training would leak the other half's style verbatim -- the
same defect the trainer's own validation split has when given a single-file
corpus. Holding out whole files is the only split the evaluation can trust.

*One JSONL record per file, not lines of text.* The trainer's existing
line-oriented corpus reader strips each line, which destroys indentation --
the single most load-bearing character class in Python source. Documents are
therefore carried as JSON strings, where newlines and leading whitespace
survive byte-exactly, and the consumer decodes documents instead of splitting
lines.
"""

from __future__ import annotations

import hashlib
import pathlib
import random
from collections.abc import Sequence

from platform_core.json_utils import dump_json_str

from code_corpus.contracts.corpus import (
    CodeCorpusManifest,
    LanguageStats,
    RepoPin,
    SourceFileRecord,
    encode_source_file_record,
)
from code_corpus.core import _test_hooks
from code_corpus.core.select import SelectedFile


def render_document(file: SelectedFile) -> str:
    """Render a selected file as its training document.

    The first line is a comment naming the file's repository and path. The
    header is part of the document on purpose: it fixes the training view at
    emission time, so what a model saw is exactly what the corpus file holds,
    and it gives the model the path context that house structure conventions
    (contracts/, core/, ``_test_hooks``) hang off.

    Args:
        file: File to render.

    Returns:
        The header line followed by the file's content, ending in a newline.
    """
    body = file.text if file.text.endswith("\n") else file.text + "\n"
    return f"# {file.repo}/{file.path}\n{body}"


def dedup_files(files: Sequence[SelectedFile]) -> tuple[list[SelectedFile], int]:
    """Drop files whose content an earlier file already contributed.

    Byte-identical files are common here by design -- the per-project guard
    scripts are lifted verbatim -- and training on N copies of one file
    overweights it by N while speeding forgetting of what generalises.
    The first occurrence wins, so the caller's ordering decides which path
    represents the content.

    Args:
        files: Files in selection order.

    Returns:
        The files kept, in order, and the count of duplicates dropped.
    """
    seen: set[str] = set()
    kept: list[SelectedFile] = []
    duplicates = 0
    for file in files:
        if file.sha256 in seen:
            duplicates += 1
            continue
        seen.add(file.sha256)
        kept.append(file)
    return kept, duplicates


def split_holdout(
    files: Sequence[SelectedFile], *, fraction: float, seed: int
) -> tuple[list[SelectedFile], list[SelectedFile]]:
    """Split files into a shuffled training list and a sorted holdout list.

    One ``random.Random(seed)`` stream drives both the holdout sample and the
    training-order shuffle, so the whole split is reproducible from the
    recorded seed alone. Training order is shuffled to avoid a repository-
    order curriculum; holdout order is sorted by repository and path because
    it is read by evaluation tooling, not trained on.

    Args:
        files: Files to split.
        fraction: Fraction of files to hold out, in ``[0, 1)``. Zero means
            no holdout.
        seed: Seed for the sample and the shuffle.

    Returns:
        The training files and the holdout files.

    Raises:
        ValueError: If the fraction is outside ``[0, 1)``, no files were
            given, or a positive fraction selects zero files -- a silent
            empty holdout would pin the evaluation to nothing.
    """
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError(f"holdout fraction must be in [0, 1), got {fraction}")
    if len(files) == 0:
        raise ValueError("no files to split; nothing to emit")

    count = int(len(files) * fraction)
    if fraction > 0.0 and count == 0:
        raise ValueError(
            f"holdout fraction {fraction} of {len(files)} files selects zero holdout "
            "files; use 0 for no holdout or provide more files"
        )

    rng = random.Random(seed)
    holdout_indices = set(rng.sample(range(len(files)), count))
    train = [file for index, file in enumerate(files) if index not in holdout_indices]
    holdout = [files[index] for index in sorted(holdout_indices)]
    holdout.sort(key=lambda file: (file.repo, file.path))
    rng.shuffle(train)
    return train, holdout


def to_record(file: SelectedFile) -> SourceFileRecord:
    """Build the serialized record for a selected file.

    Args:
        file: File to serialize.

    Returns:
        The record, with ``text`` holding the rendered document and
        ``sha256`` still digesting the raw normalized content.
    """
    return SourceFileRecord(
        repo=file.repo,
        path=file.path,
        language=file.language,
        sha256=file.sha256,
        tokens_approx=file.tokens_approx,
        text=render_document(file),
    )


def jsonl_text(records: Sequence[SourceFileRecord]) -> str:
    """Serialize records as JSON Lines.

    Args:
        records: Records to serialize.

    Returns:
        One compact JSON object per line; empty string for no records.
    """
    return "".join(dump_json_str(encode_source_file_record(record)) + "\n" for record in records)


def language_stats(files: Sequence[SelectedFile]) -> dict[str, LanguageStats]:
    """Total the kept files by language.

    Args:
        files: Every kept file, train and holdout together.

    Returns:
        Per-language file and token totals, keyed in sorted order so the
        manifest is stable across emissions.
    """
    totals: dict[str, LanguageStats] = {}
    for file in files:
        if file.language not in totals:
            totals[file.language] = LanguageStats(files=0, tokens_approx=0)
        totals[file.language]["files"] += 1
        totals[file.language]["tokens_approx"] += file.tokens_approx
    return {language: totals[language] for language in sorted(totals)}


def build_manifest(
    *,
    train_output: pathlib.Path,
    holdout_output: pathlib.Path,
    seed: int,
    holdout_fraction: float,
    repos: Sequence[RepoPin],
    train: Sequence[SelectedFile],
    holdout: Sequence[SelectedFile],
    excluded_generated: int,
    excluded_duplicate: int,
    excluded_empty: int,
) -> CodeCorpusManifest:
    """Describe an emission, digesting the bytes actually written.

    The digests are taken by reading the output files back rather than by
    hashing strings in memory, so they certify what a consumer will read.

    Args:
        train_output: The emitted training JSONL.
        holdout_output: The emitted holdout JSONL.
        seed: Seed behind the split.
        holdout_fraction: Fraction of files held out.
        repos: State of every contributing repository.
        train: Files written to the training output.
        holdout: Files written to the holdout output.
        excluded_generated: Files refused as generator output.
        excluded_duplicate: Files dropped as duplicates.
        excluded_empty: Files refused as whitespace-only.

    Returns:
        The manifest.
    """
    return CodeCorpusManifest(
        train_output=train_output.name,
        train_sha256=hashlib.sha256(_test_hooks.read_bytes(train_output)).hexdigest(),
        holdout_output=holdout_output.name,
        holdout_sha256=hashlib.sha256(_test_hooks.read_bytes(holdout_output)).hexdigest(),
        seed=seed,
        holdout_fraction=holdout_fraction,
        repos=list(repos),
        files_train=len(train),
        files_holdout=len(holdout),
        excluded_generated=excluded_generated,
        excluded_duplicate=excluded_duplicate,
        excluded_empty=excluded_empty,
        languages=language_stats([*train, *holdout]),
        tokens_approx_train=sum(file.tokens_approx for file in train),
        tokens_approx_holdout=sum(file.tokens_approx for file in holdout),
    )


__all__ = [
    "build_manifest",
    "dedup_files",
    "jsonl_text",
    "language_stats",
    "render_document",
    "split_holdout",
    "to_record",
]
