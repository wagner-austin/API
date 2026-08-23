from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger

from ...contracts.dataset import CorpusSplit, DatasetConfig
from ...encoding import Encoded, Encoder
from ..data.corpus import list_text_files, open_corpus

_logger: Final = get_logger(__name__)


def read_corpus_lines(files: Sequence[str]) -> tuple[str, ...]:
    """Read every content-bearing line of the given files, in file order.

    Blank lines are dropped here rather than during tokenization so that the
    split fractions are taken over lines that actually carry corpus content.

    Args:
        files: Paths to read, in the order their contents are concatenated.

    Returns:
        The stripped, non-empty lines of every file, concatenated in order.
    """
    lines: list[str] = []
    for path in files:
        with open_corpus(path) as handle:
            for raw in handle:
                stripped = raw.strip()
                if stripped:
                    lines.append(stripped)
    return tuple(lines)


def _partition_size(total: int, ratio: float) -> int:
    """Line count a requested ratio claims, never rounding a wanted split to nothing.

    Args:
        total: Content lines available in the whole corpus.
        ratio: Requested fraction; zero or less requests no partition at all.

    Returns:
        The partition's line count, or zero when no partition was requested.
    """
    if ratio <= 0:
        return 0
    return max(1, int(total * ratio))


def split_corpus(cfg: DatasetConfig) -> CorpusSplit:
    """Partition a corpus into disjoint train, validation and test lines.

    Lines are assigned in corpus order -- train, then validation, then test --
    so a given corpus and set of ratios always produce the same partition.

    Args:
        cfg: Dataset configuration carrying corpus_path, holdout_fraction and
            test_split_ratio.

    Returns:
        The three disjoint partitions. Validation and test are empty when their
        ratio is zero, which is how a caller asks to train without a holdout.

    Raises:
        AppError: ``CORPUS_EMPTY`` when corpus_path holds no text files, or
            holds only blank ones. ``CORPUS_HOLDOUT_UNSATISFIABLE`` when the
            requested fractions would leave no line to train on.
    """
    files = list_text_files(cfg.corpus_path)
    if not files:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"No text files found under {cfg.corpus_path}",
            model_trainer_status_for(ModelTrainerErrorCode.CORPUS_EMPTY),
        )

    lines = read_corpus_lines(files)
    total = len(lines)
    if total == 0:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"The {len(files)} text file(s) under {cfg.corpus_path} hold no non-blank lines",
            model_trainer_status_for(ModelTrainerErrorCode.CORPUS_EMPTY),
        )

    test_n = _partition_size(total, cfg.test_split_ratio)
    val_n = _partition_size(total, cfg.holdout_fraction)
    if test_n + val_n >= total:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_HOLDOUT_UNSATISFIABLE,
            (
                f"A corpus of {total} line(s) cannot yield {val_n} validation and "
                f"{test_n} test line(s) disjoint from its training lines. Lower "
                f"holdout_fraction (={cfg.holdout_fraction}) or test_split_ratio "
                f"(={cfg.test_split_ratio}), enlarge the corpus, or pass 0 for both "
                f"to train without a holdout."
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CORPUS_HOLDOUT_UNSATISFIABLE),
        )

    train_end = total - val_n - test_n
    val_end = train_end + val_n
    return CorpusSplit(
        train=lines[:train_end],
        validation=lines[train_end:val_end],
        test=lines[val_end:],
    )


IGNORE_INDEX = -100


def _split_line(
    line: str,
    tokenizer: Encoder,
    separator: str | None,
) -> tuple[list[int], list[int]]:
    """Tokenize one corpus line, splitting off a masked prefix if configured.

    Args:
        line: The stripped corpus line.
        tokenizer: Encoder used for both halves.
        separator: Marker separator, or None to mask nothing. A line that does
            not contain it is treated as having no prefix, because the corpus
            legitimately mixes marked wiki paragraphs with unmarked dilution.

    Returns:
        Token ids of the prefix (to be excluded from the loss) and of the body.
    """
    if separator is None:
        encoded: Encoded = tokenizer.encode(line)
        return [], list(encoded.ids)

    head, found, tail = line.partition(separator)
    if found == "":
        unmarked: Encoded = tokenizer.encode(line)
        return [], list(unmarked.ids)

    prefix: Encoded = tokenizer.encode(head + separator)
    body: Encoded = tokenizer.encode(tail)
    return list(prefix.ids), list(body.ids)


class CausalLMDataset:
    """Packs a corpus into fixed-length blocks of (input_ids, labels).

    Labels normally equal the inputs. When ``loss_mask_prefix_separator`` is
    set, the part of each line up to and including the first occurrence of that
    separator is treated as metadata: it is still fed to the model as context,
    but its label positions carry :data:`IGNORE_INDEX` so no gradient flows
    from predicting it.

    That distinction matters for corpus-marker experiments. Prepending a domain
    marker to every paragraph and then training on the marker tokens is not the
    same intervention as prepending it and excluding it from the loss, and the
    two have been measured to differ. Splitting the line and tokenising the two
    halves separately is what makes the boundary addressable at all -- encoding
    the joined line gives no way to know where the marker's tokens end, because
    BPE may merge across the seam.
    """

    def __init__(
        self: CausalLMDataset,
        *,
        lines: Sequence[str],
        tokenizer: Encoder,
        max_len: int,
        eos_id: int,
        pad_id: int,
        loss_mask_prefix_separator: str | None = None,
    ) -> None:
        total_lines = len(lines)
        _logger.info(
            "Tokenization started lines=%d",
            total_lines,
            extra={
                "category": "dataset",
                "event": "tokenization_started",
                "total_lines": total_lines,
            },
        )

        self._ids: list[int] = []
        self._labels: list[int] = []
        masked_tokens = 0

        # Log progress every 10% of lines, and never more than ten times.
        log_interval = max(1, total_lines // 10)

        for index, line in enumerate(lines, start=1):
            prefix_ids, body_ids = _split_line(line, tokenizer, loss_mask_prefix_separator)
            self._ids.extend([*prefix_ids, *body_ids, eos_id])
            self._labels.extend(
                [
                    *([IGNORE_INDEX] * len(prefix_ids)),
                    *body_ids,
                    eos_id,
                ]
            )
            masked_tokens += len(prefix_ids)

            if index % log_interval == 0:
                progress_pct = int((index * 100) / total_lines)
                _logger.info(
                    "Tokenization progress lines=%d/%d (%d%%) tokens=%d",
                    index,
                    total_lines,
                    progress_pct,
                    len(self._ids),
                    extra={
                        "category": "dataset",
                        "event": "tokenization_progress",
                        "lines_processed": index,
                        "total_lines": total_lines,
                        "progress_pct": progress_pct,
                        "tokens": len(self._ids),
                    },
                )

        self._max_len = max_len
        self._pad_id = pad_id
        num_chunks = max(1, (len(self._ids) + max_len - 1) // max_len) if self._ids else 0

        # The masked share is reported because the separator cannot tell a
        # marker from the same characters occurring inside ordinary prose. A
        # separator that collides with the corpus masks arbitrary spans of it,
        # which is a different intervention from excluding the marker and would
        # otherwise be invisible. A marker is a few tokens on a paragraph, so a
        # healthy run reports a low single-digit percentage; anything larger
        # means the separator is not discriminating.
        masked_pct = (100.0 * masked_tokens / len(self._ids)) if self._ids else 0.0
        _logger.info(
            "Tokenization completed lines=%d tokens=%d chunks=%d masked=%d (%.2f%%)",
            total_lines,
            len(self._ids),
            num_chunks,
            masked_tokens,
            masked_pct,
            extra={
                "category": "dataset",
                "event": "tokenization_completed",
                "lines": total_lines,
                "tokens": len(self._ids),
                "chunks": num_chunks,
                "max_len": max_len,
                "masked_tokens": masked_tokens,
                "masked_pct": masked_pct,
            },
        )

    def __len__(self: CausalLMDataset) -> int:
        if not self._ids:
            return 0
        # Number of chunks, include partial trailing chunk
        return max(1, (len(self._ids) + self._max_len - 1) // self._max_len)

    def __getitem__(self: CausalLMDataset, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one block as (input_ids, labels).

        Padding is excluded from the loss for the same reason a masked prefix
        is: the model should not be scored on predicting filler. Only the final
        block is ever padded.

        Args:
            idx: Block index.

        Returns:
            The block's input ids and its label ids.
        """
        start = idx * self._max_len
        end = start + self._max_len
        chunk = self._ids[start:end]
        labels = self._labels[start:end]
        if len(chunk) < self._max_len:
            missing = self._max_len - len(chunk)
            chunk = chunk + [self._pad_id] * missing
            labels = labels + [IGNORE_INDEX] * missing
        return (
            torch.tensor(chunk, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
