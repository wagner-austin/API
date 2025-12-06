from __future__ import annotations

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from ...contracts.dataset import DatasetConfig
from ...encoding import Encoded, Encoder
from ..data.corpus import list_text_files


def split_corpus_files(cfg: DatasetConfig) -> tuple[list[str], list[str], list[str]]:
    """Split corpus files into train/val/test sets.

    Files are split in order: train | val | test.
    The split ratios are controlled by holdout_fraction (val) and test_split_ratio (test).

    Args:
        cfg: Dataset configuration with corpus_path, holdout_fraction, test_split_ratio.

    Returns:
        Tuple of (train_files, val_files, test_files).

    Raises:
        AppError: If no text files found under corpus_path (CORPUS_EMPTY).
    """
    files = list_text_files(cfg.corpus_path)
    if not files:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"No text files found under {cfg.corpus_path}",
            model_trainer_status_for(ModelTrainerErrorCode.CORPUS_EMPTY),
        )

    n = len(files)

    # Calculate split counts (ensure at least 1 file per non-zero split)
    test_n = max(1, int(n * cfg.test_split_ratio)) if cfg.test_split_ratio > 0 else 0
    val_n = max(1, int(n * cfg.holdout_fraction)) if cfg.holdout_fraction > 0 else 0

    # Handle edge case: not enough files for all splits
    if test_n + val_n >= n:
        # If only 1 file, use it for all splits
        if n == 1:
            return files, files, files
        # Otherwise, give priority to train, then val, then test
        test_n = min(test_n, max(0, n - 2))
        val_n = min(val_n, max(0, n - test_n - 1))

    # Split: train | val | test (from end of list)
    test_files = files[-test_n:] if test_n > 0 else []
    remaining = files[:-test_n] if test_n > 0 else files
    val_files = remaining[-val_n:] if val_n > 0 else []
    train_files = remaining[:-val_n] if val_n > 0 and len(remaining) > val_n else remaining

    return train_files, val_files, test_files


class CausalLMDataset:
    def __init__(
        self: CausalLMDataset,
        *,
        files: list[str],
        tokenizer: Encoder,
        max_len: int,
        eos_id: int,
        pad_id: int,
    ) -> None:
        self._ids: list[int] = []
        for fp in files:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    enc: Encoded = tokenizer.encode(s)
                    self._ids.extend([*enc.ids, eos_id])
        self._max_len = max_len
        self._pad_id = pad_id

    def __len__(self: CausalLMDataset) -> int:
        if not self._ids:
            return 0
        # Number of chunks, include partial trailing chunk
        return max(1, (len(self._ids) + self._max_len - 1) // self._max_len)

    def __getitem__(self: CausalLMDataset, idx: int) -> torch.Tensor:
        start = idx * self._max_len
        end = start + self._max_len
        chunk = self._ids[start:end]
        if len(chunk) < self._max_len:
            pad = [self._pad_id] * (self._max_len - len(chunk))
            chunk = chunk + pad
        return torch.tensor(chunk, dtype=torch.long)
