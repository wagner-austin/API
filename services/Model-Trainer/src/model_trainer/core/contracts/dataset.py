from __future__ import annotations

from typing import Protocol


class DatasetConfig:
    """Configuration for dataset splitting."""

    corpus_path: str
    holdout_fraction: float
    test_split_ratio: float

    def __init__(
        self: DatasetConfig,
        corpus_path: str,
        holdout_fraction: float = 0.01,
        test_split_ratio: float = 0.15,
    ) -> None:
        """Initialize dataset configuration.

        Args:
            corpus_path: Path to corpus directory or file.
            holdout_fraction: Fraction of files for validation (default 0.01).
            test_split_ratio: Fraction of files for testing (default 0.15).
        """
        self.corpus_path = corpus_path
        self.holdout_fraction = holdout_fraction
        self.test_split_ratio = test_split_ratio


class DatasetBuilder(Protocol):
    """Protocol for dataset builders that split corpus files."""

    def split(self: DatasetBuilder, cfg: DatasetConfig) -> tuple[list[str], list[str], list[str]]:
        """Split corpus files into train/val/test sets.

        Args:
            cfg: Dataset configuration with split ratios.

        Returns:
            Tuple of (train_files, val_files, test_files).
        """
        ...
