from __future__ import annotations

from typing import Protocol, TypedDict


class CorpusSplit(TypedDict):
    """Three disjoint partitions of a corpus, carried as lines rather than paths.

    The partitions are lines because ``holdout_fraction`` and
    ``test_split_ratio`` are fractions of the corpus, not of its file list.
    Splitting by file made a single-file corpus -- the prevailing layout in this
    service and in every experiment run against it -- collapse to the same file
    in all three partitions, so validation loss, early stopping and
    best-checkpoint selection all ran against training data while reporting
    numbers indistinguishable from real ones.
    """

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


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

        Both ratios are fractions of the corpus's content lines, not of its
        file list. A corpus is one logical body of text however many files it
        happens to occupy, so a single-file corpus splits exactly like a
        thousand-file one.

        Args:
            corpus_path: Path to corpus directory or file.
            holdout_fraction: Fraction of corpus lines held out for validation
                (default 0.01). Zero trains without a validation split.
            test_split_ratio: Fraction of corpus lines held out for testing
                (default 0.15). Zero trains without a test split.
        """
        self.corpus_path = corpus_path
        self.holdout_fraction = holdout_fraction
        self.test_split_ratio = test_split_ratio


class DatasetBuilder(Protocol):
    """Protocol for dataset builders that partition a corpus."""

    def split(self: DatasetBuilder, cfg: DatasetConfig) -> CorpusSplit:
        """Partition a corpus into disjoint train/validation/test lines.

        Args:
            cfg: Dataset configuration with corpus path and split ratios.

        Returns:
            The three disjoint partitions, as corpus lines.
        """
        ...
