from __future__ import annotations

from ...contracts.dataset import CorpusSplit, DatasetBuilder, DatasetConfig
from ...services.training.dataset_builder import split_corpus


class LocalTextDatasetBuilder(DatasetBuilder):
    def split(self: LocalTextDatasetBuilder, cfg: DatasetConfig) -> CorpusSplit:
        """Partition a corpus held in local text files.

        Args:
            cfg: Dataset configuration with corpus path and split ratios.

        Returns:
            The three disjoint partitions, as corpus lines.
        """
        return split_corpus(cfg)
