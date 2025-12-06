from __future__ import annotations

from ...contracts.dataset import DatasetBuilder, DatasetConfig
from ...services.training.dataset_builder import split_corpus_files


class LocalTextDatasetBuilder(DatasetBuilder):
    def split(
        self: LocalTextDatasetBuilder, cfg: DatasetConfig
    ) -> tuple[list[str], list[str], list[str]]:
        return split_corpus_files(cfg)
