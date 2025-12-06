from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from model_trainer.core.config.settings import Settings
from model_trainer.core.services.data.corpus_cache_cleanup import (
    CorpusCacheCleanupError,
    CorpusCacheCleanupResult,
)
from model_trainer.maintenance.cleanup import run_corpus_cleanup


def test_corpus_cleanup_success(monkeypatch: MonkeyPatch) -> None:
    class StubService:
        def __init__(self, *, settings: Settings) -> None:
            self.settings = settings

        def clean(self) -> CorpusCacheCleanupResult:
            return CorpusCacheCleanupResult(deleted_files=2, bytes_freed=4096)

    monkeypatch.setattr("model_trainer.maintenance.cleanup.CorpusCacheCleanupService", StubService)
    result = run_corpus_cleanup()
    assert result.deleted_files == 2
    assert result.bytes_freed == 4096


def test_corpus_cleanup_failure(monkeypatch: MonkeyPatch) -> None:
    class StubService:
        def __init__(self, *, settings: Settings) -> None:
            self.settings = settings

        def clean(self) -> CorpusCacheCleanupResult:
            raise CorpusCacheCleanupError("boom")

    monkeypatch.setattr("model_trainer.maintenance.cleanup.CorpusCacheCleanupService", StubService)
    with pytest.raises(CorpusCacheCleanupError):
        _ = run_corpus_cleanup()
