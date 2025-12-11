from __future__ import annotations

import pytest

from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import CorpusCacheCleanupResultProto
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.data.corpus_cache_cleanup import (
    CorpusCacheCleanupError,
)
from model_trainer.maintenance.cleanup import run_corpus_cleanup


class _StubResult:
    """Stub result implementing CorpusCacheCleanupResultProto."""

    def __init__(self, deleted_files: int, bytes_freed: int) -> None:
        self._deleted_files = deleted_files
        self._bytes_freed = bytes_freed

    @property
    def deleted_files(self) -> int:
        return self._deleted_files

    @property
    def bytes_freed(self) -> int:
        return self._bytes_freed


class _StubServiceSuccess:
    """Stub cleanup service that returns success."""

    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings

    def clean(self) -> CorpusCacheCleanupResultProto:
        return _StubResult(deleted_files=2, bytes_freed=4096)


class _StubServiceFailure:
    """Stub cleanup service that raises error."""

    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings

    def clean(self) -> CorpusCacheCleanupResultProto:
        raise CorpusCacheCleanupError("boom")


def test_corpus_cleanup_success() -> None:
    def _factory(*, settings: Settings) -> _StubServiceSuccess:
        return _StubServiceSuccess(settings=settings)

    _test_hooks.corpus_cache_cleanup_service_factory = _factory

    result = run_corpus_cleanup()
    assert result.deleted_files == 2
    assert result.bytes_freed == 4096


def test_corpus_cleanup_failure() -> None:
    def _factory(*, settings: Settings) -> _StubServiceFailure:
        return _StubServiceFailure(settings=settings)

    _test_hooks.corpus_cache_cleanup_service_factory = _factory

    with pytest.raises(CorpusCacheCleanupError):
        _ = run_corpus_cleanup()
