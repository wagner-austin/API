from __future__ import annotations

import pytest

from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import TokenizerCleanupResultProto
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.tokenizer.tokenizer_cleanup import TokenizerCleanupError
from model_trainer.maintenance.cleanup import run_tokenizer_cleanup


class _StubResult:
    """Stub result implementing TokenizerCleanupResultProto."""

    def __init__(self, deleted_tokenizers: int, bytes_freed: int) -> None:
        self._deleted_tokenizers = deleted_tokenizers
        self._bytes_freed = bytes_freed

    @property
    def deleted_tokenizers(self) -> int:
        return self._deleted_tokenizers

    @property
    def bytes_freed(self) -> int:
        return self._bytes_freed


class _StubServiceSuccess:
    """Stub cleanup service that returns success."""

    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings

    def clean(self) -> TokenizerCleanupResultProto:
        return _StubResult(deleted_tokenizers=3, bytes_freed=2048)


class _StubServiceFailure:
    """Stub cleanup service that raises error."""

    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings

    def clean(self) -> TokenizerCleanupResultProto:
        raise TokenizerCleanupError("fail")


def test_tokenizer_cleanup_success() -> None:
    def _factory(*, settings: Settings) -> _StubServiceSuccess:
        return _StubServiceSuccess(settings=settings)

    _test_hooks.tokenizer_cleanup_service_factory = _factory

    result = run_tokenizer_cleanup()
    assert result.deleted_tokenizers == 3
    assert result.bytes_freed == 2048


def test_tokenizer_cleanup_failure() -> None:
    def _factory(*, settings: Settings) -> _StubServiceFailure:
        return _StubServiceFailure(settings=settings)

    _test_hooks.tokenizer_cleanup_service_factory = _factory

    with pytest.raises(TokenizerCleanupError):
        _ = run_tokenizer_cleanup()
