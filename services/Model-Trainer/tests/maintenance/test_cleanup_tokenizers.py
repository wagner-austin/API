from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from model_trainer.core.config.settings import Settings
from model_trainer.core.services.tokenizer.tokenizer_cleanup import TokenizerCleanupError
from model_trainer.maintenance.cleanup import run_tokenizer_cleanup


class _TokenizersResult:
    def __init__(self, deleted_tokenizers: int, bytes_freed: int) -> None:
        self.deleted_tokenizers = deleted_tokenizers
        self.bytes_freed = bytes_freed


def test_tokenizer_cleanup_success(monkeypatch: MonkeyPatch) -> None:
    class StubService:
        def __init__(self, *, settings: Settings) -> None:
            self.settings = settings

        def clean(self) -> _TokenizersResult:
            return _TokenizersResult(deleted_tokenizers=3, bytes_freed=2048)

    monkeypatch.setattr("model_trainer.maintenance.cleanup.TokenizerCleanupService", StubService)
    result = run_tokenizer_cleanup()
    assert result.deleted_tokenizers == 3
    assert result.bytes_freed == 2048


def test_tokenizer_cleanup_failure(monkeypatch: MonkeyPatch) -> None:
    class StubService:
        def __init__(self, *, settings: Settings) -> None:
            self.settings = settings

        def clean(self) -> _TokenizersResult:
            raise TokenizerCleanupError("fail")

    monkeypatch.setattr("model_trainer.maintenance.cleanup.TokenizerCleanupService", StubService)
    with pytest.raises(TokenizerCleanupError):
        _ = run_tokenizer_cleanup()
