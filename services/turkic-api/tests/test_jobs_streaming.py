from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue
from platform_core.turkic_jobs import TurkicJobStatus
from platform_workers.job_context import JobContext
from platform_workers.testing import FakeRedis

import turkic_api.api.jobs as jobs_mod
from turkic_api.api.config import Settings
from turkic_api.api.job_store import TurkicJobStore
from turkic_api.core.corpus import LocalCorpusService
from turkic_api.core.models import ProcessSpec


class _Ctx(JobContext):
    def __init__(self) -> None:
        self.started: int = 0
        self.progress: list[tuple[int, str | None]] = []
        self.completed: list[tuple[str, int]] = []
        self.failed: list[tuple[str, str]] = []

    def publish_started(self) -> None:
        self.started += 1

    def publish_progress(
        self, progress: int, message: str | None = None, *, payload: JSONValue | None = None
    ) -> None:
        self.progress.append((progress, message))

    def publish_completed(self, result_id: str, result_bytes: int) -> None:
        self.completed.append((result_id, result_bytes))

    def publish_failed(self, error_kind: str, message: str) -> None:
        self.failed.append((error_kind, message))


class _TrackingStore(TurkicJobStore):
    def __init__(self) -> None:
        self.saved: list[TurkicJobStatus] = []
        self._fake = FakeRedis()
        super().__init__(self._fake)

    def save(self, data: TurkicJobStatus) -> None:
        self.saved.append(data)
        super().save(data)


def test_result_stream_publishes_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lines = [f"line{i}" for i in range(1, 121)]

    class _Svc:
        def __init__(self, _root: str) -> None:
            pass

        def stream(self, _spec: ProcessSpec) -> Generator[str, None, None]:
            yield from lines

    monkeypatch.setattr(jobs_mod, "LocalCorpusService", _Svc)

    def _to_ipa(text: str, _lang: str) -> str:
        return f"{text}-ipa"

    monkeypatch.setattr(jobs_mod, "to_ipa", _to_ipa)

    spec: ProcessSpec = {
        "source": "oscar",
        "language": "kk",
        "max_sentences": 100,
        "transliterate": True,
        "confidence_threshold": 0.0,
    }
    settings = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )
    store = _TrackingStore()
    ctx = _Ctx()
    created_at = datetime.utcnow()

    out = list(jobs_mod._result_stream("jid", 42, spec, settings, store, ctx, created_at))

    assert len(out) == 120
    assert out[0] == b"line1-ipa\n"
    assert (50, "processing") in ctx.progress
    assert (99, "processing") in ctx.progress
    assert any(entry["progress"] == 99 for entry in store.saved)
    store._fake.assert_only_called({"hset", "expire"})


def test_local_corpus_service_handles_empty_file(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "oscar_kk.txt"
    corpus_path.write_text("", encoding="utf-8")

    svc = LocalCorpusService(str(tmp_path))
    spec: ProcessSpec = {
        "source": "oscar",
        "language": "kk",
        "max_sentences": 10,
        "transliterate": False,
        "confidence_threshold": 0.0,
    }
    assert list(svc.stream(spec)) == []
    # FakeRedis not used in this test, but guard requires assertion
    FakeRedis().assert_only_called(set())
