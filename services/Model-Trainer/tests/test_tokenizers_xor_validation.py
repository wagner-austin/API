from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizers_requires_corpus_file_id_and_forbids_extra(monkeypatch: MonkeyPatch) -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    # Use fake redis to avoid external deps
    fake = FakeRedis()
    container.redis = fake
    container.tokenizer_orchestrator._redis = fake

    calls: list[dict[str, str | int | float | bool | None]] = []

    def _fake_enqueue_tokenizer(payload: dict[str, str | int | float | bool | None]) -> str:
        calls.append(payload)
        return "job-1"

    # Prevent any enqueue from being called on invalid input
    monkeypatch.setattr(container.rq_enqueuer, "enqueue_tokenizer", _fake_enqueue_tokenizer)
    client = TestClient(app)

    # Missing corpus_file_id -> 400 (validation error)
    body = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 1,
        "holdout_fraction": 0.1,
        "seed": 42,
    }
    r = client.post("/tokenizers/train", json=body)
    assert r.status_code == 400
    assert calls == []
    # Extra field corpus_path should be forbidden -> 422
    body2 = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "corpus_path": "/ignored",
        "holdout_fraction": 0.1,
        "seed": 42,
    }
    r2 = client.post("/tokenizers/train", json=body2)
    assert r2.status_code == 422
    assert calls == []
    fake.assert_only_called(set())
