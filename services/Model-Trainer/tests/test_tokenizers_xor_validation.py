from __future__ import annotations

from fastapi.testclient import TestClient
from platform_workers.redis import _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue
from platform_workers.testing import FakeQueue, FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import load_settings


def test_tokenizers_requires_corpus_file_id_and_forbids_extra() -> None:
    """Test validation: corpus_file_id required, extra fields forbidden."""
    fake_queue = FakeQueue()
    fake_redis = FakeRedis()

    def _fake_queue_factory(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    def _fake_kv_factory(url: str) -> FakeRedis:
        return fake_redis

    _test_hooks.rq_queue_factory = _fake_queue_factory
    _test_hooks.kv_store_factory = _fake_kv_factory

    app = create_app(load_settings())
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
    assert len(fake_queue.jobs) == 0

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
    assert len(fake_queue.jobs) == 0
    fake_redis.assert_only_called(set())
