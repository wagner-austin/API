from __future__ import annotations

from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.api.schemas.tokenizers import TokenizerTrainRequest, TokenizerTrainResponse
from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import TokenizerOrchestratorProto
from model_trainer.core.config.settings import load_settings


def test_tokenizers_enqueue_returns_none_results_in_500() -> None:
    """Test that API returns 500 when orchestrator enqueue returns None."""
    fake_redis = FakeRedis()

    def _fake_kv_factory(url: str) -> FakeRedis:
        return fake_redis

    _test_hooks.kv_store_factory = _fake_kv_factory

    # Use hook to make orchestrator return None
    def _enqueue_returns_none(
        orchestrator: TokenizerOrchestratorProto, req: TokenizerTrainRequest
    ) -> TokenizerTrainResponse | None:
        return None

    _test_hooks.tokenizer_enqueue_hook = _enqueue_returns_none

    app = create_app(load_settings())
    client = TestClient(app, raise_server_exceptions=False)

    body = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    r = client.post("/tokenizers/train", json=body)
    assert r.status_code == 500
    # Enqueue hook returns None, so no Redis calls should happen for status
    fake_redis.assert_only_called(set())
