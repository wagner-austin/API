from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator


def test_tokenizers_enqueue_returns_none_results_in_500(monkeypatch: MonkeyPatch) -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container

    # Force orchestrator to return None on enqueue (patch class method to preserve binding)
    def _enqueue_fail(self: TokenizerOrchestrator, req: str) -> None:
        return None

    monkeypatch.setattr(
        type(container.tokenizer_orchestrator),
        "enqueue_training",
        _enqueue_fail,
        raising=True,
    )

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
