from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizer_enqueue_passes_method(monkeypatch: MonkeyPatch) -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    container.tokenizer_orchestrator._redis = fake

    captured: dict[str, str | int | float | bool | None] = {}

    def _fake_enqueue_tokenizer(payload: dict[str, str | int | float | bool | None]) -> str:
        nonlocal captured
        captured = payload
        return "job-1"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_tokenizer", _fake_enqueue_tokenizer)
    # Stub CorpusFetcher to map file id to a local path
    import tempfile
    from pathlib import Path

    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, file_id: str) -> Path:  # return a valid path
            return Path(tempfile.gettempdir())

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    client = TestClient(app)

    body = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 42,
    }
    r = client.post("/tokenizers/train", json=body)
    assert r.status_code == 200

    class _TokTrainResp(TypedDict):
        tokenizer_id: str
        job_id: str

    obj_raw = load_json_str(r.text)
    assert isinstance(obj_raw, dict) and "tokenizer_id" in obj_raw
    obj: dict[str, JSONValue] = obj_raw
    _resp: _TokTrainResp = {
        "tokenizer_id": str(obj.get("tokenizer_id", "")),
        "job_id": str(obj.get("job_id", "")),
    }
    assert captured.get("method") == "bpe"
