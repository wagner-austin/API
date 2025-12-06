from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizers_get_status_and_stats() -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake

    tok_id = "tok-xyz"
    fake.set(f"tokenizer:{tok_id}:status", "completed")
    fake.set(
        f"tokenizer:{tok_id}:stats",
        '{"coverage":0.9,"oov_rate":0.1,"token_count":1000,"char_coverage":0.8}',
    )

    client = TestClient(app)
    r = client.get(f"/tokenizers/{tok_id}")
    assert r.status_code == 200

    class _TokInfo(TypedDict):
        tokenizer_id: str
        artifact_path: str
        status: str
        coverage: float | None
        oov_rate: float | None
        token_count: int | None
        char_coverage: float | None

    obj_raw = load_json_str(r.text)
    assert isinstance(obj_raw, dict) and "tokenizer_id" in obj_raw
    obj: dict[str, JSONValue] = obj_raw
    cov_v = obj.get("coverage")
    oov_v = obj.get("oov_rate")
    tok_v = obj.get("token_count")
    ch_v = obj.get("char_coverage")
    body: _TokInfo = {
        "tokenizer_id": str(obj.get("tokenizer_id", "")),
        "artifact_path": str(obj.get("artifact_path", "")),
        "status": str(obj.get("status", "")),
        "coverage": float(cov_v) if isinstance(cov_v, int | float) else None,
        "oov_rate": float(oov_v) if isinstance(oov_v, int | float) else None,
        "token_count": int(tok_v) if isinstance(tok_v, int) else None,
        "char_coverage": float(ch_v) if isinstance(ch_v, int | float) else None,
    }
    assert body["tokenizer_id"] == tok_id
    assert body["status"] == "completed"
    assert body["coverage"] == 0.9
    assert body["oov_rate"] == 0.1
    assert body["token_count"] == 1000
    assert body["char_coverage"] == 0.8
