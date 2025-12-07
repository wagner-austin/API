from __future__ import annotations

from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizers_get_with_invalid_stats_json_returns_none_fields() -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake

    tok_id = "tok-bad-stats"
    fake.set(f"tokenizer:{tok_id}:status", "completed")
    fake.set(f"tokenizer:{tok_id}:stats", "[]")  # not a dict; should be ignored

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

    obj_raw: dict[str, str | int | float | bool | None] = r.json()
    assert isinstance(obj_raw, dict) and "tokenizer_id" in obj_raw
    obj: _TokInfo = {
        "tokenizer_id": str(obj_raw.get("tokenizer_id", "")),
        "artifact_path": str(obj_raw.get("artifact_path", "")),
        "status": str(obj_raw.get("status", "")),
        "coverage": None,
        "oov_rate": None,
        "token_count": None,
        "char_coverage": None,
    }
    assert obj["coverage"] is None
    assert obj["oov_rate"] is None
    assert obj["token_count"] is None
    assert obj["char_coverage"] is None
    fake.assert_only_called({"set", "get"})
