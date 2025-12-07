from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizers_get_unknown_status_and_no_stats() -> None:
    app = create_app(load_settings())
    container: ServiceContainer = app.state.container
    # Use isolated fake redis with no keys set
    fake = FakeRedis()
    container.redis = fake
    client = TestClient(app)

    tok_id = "tok-unknown"
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
    assert body["status"] == "unknown"
    assert body["coverage"] is None and body["oov_rate"] is None and body["token_count"] is None
    fake.assert_only_called({"get"})
