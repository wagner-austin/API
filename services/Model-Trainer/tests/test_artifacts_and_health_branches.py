from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings, load_settings


def _with_artifacts(settings_path: Path) -> Settings:
    base = load_settings()
    base["app"] = {**base["app"], "artifacts_root": str(settings_path)}
    return base


def test_healthz_logs_and_ready_branches(tmp_path: Path) -> None:
    settings = _with_artifacts(tmp_path / "artifacts")
    app = create_app(settings)
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    obj_raw = load_json_str(r.text)
    # Combine isinstance with value check to verify both type and content
    assert isinstance(obj_raw, dict) and obj_raw.get("status") == "ok"


def test_runs_artifact_pointer_not_found(tmp_path: Path) -> None:
    from model_trainer.core.services.container import ServiceContainer

    settings = _with_artifacts(tmp_path / "artifacts")
    app = create_app(settings)
    cont: ServiceContainer = app.state.container
    fake = FakeRedis()
    cont.redis = fake
    cont.training_orchestrator._redis = fake
    client = TestClient(app, raise_server_exceptions=False)
    # Missing pointer returns 404
    r = client.get("/runs/run-xyz/artifact")
    assert r.status_code == 404
    fake.assert_only_called({"get"})


def test_tokenizers_stats_branch(tmp_path: Path) -> None:
    from model_trainer.core.contracts.tokenizer import TokenizerTrainStats
    from model_trainer.core.services.container import ServiceContainer

    settings = _with_artifacts(tmp_path / "artifacts")
    app = create_app(settings)
    client = TestClient(app)
    cont: ServiceContainer = app.state.container
    fake = FakeRedis()
    cont.redis = fake
    tok_id = "t-123"
    stats = TokenizerTrainStats(coverage=1.0, oov_rate=0.0, token_count=1, char_coverage=1.0)
    fake.set(f"tokenizer:{tok_id}:status", "completed")
    fake.set(f"tokenizer:{tok_id}:stats", stats.model_dump_json())
    r = client.get(f"/tokenizers/{tok_id}")
    assert r.status_code == 200
    obj2_raw = load_json_str(r.text)
    # Combine isinstance with value checks
    assert isinstance(obj2_raw, dict) and obj2_raw.get("status") == "completed"
    assert isinstance(obj2_raw, dict) and obj2_raw.get("token_count") == 1
    fake.assert_only_called({"set", "get"})
