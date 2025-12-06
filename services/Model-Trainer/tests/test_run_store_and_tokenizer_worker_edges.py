from __future__ import annotations

from pathlib import Path

import pytest
from platform_workers.testing import FakeRedis

from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.infra.storage.run_store import RunStore
from model_trainer.worker import tokenizer_worker as tkw


def test_run_store_manifest_write_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rs = RunStore(artifacts_root=str(tmp_path / "artifacts"))
    # Force dump_json_str to raise OSError to trigger error path
    from platform_core.json_utils import _JSONInputValue

    def _fail_dump(value: _JSONInputValue, *, compact: bool = True) -> str:
        raise OSError("disk full")

    # Patch at the import location in run_store module
    monkeypatch.setattr("model_trainer.infra.storage.run_store.dump_json_str", _fail_dump)
    # We expect the manifest write to either succeed (if error is silently handled)
    # or raise OSError (if error propagates).
    # This test exercises the code path for manifest write errors.
    with pytest.raises(OSError, match="disk full"):
        _ = rs.create_run("gpt2", "small")


def test_tokenizer_worker_sentencepiece_missing_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRedis()
    from model_trainer.core.config.settings import Settings

    def _fake_redis(settings: Settings) -> FakeRedis:
        return fake

    monkeypatch.setattr(tkw, "_redis_client", _fake_redis)
    # Artifacts root
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    payload: TokenizerTrainPayload = {
        "tokenizer_id": "t1",
        "method": "sentencepiece",
        "vocab_size": 100,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    # Worker fetcher returns a local path
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            p = tmp_path / "corpus.txt"
            p.write_text("x\n", encoding="utf-8")
            return p

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    import shutil

    def _which_none(name: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _which_none)
    tkw.process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:t1:status") == "failed"
