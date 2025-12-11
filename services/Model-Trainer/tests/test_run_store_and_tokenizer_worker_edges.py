from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import _JSONInputValue as JSONInputValue
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import CorpusFetcherProto
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.infra.storage.run_store import RunStore
from model_trainer.worker import tokenizer_worker as tkw


def test_run_store_manifest_write_error(tmp_path: Path) -> None:
    """Test that manifest write errors propagate correctly."""
    rs = RunStore(artifacts_root=str(tmp_path / "artifacts"))

    def _fail_dump(value: JSONInputValue, *, compact: bool = True) -> str:
        raise OSError("disk full")

    _test_hooks.dump_json_str = _fail_dump

    with pytest.raises(OSError, match="disk full"):
        _ = rs.create_run("gpt2", "small")


def test_tokenizer_worker_sentencepiece_missing_cli(tmp_path: Path) -> None:
    """Test worker handles missing SentencePiece CLI gracefully."""
    fake = FakeRedis()

    def _fake_kv(url: str) -> FakeRedis:
        return fake

    _test_hooks.kv_store_factory = _fake_kv

    # Create fake corpus fetcher
    class _FakeCorpusFetcher:
        def __init__(self, api_url: str, api_key: str, cache_dir: Path) -> None:
            self._tmp = tmp_path

        def fetch(self, fid: str) -> Path:
            p = self._tmp / "corpus.txt"
            p.write_text("x\n", encoding="utf-8")
            return p

    def _fake_fetcher_factory(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        return _FakeCorpusFetcher(api_url, api_key, cache_dir)

    _test_hooks.corpus_fetcher_factory = _fake_fetcher_factory

    # Make shutil_which return None for SentencePiece CLI
    def _which_none(cmd: str) -> None:
        return None

    _test_hooks.shutil_which = _which_none

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "t1",
        "method": "sentencepiece",
        "vocab_size": 100,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }

    tkw.process_tokenizer_train_job(payload)

    assert fake.get("tokenizer:t1:status") == "failed"
    fake.assert_only_called({"set", "get", "publish"})
