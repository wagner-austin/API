from __future__ import annotations

from typing import Literal

from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.redis import _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue
from platform_workers.testing import FakeQueue, FakeRedis
from typing_extensions import TypedDict

from model_trainer.api.main import create_app
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import load_settings
from model_trainer.core.contracts.queue import TokenizerTrainPayload


def _validate_str(val: JSONValue, name: str) -> str:
    if not isinstance(val, str):
        raise AssertionError(f"{name} must be str")
    return val


def _validate_int(val: JSONValue, name: str) -> int:
    if not isinstance(val, int):
        raise AssertionError(f"{name} must be int")
    return val


def _validate_method(val: JSONValue) -> Literal["bpe", "sentencepiece", "char"]:
    if val == "bpe":
        return "bpe"
    if val == "sentencepiece":
        return "sentencepiece"
    if val == "char":
        return "char"
    raise AssertionError("method must be bpe, sentencepiece, or char")


def _decode_tokenizer_train_payload(raw: JSONValue) -> TokenizerTrainPayload:
    """Decode a raw JSON value to TokenizerTrainPayload for test assertions."""
    if not isinstance(raw, dict):
        raise AssertionError("payload must be dict")
    holdout_fraction = raw.get("holdout_fraction")
    if not isinstance(holdout_fraction, (int, float)):
        raise AssertionError("holdout_fraction must be number")
    return TokenizerTrainPayload(
        tokenizer_id=_validate_str(raw.get("tokenizer_id"), "tokenizer_id"),
        method=_validate_method(raw.get("method")),
        vocab_size=_validate_int(raw.get("vocab_size"), "vocab_size"),
        min_frequency=_validate_int(raw.get("min_frequency"), "min_frequency"),
        corpus_file_id=_validate_str(raw.get("corpus_file_id"), "corpus_file_id"),
        holdout_fraction=float(holdout_fraction),
        seed=_validate_int(raw.get("seed"), "seed"),
    )


def test_tokenizer_enqueue_passes_method() -> None:
    """Test that the tokenizer method is passed through to the enqueued job."""
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

    # Verify the job was enqueued with the correct method
    assert len(fake_queue.jobs) == 1
    job = fake_queue.jobs[0]
    # Decode the raw JSON value to typed payload
    payload = _decode_tokenizer_train_payload(job.args[0])
    assert payload["method"] == "bpe"
    fake_redis.assert_only_called({"set"})
