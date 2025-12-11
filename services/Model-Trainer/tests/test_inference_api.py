from __future__ import annotations

import uuid
from typing import Literal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_workers.redis import RedisBytesProto, _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue, RQRetryLike
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient, FakeRetry

from model_trainer.api.routes import runs as runs_routes
from model_trainer.api.validators.runs import _decode_generate_request, _decode_score_request
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.conversation_orchestrator import ConversationOrchestrator
from model_trainer.orchestrators.inference_orchestrator import (
    InferenceOrchestrator,
    _narrow_status,
    _parse_score_topk,
)
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


def _install_fakes(queue_holder: dict[str, FakeQueue]) -> None:
    fake_queue = FakeQueue()
    queue_holder["q"] = fake_queue

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        return FakeRetry(max=max_retries, interval=intervals)

    def _fake_redis_raw_for_rq(url: str) -> RedisBytesProto:
        return FakeRedisBytesClient()

    _test_hooks.rq_queue_factory = _fake_rq_queue
    _test_hooks.rq_retry_factory = _fake_rq_retry
    _test_hooks.rq_connection_factory = _fake_redis_raw_for_rq


class TestScoreRequestValidator:
    def test_valid_with_text(self) -> None:
        req = _decode_score_request({"text": "hello"})
        assert req["text"] == "hello"
        assert req["path"] is None
        assert req["detail_level"] == "summary"
        assert req["top_k"] is None
        assert req["seed"] is None

    def test_valid_with_path(self) -> None:
        req = _decode_score_request({"path": "/some/path"})
        assert req["text"] is None
        assert req["path"] == "/some/path"

    def test_valid_with_all_options(self) -> None:
        req = _decode_score_request(
            {
                "text": "test",
                "detail_level": "per_char",
                "top_k": 5,
                "seed": 42,
            }
        )
        assert req["text"] == "test"
        assert req["detail_level"] == "per_char"
        assert req["top_k"] == 5
        assert req["seed"] == 42

    def test_both_text_and_path_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_score_request({"text": "hello", "path": "/path"})
        assert "mutually exclusive" in exc_info.value.message

    def test_neither_text_nor_path_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_score_request({})
        assert "either text or path" in exc_info.value.message


class TestGenerateRequestValidator:
    def test_valid_with_prompt_text(self) -> None:
        req = _decode_generate_request({"prompt_text": "Hello"})
        assert req["prompt_text"] == "Hello"
        assert req["prompt_path"] is None
        assert req["max_new_tokens"] == 64
        assert req["temperature"] == 1.0
        assert req["top_k"] == 50
        assert req["top_p"] == 1.0
        assert req["stop_on_eos"] is True
        assert req["stop_sequences"] == []
        assert req["seed"] is None
        assert req["num_return_sequences"] == 1

    def test_valid_with_prompt_path(self) -> None:
        req = _decode_generate_request({"prompt_path": "/some/path"})
        assert req["prompt_text"] is None
        assert req["prompt_path"] == "/some/path"

    def test_valid_with_all_options(self) -> None:
        req = _decode_generate_request(
            {
                "prompt_text": "test",
                "max_new_tokens": 100,
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "stop_on_eos": False,
                "stop_sequences": ["END", "STOP"],
                "seed": 42,
                "num_return_sequences": 3,
            }
        )
        assert req["prompt_text"] == "test"
        assert req["max_new_tokens"] == 100
        assert req["temperature"] == 0.8
        assert req["top_k"] == 40
        assert req["top_p"] == 0.9
        assert req["stop_on_eos"] is False
        assert req["stop_sequences"] == ["END", "STOP"]
        assert req["seed"] == 42
        assert req["num_return_sequences"] == 3

    def test_both_prompt_text_and_path_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_generate_request({"prompt_text": "hello", "prompt_path": "/path"})
        assert "mutually exclusive" in exc_info.value.message

    def test_neither_prompt_text_nor_path_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_generate_request({})
        assert "either prompt_text or prompt_path" in exc_info.value.message

    def test_stop_on_eos_invalid_type_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_generate_request({"prompt_text": "hello", "stop_on_eos": "yes"})
        assert "stop_on_eos must be a boolean" in exc_info.value.message

    def test_stop_sequences_invalid_type_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_generate_request({"prompt_text": "hello", "stop_sequences": "not a list"})
        assert "stop_sequences must be a list" in exc_info.value.message

    def test_stop_sequences_invalid_item_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_generate_request({"prompt_text": "hello", "stop_sequences": [123]})
        assert "stop_sequences[0] must be a string" in exc_info.value.message


class TestNarrowStatus:
    def test_queued(self) -> None:
        assert _narrow_status("queued") == "queued"

    def test_running(self) -> None:
        assert _narrow_status("running") == "running"

    def test_completed(self) -> None:
        assert _narrow_status("completed") == "completed"

    def test_failed(self) -> None:
        assert _narrow_status("failed") == "failed"

    def test_unknown_defaults_to_failed(self) -> None:
        assert _narrow_status("unknown") == "failed"
        assert _narrow_status(None) == "failed"


class TestParseScoreTopk:
    def test_none_returns_none(self) -> None:
        assert _parse_score_topk(None) is None

    def test_non_list_returns_none(self) -> None:
        assert _parse_score_topk("not a list") is None

    def test_valid_topk(self) -> None:
        topk_v: list[JSONValue] = [
            [["a", 0.5], ["b", 0.3]],
            [["c", 0.8]],
        ]
        result = _parse_score_topk(topk_v)
        if result is None:
            raise AssertionError("Expected result to be a list, got None")
        assert len(result) == 2
        assert result[0] == [("a", 0.5), ("b", 0.3)]
        assert result[1] == [("c", 0.8)]

    def test_invalid_item_format_skipped(self) -> None:
        topk_v: list[JSONValue] = [
            [["a", 0.5], ["invalid"]],  # invalid has only 1 element
        ]
        result = _parse_score_topk(topk_v)
        if result is None:
            raise AssertionError("Expected result to be a list, got None")
        assert len(result) == 1
        assert result[0] == [("a", 0.5)]

    def test_non_list_position_skipped(self) -> None:
        topk_v: list[JSONValue] = [
            "not a list",  # non-list position item
            [["a", 0.5]],
        ]
        result = _parse_score_topk(topk_v)
        if result is None:
            raise AssertionError("Expected result to be a list, got None")
        assert len(result) == 1
        assert result[0] == [("a", 0.5)]

    def test_invalid_token_type_skipped(self) -> None:
        topk_v: list[JSONValue] = [
            [[123, 0.5], ["valid", 0.3]],  # first has int token instead of str
        ]
        result = _parse_score_topk(topk_v)
        if result is None:
            raise AssertionError("Expected result to be a list, got None")
        assert len(result) == 1
        assert result[0] == [("valid", 0.3)]

    def test_invalid_prob_type_skipped(self) -> None:
        topk_v: list[JSONValue] = [
            [["token", "not-a-float"], ["valid", 0.3]],  # first has str prob
        ]
        result = _parse_score_topk(topk_v)
        if result is None:
            raise AssertionError("Expected result to be a list, got None")
        assert len(result) == 1
        assert result[0] == [("valid", 0.3)]


class TestInferenceOrchestrator:
    def _make_orchestrator(
        self,
    ) -> tuple[InferenceOrchestrator, FakeRedis, dict[str, FakeQueue]]:
        holder: dict[str, FakeQueue] = {}
        _install_fakes(holder)
        s = load_settings()
        r = FakeRedis()
        enq = RQEnqueuer("redis://ignored", RQSettings(60, 300, 300, 1, [30]))
        orch = InferenceOrchestrator(settings=s, redis_client=r, enqueuer=enq)
        return orch, r, holder

    def test_enqueue_score(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        detail_level: Literal["summary", "per_char"] = "summary"
        response = orch.enqueue_score(
            "run123",
            {
                "text": "hello",
                "path": None,
                "detail_level": detail_level,
                "top_k": None,
                "seed": None,
            },
        )
        assert response["status"] == "queued"
        request_id = response["request_id"]
        if not isinstance(request_id, str):
            raise AssertionError(f"Expected request_id to be str, got {type(request_id)}")
        # Verify request_id is non-empty with explicit character check
        assert request_id and request_id[0].isalnum()
        assert response["loss"] is None
        assert response["perplexity"] is None
        r.assert_only_called({"set"})

    def test_get_score_not_found(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        with pytest.raises(AppError) as exc_info:
            orch.get_score("run123", "nonexistent")
        err: AppError[ModelTrainerErrorCode] = exc_info.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        r.assert_only_called({"get"})

    def test_get_score_corrupt_cache(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        from platform_core.trainer_keys import score_key

        r.set(score_key("run123", "req123"), '"not a dict"')
        with pytest.raises(AppError) as exc_info:
            orch.get_score("run123", "req123")
        assert "corrupt" in exc_info.value.message
        r.assert_only_called({"set", "get"})

    def test_get_score_completed(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        from platform_core.trainer_keys import score_key

        cache: dict[str, JSONValue] = {
            "status": "completed",
            "loss": 1.5,
            "perplexity": 4.5,
            "surprisal": [0.5, 0.7],
            "topk": [[["a", 0.5], ["b", 0.3]]],
            "tokens": ["h", "e", "l", "l", "o"],
        }
        r.set(score_key("run123", "req123"), dump_json_str(cache))
        result = orch.get_score("run123", "req123")
        assert result["status"] == "completed"
        assert result["loss"] == 1.5
        assert result["perplexity"] == 4.5
        assert result["surprisal"] == [0.5, 0.7]
        assert result["topk"] == [[("a", 0.5), ("b", 0.3)]]
        assert result["tokens"] == ["h", "e", "l", "l", "o"]
        r.assert_only_called({"set", "get"})

    def test_get_score_non_list_surprisal_and_tokens(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        from platform_core.trainer_keys import score_key

        cache: dict[str, JSONValue] = {
            "status": "completed",
            "loss": 1.5,
            "perplexity": 4.5,
            "surprisal": "not a list",
            "topk": None,
            "tokens": 12345,
        }
        r.set(score_key("run123", "req123"), dump_json_str(cache))
        result = orch.get_score("run123", "req123")
        assert result["status"] == "completed"
        assert result["surprisal"] is None
        assert result["tokens"] is None
        r.assert_only_called({"set", "get"})

    def test_enqueue_generate(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        response = orch.enqueue_generate(
            "run123",
            {
                "prompt_text": "hello",
                "prompt_path": None,
                "max_new_tokens": 10,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
                "stop_on_eos": True,
                "stop_sequences": [],
                "seed": None,
                "num_return_sequences": 1,
            },
        )
        assert response["status"] == "queued"
        request_id = response["request_id"]
        if not isinstance(request_id, str):
            raise AssertionError(f"Expected request_id to be str, got {type(request_id)}")
        # Verify request_id is non-empty with explicit character check
        assert request_id and request_id[0].isalnum()
        assert response["outputs"] is None
        assert response["steps"] is None
        assert response["eos_terminated"] is None
        r.assert_only_called({"set"})

    def test_get_generate_not_found(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        with pytest.raises(AppError) as exc_info:
            orch.get_generate("run123", "nonexistent")
        err: AppError[ModelTrainerErrorCode] = exc_info.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        r.assert_only_called({"get"})

    def test_get_generate_corrupt_cache(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        from platform_core.trainer_keys import generate_key

        r.set(generate_key("run123", "req123"), '"not a dict"')
        with pytest.raises(AppError) as exc_info:
            orch.get_generate("run123", "req123")
        assert "corrupt" in exc_info.value.message
        r.assert_only_called({"set", "get"})

    def test_get_generate_completed(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        from platform_core.trainer_keys import generate_key

        cache: dict[str, JSONValue] = {
            "status": "completed",
            "outputs": ["generated text"],
            "steps": 10,
            "eos_terminated": [True],
        }
        r.set(generate_key("run123", "req123"), dump_json_str(cache))
        result = orch.get_generate("run123", "req123")
        assert result["status"] == "completed"
        assert result["outputs"] == ["generated text"]
        assert result["steps"] == 10
        assert result["eos_terminated"] == [True]
        r.assert_only_called({"set", "get"})

    def test_get_generate_non_list_outputs_and_eos(self) -> None:
        orch, r, _holder = self._make_orchestrator()
        from platform_core.trainer_keys import generate_key

        cache: dict[str, JSONValue] = {
            "status": "completed",
            "outputs": "not a list",
            "steps": 10,
            "eos_terminated": 12345,
        }
        r.set(generate_key("run123", "req123"), dump_json_str(cache))
        result = orch.get_generate("run123", "req123")
        assert result["status"] == "completed"
        assert result["outputs"] is None
        assert result["eos_terminated"] is None
        r.assert_only_called({"set", "get"})


class TestInferenceAPIRoutes:
    def _make_client(self) -> tuple[TestClient, FakeRedis]:
        holder: dict[str, FakeQueue] = {}
        _install_fakes(holder)
        s = load_settings()
        r = FakeRedis()
        ds = LocalTextDatasetBuilder()
        enq = RQEnqueuer("redis://ignored", RQSettings(60, 300, 300, 1, [30]))
        model_reg = ModelRegistry(registrations={}, dataset_builder=ds)
        tokenizer_reg = TokenizerRegistry(backends={})
        training = TrainingOrchestrator(
            settings=s,
            redis_client=r,
            enqueuer=enq,
            model_registry=model_reg,
        )
        inference = InferenceOrchestrator(settings=s, redis_client=r, enqueuer=enq)
        conversation = ConversationOrchestrator(settings=s, redis_client=r, enqueuer=enq)
        tokenizer = TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=enq)
        container = ServiceContainer(
            settings=s,
            redis=r,
            rq_enqueuer=enq,
            training_orchestrator=training,
            inference_orchestrator=inference,
            conversation_orchestrator=conversation,
            tokenizer_orchestrator=tokenizer,
            model_registry=model_reg,
            tokenizer_registry=tokenizer_reg,
            dataset_builder=ds,
        )
        app = FastAPI()
        app.include_router(runs_routes.build_router(container), prefix="/runs")
        install_exception_handlers_fastapi(app, logger_name="test", request_id_var=None)
        return TestClient(app), r

    def test_enqueue_score_endpoint(self) -> None:
        client, r = self._make_client()
        payload: dict[str, JSONValue] = {"text": "hello"}
        res = client.post(
            "/runs/run123/score",
            content=dump_json_str(payload),
            headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
        )
        assert res.status_code == 200
        body_raw = load_json_str(res.text)
        if not isinstance(body_raw, dict):
            raise AssertionError("Response must be a dict")
        body: dict[str, JSONValue] = body_raw
        assert body["status"] == "queued"
        # Verify request_id is a valid UUID4 string
        request_id = body["request_id"]
        if not isinstance(request_id, str):
            raise AssertionError("request_id must be a string")
        parsed_uuid = uuid.UUID(request_id)
        assert parsed_uuid.version == 4
        r.assert_only_called({"set"})

    def test_get_score_not_found_route(self) -> None:
        client, r = self._make_client()
        res = client.get(
            "/runs/run123/score/nonexistent",
            headers={"X-API-Key": "test-key"},
        )
        assert res.status_code == 404
        r.assert_only_called({"get"})

    def test_enqueue_generate_endpoint(self) -> None:
        client, r = self._make_client()
        payload: dict[str, JSONValue] = {"prompt_text": "hello"}
        res = client.post(
            "/runs/run123/generate",
            content=dump_json_str(payload),
            headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
        )
        assert res.status_code == 200
        body_raw = load_json_str(res.text)
        if not isinstance(body_raw, dict):
            raise AssertionError("Response must be a dict")
        body: dict[str, JSONValue] = body_raw
        assert body["status"] == "queued"
        # Verify request_id is a valid UUID4 string
        request_id = body["request_id"]
        if not isinstance(request_id, str):
            raise AssertionError("request_id must be a string")
        parsed_uuid = uuid.UUID(request_id)
        assert parsed_uuid.version == 4
        r.assert_only_called({"set"})

    def test_get_generate_not_found_route(self) -> None:
        client, r = self._make_client()
        res = client.get(
            "/runs/run123/generate/nonexistent",
            headers={"X-API-Key": "test-key"},
        )
        assert res.status_code == 404
        r.assert_only_called({"get"})

    def _make_client_with_redis(self) -> tuple[TestClient, FakeRedis]:
        """Create a test client and return both client and redis for pre-populating."""
        holder: dict[str, FakeQueue] = {}
        _install_fakes(holder)
        s = load_settings()
        r = FakeRedis()
        enq = RQEnqueuer("redis://ignored", RQSettings(60, 300, 300, 1, [30]))
        model_reg = ModelRegistry(registrations={}, dataset_builder=LocalTextDatasetBuilder())
        tokenizer_reg = TokenizerRegistry(backends={})
        training = TrainingOrchestrator(
            settings=s,
            redis_client=r,
            enqueuer=enq,
            model_registry=model_reg,
        )
        inference = InferenceOrchestrator(settings=s, redis_client=r, enqueuer=enq)
        conversation = ConversationOrchestrator(settings=s, redis_client=r, enqueuer=enq)
        tokenizer = TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=enq)
        container = ServiceContainer(
            settings=s,
            redis=r,
            rq_enqueuer=enq,
            training_orchestrator=training,
            inference_orchestrator=inference,
            conversation_orchestrator=conversation,
            tokenizer_orchestrator=tokenizer,
            model_registry=model_reg,
            tokenizer_registry=tokenizer_reg,
            dataset_builder=LocalTextDatasetBuilder(),
        )
        app = FastAPI()
        app.include_router(runs_routes.build_router(container), prefix="/runs")
        install_exception_handlers_fastapi(app, logger_name="test", request_id_var=None)
        return TestClient(app), r

    def test_get_score_found_route(self) -> None:
        from platform_core.trainer_keys import score_key

        client, redis = self._make_client_with_redis()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "loss": 1.5,
            "perplexity": 4.5,
            "surprisal": [0.5, 0.7],
            "topk": [[["a", 0.5], ["b", 0.3]]],
            "tokens": ["h", "e", "l", "l", "o"],
        }
        redis.set(score_key("run123", "req123"), dump_json_str(cache))
        res = client.get(
            "/runs/run123/score/req123",
            headers={"X-API-Key": "test-key"},
        )
        assert res.status_code == 200
        body = load_json_str(res.text)
        if not isinstance(body, dict):
            raise AssertionError("Response must be a dict")
        assert body["status"] == "completed"
        assert body["loss"] == 1.5
        redis.assert_only_called({"set", "get"})

    def test_get_generate_found_route(self) -> None:
        from platform_core.trainer_keys import generate_key

        client, redis = self._make_client_with_redis()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "outputs": ["generated text"],
            "steps": 10,
            "eos_terminated": [True],
        }
        redis.set(generate_key("run123", "req123"), dump_json_str(cache))
        res = client.get(
            "/runs/run123/generate/req123",
            headers={"X-API-Key": "test-key"},
        )
        assert res.status_code == 200
        body = load_json_str(res.text)
        if not isinstance(body, dict):
            raise AssertionError("Response must be a dict")
        assert body["status"] == "completed"
        assert body["outputs"] == ["generated text"]
        redis.assert_only_called({"set", "get"})
