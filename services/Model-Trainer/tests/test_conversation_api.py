"""Tests for conversation orchestrator and chat API endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_core.trainer_keys import conversation_key, conversation_meta_key
from platform_workers.redis import RedisBytesProto, _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue, RQRetryLike
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient, FakeRetry

from model_trainer.api.routes import runs as runs_routes
from model_trainer.api.schemas.runs import ChatRequest
from model_trainer.api.validators.runs import _decode_chat_request
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.conversation_orchestrator import (
    ConversationOrchestrator,
    _decode_messages_from_json,
    _narrow_status,
)
from model_trainer.orchestrators.inference_orchestrator import InferenceOrchestrator
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


class TestChatRequestValidator:
    def test_valid_minimal(self) -> None:
        req = _decode_chat_request({"message": "Hello"})
        assert req["message"] == "Hello"
        assert req["session_id"] is None
        assert req["max_new_tokens"] == 128
        assert req["temperature"] == 0.8
        assert req["top_k"] == 50
        assert req["top_p"] == 0.95

    def test_valid_with_session_id(self) -> None:
        req = _decode_chat_request({"message": "Hello", "session_id": "sess-123"})
        assert req["message"] == "Hello"
        assert req["session_id"] == "sess-123"

    def test_valid_with_all_options(self) -> None:
        req = _decode_chat_request(
            {
                "message": "test",
                "session_id": "sess-abc",
                "max_new_tokens": 256,
                "temperature": 0.5,
                "top_k": 10,
                "top_p": 0.9,
            }
        )
        assert req["message"] == "test"
        assert req["session_id"] == "sess-abc"
        assert req["max_new_tokens"] == 256
        assert req["temperature"] == 0.5
        assert req["top_k"] == 10
        assert req["top_p"] == 0.9

    def test_missing_message_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_chat_request({})
        assert "message" in exc_info.value.message.lower()

    def test_invalid_message_type_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            _decode_chat_request({"message": 123})
        assert "message" in exc_info.value.message.lower()


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
        assert _narrow_status(123) == "failed"


class TestDecodeMessagesFromJson:
    def test_empty_list(self) -> None:
        assert _decode_messages_from_json([]) == []

    def test_valid_messages(self) -> None:
        msgs = _decode_messages_from_json(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hi there"

    def test_non_list_returns_empty(self) -> None:
        assert _decode_messages_from_json("not a list") == []
        assert _decode_messages_from_json({"role": "user"}) == []
        assert _decode_messages_from_json(None) == []

    def test_invalid_items_skipped(self) -> None:
        msgs = _decode_messages_from_json(
            [
                {"role": "user", "content": "Valid"},
                {"role": "invalid", "content": "Skipped"},
                {"role": "user"},  # missing content
                "not a dict",
                {"role": "assistant", "content": "Also valid"},
            ]
        )
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Valid"
        assert msgs[1]["content"] == "Also valid"


class TestConversationOrchestrator:
    def test_enqueue_chat_new_session(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        req: ChatRequest = {
            "message": "Hello",
            "session_id": None,
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
        }
        resp = orch.enqueue_chat("run-1", req)

        assert resp["status"] == "queued"
        assert resp["response"] is None
        assert isinstance(resp["session_id"], str) and len(resp["session_id"]) > 0
        assert isinstance(resp["request_id"], str) and len(resp["request_id"]) > 0

        # Check session was initialized
        conv_key = conversation_key("run-1", resp["session_id"])
        assert redis.get(conv_key) == "[]"
        redis.assert_only_called({"set", "get", "expire"})

    def test_enqueue_chat_existing_session(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        req: ChatRequest = {
            "message": "Hello",
            "session_id": "existing-session",
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
        }
        resp = orch.enqueue_chat("run-1", req)

        assert resp["session_id"] == "existing-session"
        assert resp["status"] == "queued"
        redis.assert_only_called({"set", "get"})

    def test_get_chat_result_not_found(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        with pytest.raises(AppError) as exc_info:
            orch.get_chat_result("run-1", "sess-1", "req-1")
        err: AppError[ModelTrainerErrorCode] = exc_info.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        redis.assert_only_called({"get"})

    def test_get_chat_result_completed(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        # Pre-populate result
        result_key = "runs:chat:run-1:sess-1:req-1"
        redis.set(result_key, dump_json_str({"status": "completed", "response": "Hi!"}))

        resp = orch.get_chat_result("run-1", "sess-1", "req-1")
        assert resp["status"] == "completed"
        assert resp["response"] == "Hi!"
        redis.assert_only_called({"set", "get"})

    def test_get_chat_result_corrupt_cache(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        # Pre-populate with non-dict
        result_key = "runs:chat:run-1:sess-1:req-1"
        redis.set(result_key, "[]")

        with pytest.raises(AppError) as exc_info:
            orch.get_chat_result("run-1", "sess-1", "req-1")
        err: AppError[ModelTrainerErrorCode] = exc_info.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        assert "corrupt" in err.message
        redis.assert_only_called({"set", "get"})

    def test_get_history_session_not_found(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        with pytest.raises(AppError) as exc_info:
            orch.get_history("run-1", "nonexistent")
        err: AppError[ModelTrainerErrorCode] = exc_info.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        redis.assert_only_called({"get"})

    def test_get_history_valid(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        # Pre-populate session
        conv_key = conversation_key("run-1", "sess-1")
        meta_key = conversation_meta_key("run-1", "sess-1")
        redis.set(
            conv_key,
            dump_json_str(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            ),
        )
        redis.set(
            meta_key,
            dump_json_str(
                {
                    "run_id": "run-1",
                    "created_at": "2025-01-01T00:00:00Z",
                    "session_ttl_sec": 3600,
                }
            ),
        )

        resp = orch.get_history("run-1", "sess-1")
        assert resp["session_id"] == "sess-1"
        assert resp["run_id"] == "run-1"
        assert len(resp["messages"]) == 2
        assert resp["created_at"] == "2025-01-01T00:00:00Z"
        redis.assert_only_called({"set", "get"})

    def test_get_history_corrupt_meta(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        # Pre-populate with non-dict meta
        meta_key = conversation_meta_key("run-1", "sess-1")
        redis.set(meta_key, "[]")

        with pytest.raises(AppError) as exc_info:
            orch.get_history("run-1", "sess-1")
        err: AppError[ModelTrainerErrorCode] = exc_info.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        assert "corrupt" in err.message
        redis.assert_only_called({"set", "get"})

    def test_get_history_empty_messages(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        # Pre-populate session with no messages
        meta_key = conversation_meta_key("run-1", "sess-1")
        redis.set(
            meta_key,
            dump_json_str(
                {
                    "run_id": "run-1",
                    "created_at": "2025-01-01T00:00:00Z",
                    "session_ttl_sec": 3600,
                }
            ),
        )

        resp = orch.get_history("run-1", "sess-1")
        assert resp["messages"] == []
        redis.assert_only_called({"set", "get"})

    def test_delete_session(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        settings = load_settings()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        orch = ConversationOrchestrator(settings=settings, redis_client=redis, enqueuer=enq)

        # Pre-populate session
        conv_key = conversation_key("run-1", "sess-1")
        meta_key = conversation_meta_key("run-1", "sess-1")
        redis.set(conv_key, "[]")
        redis.set(meta_key, "{}")

        orch.delete_session("run-1", "sess-1")

        assert redis.get(conv_key) is None
        assert redis.get(meta_key) is None
        redis.assert_only_called({"set", "get", "delete"})


def _make_container(redis: FakeRedis, enqueuer: RQEnqueuer) -> ServiceContainer:
    settings = load_settings()
    ds = LocalTextDatasetBuilder()
    model_reg = ModelRegistry(registrations={}, dataset_builder=ds)
    tok_reg = TokenizerRegistry(backends={})
    training = TrainingOrchestrator(
        settings=settings, redis_client=redis, enqueuer=enqueuer, model_registry=model_reg
    )
    inference = InferenceOrchestrator(settings=settings, redis_client=redis, enqueuer=enqueuer)
    conversation = ConversationOrchestrator(
        settings=settings, redis_client=redis, enqueuer=enqueuer
    )
    tokenizer = TokenizerOrchestrator(settings=settings, redis_client=redis, enqueuer=enqueuer)
    return ServiceContainer(
        settings=settings,
        redis=redis,
        rq_enqueuer=enqueuer,
        training_orchestrator=training,
        inference_orchestrator=inference,
        conversation_orchestrator=conversation,
        tokenizer_orchestrator=tokenizer,
        model_registry=model_reg,
        tokenizer_registry=tok_reg,
        dataset_builder=ds,
    )


def _build_app(container: ServiceContainer) -> TestClient:
    app = FastAPI()
    app.include_router(runs_routes.build_router(container), prefix="/runs")
    install_exception_handlers_fastapi(app, logger_name="test", request_id_var=None)
    return TestClient(app)


class TestChatApiRoutes:
    def test_enqueue_chat_creates_session(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        container = _make_container(redis, enq)
        client = _build_app(container)

        body: dict[str, str] = {"message": "Hello"}
        resp = client.post(
            "/runs/run-1/chat",
            json=body,
            headers={"x-api-key": container.settings["security"]["api_key"]},
        )
        assert resp.status_code == 200
        data_raw = load_json_str(resp.text)
        assert isinstance(data_raw, dict) and data_raw.get("status") == "queued"
        data: dict[str, JSONValue] = data_raw
        assert isinstance(data["session_id"], str) and len(data["session_id"]) > 0
        assert isinstance(data["request_id"], str) and len(data["request_id"]) > 0
        redis.assert_only_called({"set", "expire"})

    def test_get_chat_result_not_found(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        container = _make_container(redis, enq)
        client = _build_app(container)

        resp = client.get(
            "/runs/run-1/chat/sess-1/req-1",
            headers={"x-api-key": container.settings["security"]["api_key"]},
        )
        assert resp.status_code == 404
        redis.assert_only_called({"get"})

    def test_get_chat_result_completed(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        container = _make_container(redis, enq)
        client = _build_app(container)

        # Pre-populate result
        result_key = "runs:chat:run-1:sess-1:req-1"
        redis.set(result_key, dump_json_str({"status": "completed", "response": "Hi!"}))

        resp = client.get(
            "/runs/run-1/chat/sess-1/req-1",
            headers={"x-api-key": container.settings["security"]["api_key"]},
        )
        assert resp.status_code == 200
        data_raw = load_json_str(resp.text)
        assert isinstance(data_raw, dict) and data_raw.get("status") == "completed"
        data: dict[str, JSONValue] = data_raw
        assert data["response"] == "Hi!"
        redis.assert_only_called({"set", "get"})

    def test_get_chat_history_not_found(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        container = _make_container(redis, enq)
        client = _build_app(container)

        resp = client.get(
            "/runs/run-1/chat/sess-1",
            headers={"x-api-key": container.settings["security"]["api_key"]},
        )
        assert resp.status_code == 404
        redis.assert_only_called({"get"})

    def test_get_chat_history_valid(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        container = _make_container(redis, enq)
        client = _build_app(container)

        # Pre-populate session
        conv_key = conversation_key("run-1", "sess-1")
        meta_key = conversation_meta_key("run-1", "sess-1")
        redis.set(conv_key, dump_json_str([{"role": "user", "content": "Hi"}]))
        redis.set(
            meta_key,
            dump_json_str(
                {
                    "run_id": "run-1",
                    "created_at": "2025-01-01T00:00:00Z",
                    "session_ttl_sec": 3600,
                }
            ),
        )

        resp = client.get(
            "/runs/run-1/chat/sess-1",
            headers={"x-api-key": container.settings["security"]["api_key"]},
        )
        assert resp.status_code == 200
        data_raw = load_json_str(resp.text)
        assert isinstance(data_raw, dict) and data_raw.get("session_id") == "sess-1"
        data: dict[str, JSONValue] = data_raw
        msgs = data["messages"]
        assert isinstance(msgs, list) and len(msgs) == 1
        redis.assert_only_called({"set", "get"})

    def test_delete_chat_session(self) -> None:
        queue_holder: dict[str, FakeQueue] = {}
        _install_fakes(queue_holder)

        redis = FakeRedis()
        enq = RQEnqueuer("redis://localhost", RQSettings(1, 1, 1, 0, []))
        container = _make_container(redis, enq)
        client = _build_app(container)

        # Pre-populate session
        conv_key = conversation_key("run-1", "sess-1")
        meta_key = conversation_meta_key("run-1", "sess-1")
        redis.set(conv_key, "[]")
        redis.set(meta_key, "{}")

        resp = client.delete(
            "/runs/run-1/chat/sess-1",
            headers={"x-api-key": container.settings["security"]["api_key"]},
        )
        assert resp.status_code == 200
        del_raw = load_json_str(resp.text)
        assert isinstance(del_raw, dict) and del_raw.get("status") == "cancellation-requested"
        assert redis.get(conv_key) is None
        redis.assert_only_called({"set", "get", "delete"})
