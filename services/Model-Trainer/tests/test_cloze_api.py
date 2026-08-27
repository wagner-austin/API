"""Tests for the cloze request validator, orchestrator and routes.

The stack is exercised end to end against fakes for redis and the queue, so the
real validator, orchestrator and route handlers all run. Cloze is a distinct
job type rather than a mode of /evaluate, so these paths share no branches with
perplexity evaluation and are covered on their own.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.errors import AppError, ErrorCode, ModelTrainerErrorCode
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str, load_json_str
from platform_core.trainer_keys import baseline_cloze_key, cloze_key
from platform_workers.redis import RedisBytesProto, _RedisBytesClient
from platform_workers.rq_harness import RQClientQueue, RQRetryLike
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient, FakeRetry

from model_trainer.api.routes import runs as runs_routes
from model_trainer.api.schemas.runs import BaselineClozeRequest
from model_trainer.api.validators.runs import _decode_baseline_cloze_request, _decode_cloze_request
from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.conversation_orchestrator import ConversationOrchestrator
from model_trainer.orchestrators.inference_orchestrator import InferenceOrchestrator
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from tests._cloze_fingerprint_support import CLOZE_FINGERPRINT


def _install_fakes() -> FakeQueue:
    fake_queue = FakeQueue()

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
        return fake_queue

    def _fake_rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        return FakeRetry(max=max_retries, interval=intervals)

    def _fake_redis_raw_for_rq(url: str) -> RedisBytesProto:
        return FakeRedisBytesClient()

    _test_hooks.rq_queue_factory = _fake_rq_queue
    _test_hooks.rq_retry_factory = _fake_rq_retry
    _test_hooks.rq_connection_factory = _fake_redis_raw_for_rq
    return fake_queue


class TestClozeRequestValidator:
    def test_defaults_max_seq_len(self) -> None:
        req = _decode_cloze_request({"items_file_id": "file-1"})
        assert req["items_file_id"] == "file-1"
        assert req["max_seq_len"] == 512

    def test_accepts_explicit_max_seq_len(self) -> None:
        req = _decode_cloze_request({"items_file_id": "file-1", "max_seq_len": 128})
        assert req["max_seq_len"] == 128

    def test_rejects_blank_items_file_id(self) -> None:
        with pytest.raises(AppError) as excinfo:
            _decode_cloze_request({"items_file_id": "   "})
        err: AppError[ErrorCode] = excinfo.value
        assert err.code == ErrorCode.INVALID_INPUT

    def test_rejects_missing_items_file_id(self) -> None:
        with pytest.raises(AppError):
            _decode_cloze_request({})

    def test_rejects_max_seq_len_below_minimum(self) -> None:
        with pytest.raises(AppError):
            _decode_cloze_request({"items_file_id": "file-1", "max_seq_len": 4})


class TestClozeOrchestrator:
    def _make_orchestrator(self) -> tuple[InferenceOrchestrator, FakeRedis, FakeQueue]:
        queue = _install_fakes()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://ignored", RQSettings(60, 300, 300, 1, [30]))
        orch = InferenceOrchestrator(settings=load_settings(), redis_client=redis, enqueuer=enq)
        return orch, redis, queue

    def test_enqueue_writes_queued_cache_and_enqueues_job(self) -> None:
        orch, redis, queue = self._make_orchestrator()
        out = orch.enqueue_cloze("run123", {"items_file_id": "file-1", "max_seq_len": 128})
        assert out["status"] == "queued"
        assert out["total"] is None
        assert uuid.UUID(out["request_id"]).version == 4
        assert len(queue.jobs) == 1
        assert queue.jobs[0].func == "model_trainer.worker.cloze_job.process_cloze_job"
        assert queue.jobs[0].description == f"cloze:run123:{out['request_id']}"

        cached = redis.get(cloze_key("run123", out["request_id"]))
        if not isinstance(cached, str):
            raise AssertionError(f"expected cached str, got {type(cached)}")
        obj = load_json_str(cached)
        if not isinstance(obj, dict):
            raise AssertionError(f"expected dict, got {type(obj)}")
        assert obj["status"] == "queued"
        assert obj["accuracy"] is None
        # "get" appears because this assertion reads the cache back, not
        # because enqueue reads it; enqueue only writes.
        redis.assert_only_called({"set", "get"})

    def test_get_returns_completed_counts(self) -> None:
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 8,
            "correct": 6,
            "accuracy": 0.75,
            "chance": 0.25,
            "fingerprint": CLOZE_FINGERPRINT,
        }
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))
        out = orch.get_cloze("run123", "req123")
        assert out["status"] == "completed"
        assert out["total"] == 8
        assert out["correct"] == 6
        assert out["accuracy"] == pytest.approx(0.75)
        assert out["chance"] == pytest.approx(0.25)

    def test_get_returns_per_item_outcomes_in_order(self) -> None:
        """Pairing two arms on the same item set needs the per-item records."""
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 2,
            "correct": 1,
            "accuracy": 0.5,
            "chance": 0.25,
            "outcomes": [
                {"item_id": "a", "correct": True, "scores": [1.0, 2.0]},
                {"item_id": "b", "correct": False, "scores": [2.0, 1.0]},
            ],
            "fingerprint": CLOZE_FINGERPRINT,
        }
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))
        out = orch.get_cloze("run123", "req123")
        outcomes = out["outcomes"]
        if outcomes is None:
            raise AssertionError("expected outcomes to be decoded")
        assert [o["item_id"] for o in outcomes] == ["a", "b"]
        assert [o["correct"] for o in outcomes] == [True, False]
        assert outcomes[0]["scores"] == [1.0, 2.0]

    def test_get_refuses_a_completed_record_that_cannot_say_what_it_ran_on(self) -> None:
        """The invariant this whole field exists for.

        A number without its configuration is not comparable: a later
        measurement that disagrees cannot be told apart from a working image
        scored on a different card. Serving it anyway is worse than refusing,
        because it looks comparable. The published 52.3030% floor is exactly
        such a record, which is why this is not softened for old entries.
        """
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 2627,
            "correct": 1374,
            "accuracy": 0.523030072325847,
            "chance": 0.25,
        }
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))

        with pytest.raises(AppError) as excinfo:
            orch.get_cloze("run123", "req123")

        err: AppError[ModelTrainerErrorCode] = excinfo.value
        assert err.code == ModelTrainerErrorCode.CLOZE_FINGERPRINT_MISSING
        assert "re-score it" in err.message

    def test_an_unfinished_record_needs_no_fingerprint(self) -> None:
        """A queued job computed no number, so nothing needs qualifying."""
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {"status": "running"}
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))

        out = orch.get_cloze("run123", "req123")

        assert out["status"] == "running"
        assert out["fingerprint"] is None

    def test_a_completed_record_returns_its_decoded_fingerprint(self) -> None:
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 8,
            "correct": 6,
            "accuracy": 0.75,
            "chance": 0.25,
            "fingerprint": CLOZE_FINGERPRINT,
        }
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))

        out = orch.get_cloze("run123", "req123")
        fingerprint = out["fingerprint"]

        if fingerprint is None:
            raise AssertionError("expected a fingerprint on a completed record")
        assert fingerprint["gpu_model"] == "NVIDIA GeForce RTX 3090 Ti"
        assert fingerprint["driver_version"] == "591.86"
        assert fingerprint["determinism"]["stack"] == "torch"
        assert ("cublas_workspace_config", ":4096:8") in fingerprint["determinism"]["settings"]

    def test_get_rejects_outcomes_that_are_not_a_list(self) -> None:
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {"status": "completed", "outcomes": {"item_id": "a"}}
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))
        with pytest.raises(AppError) as excinfo:
            orch.get_cloze("run123", "req123")
        err: AppError[ModelTrainerErrorCode] = excinfo.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        assert "outcomes" in err.message

    def test_get_rejects_a_malformed_outcome_record(self) -> None:
        """A bad record must fail loudly, not shorten the list on the way out."""
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 1,
            "correct": 1,
            "accuracy": 1.0,
            "chance": 0.25,
            "outcomes": [{"item_id": "a", "correct": True, "scores": [2.0, 1.0]}],
        }
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))
        with pytest.raises(JSONTypeError, match="but its scores imply False"):
            orch.get_cloze("run123", "req123")

    def test_get_tolerates_absent_metric_fields(self) -> None:
        orch, redis, _ = self._make_orchestrator()
        redis.set(cloze_key("run123", "req123"), dump_json_str({"status": "running"}))
        out = orch.get_cloze("run123", "req123")
        assert out["status"] == "running"
        assert out["total"] is None
        assert out["correct"] is None
        assert out["accuracy"] is None
        assert out["chance"] is None

    def test_get_missing_request_raises(self) -> None:
        orch, _, _ = self._make_orchestrator()
        with pytest.raises(AppError) as excinfo:
            orch.get_cloze("run123", "absent")
        err: AppError[ModelTrainerErrorCode] = excinfo.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND

    def test_get_corrupt_cache_raises(self) -> None:
        orch, redis, _ = self._make_orchestrator()
        redis.set(cloze_key("run123", "req123"), dump_json_str([1, 2, 3]))
        with pytest.raises(AppError) as excinfo:
            orch.get_cloze("run123", "req123")
        err: AppError[ModelTrainerErrorCode] = excinfo.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND


class TestClozeRoutes:
    def _make_client(self) -> tuple[TestClient, FakeRedis]:
        _install_fakes()
        s = load_settings()
        r = FakeRedis()
        ds = LocalTextDatasetBuilder()
        enq = RQEnqueuer("redis://ignored", RQSettings(60, 300, 300, 1, [30]))
        model_reg = ModelRegistry(registrations={}, dataset_builder=ds)
        container = ServiceContainer(
            settings=s,
            redis=r,
            rq_enqueuer=enq,
            training_orchestrator=TrainingOrchestrator(
                settings=s, redis_client=r, enqueuer=enq, model_registry=model_reg
            ),
            inference_orchestrator=InferenceOrchestrator(settings=s, redis_client=r, enqueuer=enq),
            conversation_orchestrator=ConversationOrchestrator(
                settings=s, redis_client=r, enqueuer=enq
            ),
            tokenizer_orchestrator=TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=enq),
            model_registry=model_reg,
            tokenizer_registry=TokenizerRegistry(backends={}),
            dataset_builder=ds,
        )
        app = FastAPI()
        app.include_router(runs_routes.build_router(container), prefix="/runs")
        install_exception_handlers_fastapi(app, logger_name="test", request_id_var=None)
        return TestClient(app), r

    def test_post_cloze_enqueues(self) -> None:
        client, _ = self._make_client()
        body: dict[str, JSONValue] = {"items_file_id": "file-1", "max_seq_len": 128}
        res = client.post(
            "/runs/run123/cloze",
            content=dump_json_str(body),
            headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
        )
        assert res.status_code == 200
        parsed = load_json_str(res.text)
        if not isinstance(parsed, dict):
            raise AssertionError(f"expected dict body, got {type(parsed)}")
        assert parsed["status"] == "queued"
        assert parsed["accuracy"] is None

    def test_get_cloze_returns_result(self) -> None:
        client, redis = self._make_client()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 4,
            "correct": 3,
            "accuracy": 0.75,
            "chance": 0.25,
            "fingerprint": CLOZE_FINGERPRINT,
        }
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))
        res = client.get("/runs/run123/cloze/req123", headers={"X-API-Key": "test-key"})
        assert res.status_code == 200
        parsed = load_json_str(res.text)
        if not isinstance(parsed, dict):
            raise AssertionError(f"expected dict body, got {type(parsed)}")
        assert parsed["correct"] == 3
        assert parsed["total"] == 4
        assert parsed["accuracy"] == pytest.approx(0.75)
        redis.assert_only_called({"set", "get"})

    def test_get_cloze_returns_outcomes_in_the_body(self) -> None:
        client, redis = self._make_client()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 1,
            "correct": 1,
            "accuracy": 1.0,
            "chance": 0.25,
            "outcomes": [{"item_id": "page::1", "correct": True, "scores": [1.0, 3.0]}],
            "fingerprint": CLOZE_FINGERPRINT,
        }
        redis.set(cloze_key("run123", "req123"), dump_json_str(cache))
        res = client.get("/runs/run123/cloze/req123", headers={"X-API-Key": "test-key"})
        assert res.status_code == 200
        parsed = load_json_str(res.text)
        if not isinstance(parsed, dict):
            raise AssertionError(f"expected dict body, got {type(parsed)}")
        outcomes = parsed["outcomes"]
        if not isinstance(outcomes, list):
            raise AssertionError(f"expected outcomes list, got {type(outcomes)}")
        first = outcomes[0]
        if not isinstance(first, dict):
            raise AssertionError(f"expected outcome dict, got {type(first)}")
        assert first["item_id"] == "page::1"
        assert first["correct"] is True
        assert first["scores"] == [1.0, 3.0]

    def test_get_cloze_missing_returns_404(self) -> None:
        client, _ = self._make_client()
        res = client.get("/runs/run123/cloze/absent", headers={"X-API-Key": "test-key"})
        assert res.status_code == 404


class TestBaselineClozeRequestValidator:
    def test_defaults_max_seq_len_and_device(self) -> None:
        """Device defaults to cpu because a baseline is cheap and often local."""
        req = _decode_baseline_cloze_request({"hub_model_id": "gpt2", "items_file_id": "file-1"})
        assert req["hub_model_id"] == "gpt2"
        assert req["max_seq_len"] == 512
        assert req["device"] == "cpu"

    def test_accepts_explicit_max_seq_len_and_device(self) -> None:
        req = _decode_baseline_cloze_request(
            {
                "hub_model_id": "gpt2-medium",
                "items_file_id": "file-1",
                "max_seq_len": 128,
                "device": "cuda",
            }
        )
        assert req["max_seq_len"] == 128
        assert req["device"] == "cuda"

    def test_rejects_blank_hub_model_id(self) -> None:
        """Blank halves of the identity are rejected at the edge.

        The model id becomes part of the key the result is stored under, so a
        blank one would produce a record nobody can identify.
        """
        with pytest.raises(AppError) as excinfo:
            _decode_baseline_cloze_request({"hub_model_id": "  ", "items_file_id": "file-1"})
        err: AppError[ErrorCode] = excinfo.value
        assert err.code == ErrorCode.INVALID_INPUT

    def test_rejects_blank_items_file_id(self) -> None:
        with pytest.raises(AppError) as excinfo:
            _decode_baseline_cloze_request({"hub_model_id": "gpt2", "items_file_id": " "})
        err: AppError[ErrorCode] = excinfo.value
        assert err.code == ErrorCode.INVALID_INPUT


class TestBaselineClozeOrchestrator:
    def _make_orchestrator(self) -> tuple[InferenceOrchestrator, FakeRedis, FakeQueue]:
        queue = _install_fakes()
        redis = FakeRedis()
        enq = RQEnqueuer("redis://ignored", RQSettings(60, 300, 300, 1, [30]))
        orch = InferenceOrchestrator(settings=load_settings(), redis_client=redis, enqueuer=enq)
        return orch, redis, queue

    def test_enqueue_records_under_the_baseline_key_not_a_run_key(self) -> None:
        """The identity is the model and the item set, so there is no request id."""
        orch, redis, queue = self._make_orchestrator()

        out = orch.enqueue_baseline_cloze(
            {
                "hub_model_id": "gpt2",
                "items_file_id": "file-1",
                "max_seq_len": 128,
                "device": "cpu",
            }
        )

        assert out["status"] == "queued"
        assert out["hub_model_id"] == "gpt2"
        assert out["items_file_id"] == "file-1"
        assert len(queue.jobs) == 1
        assert (
            queue.jobs[0].func
            == "model_trainer.worker.baseline_cloze_job.process_baseline_cloze_job"
        )
        assert queue.jobs[0].description == "baseline-cloze:gpt2:file-1"

        cached = redis.get(baseline_cloze_key("gpt2", "file-1"))
        if not isinstance(cached, str):
            raise AssertionError(f"expected cached str, got {type(cached)}")
        obj = load_json_str(cached)
        if not isinstance(obj, dict):
            raise AssertionError(f"expected dict, got {type(obj)}")
        assert obj["status"] == "queued"
        redis.assert_only_called({"set", "get"})

    def test_asking_twice_addresses_one_record(self) -> None:
        """Two requests for the same pair must not produce two measurements.

        Asserted as the second enqueue overwriting the same key, because a
        scheme keyed by request id would leave two records that could disagree
        about what the same model scored on the same items.
        """
        orch, redis, queue = self._make_orchestrator()
        req: BaselineClozeRequest = {
            "hub_model_id": "gpt2",
            "items_file_id": "file-1",
            "max_seq_len": 128,
            "device": "cpu",
        }

        first = orch.enqueue_baseline_cloze(req)
        second = orch.enqueue_baseline_cloze(req)

        assert first["hub_model_id"] == second["hub_model_id"]
        assert first["items_file_id"] == second["items_file_id"]
        assert len(queue.jobs) == 2
        # Both writes addressed the same key, so the second measurement
        # replaces the first rather than sitting beside it.
        written = {call.args[0] for call in redis.get_calls("set")}
        assert written == {baseline_cloze_key("gpt2", "file-1")}

    def test_get_returns_completed_counts(self) -> None:
        orch, redis, _ = self._make_orchestrator()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 2627,
            "correct": 1374,
            "accuracy": 0.523,
            "chance": 0.25,
            "fingerprint": CLOZE_FINGERPRINT,
        }
        redis.set(baseline_cloze_key("gpt2", "file-1"), dump_json_str(cache))

        out = orch.get_baseline_cloze("gpt2", "file-1")

        assert out["status"] == "completed"
        assert out["total"] == 2627
        assert out["accuracy"] == pytest.approx(0.523)
        assert out["hub_model_id"] == "gpt2"

    def test_get_on_an_unmeasured_pair_raises_rather_than_returning_empty(self) -> None:
        """ "Never measured" must not be readable as "measured zero".

        A caller that treated absence as a result would publish a baseline that
        does not exist, which is the defect this capability was built to end.
        """
        orch, _, _ = self._make_orchestrator()

        with pytest.raises(AppError) as excinfo:
            orch.get_baseline_cloze("gpt2", "never-scored")

        err: AppError[ModelTrainerErrorCode] = excinfo.value
        assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND
        assert "never-scored" in str(err)


class TestBaselineClozeRoutes:
    def _make_client(self) -> tuple[TestClient, FakeRedis]:
        _install_fakes()
        s = load_settings()
        r = FakeRedis()
        ds = LocalTextDatasetBuilder()
        enq = RQEnqueuer("redis://ignored", RQSettings(60, 300, 300, 1, [30]))
        model_reg = ModelRegistry(registrations={}, dataset_builder=ds)
        container = ServiceContainer(
            settings=s,
            redis=r,
            rq_enqueuer=enq,
            training_orchestrator=TrainingOrchestrator(
                settings=s, redis_client=r, enqueuer=enq, model_registry=model_reg
            ),
            inference_orchestrator=InferenceOrchestrator(settings=s, redis_client=r, enqueuer=enq),
            conversation_orchestrator=ConversationOrchestrator(
                settings=s, redis_client=r, enqueuer=enq
            ),
            tokenizer_orchestrator=TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=enq),
            model_registry=model_reg,
            tokenizer_registry=TokenizerRegistry(backends={}),
            dataset_builder=ds,
        )
        app = FastAPI()
        app.include_router(runs_routes.build_router(container), prefix="/runs")
        install_exception_handlers_fastapi(app, logger_name="test", request_id_var=None)
        return TestClient(app), r

    def test_post_baseline_cloze_enqueues(self) -> None:
        """`baselines` must route to the baseline handler, not be read as a run id.

        The literal segment only wins over the /{run_id}/ parameter because it
        is registered first, so this also pins that ordering.
        """
        client, _ = self._make_client()

        res = client.post(
            "/runs/baselines/cloze",
            headers={"X-API-Key": "test-key"},
            content=dump_json_str({"hub_model_id": "gpt2", "items_file_id": "file-1"}),
        )

        assert res.status_code == 200
        parsed = load_json_str(res.text)
        if not isinstance(parsed, dict):
            raise AssertionError(f"expected dict body, got {type(parsed)}")
        assert parsed["hub_model_id"] == "gpt2"
        assert parsed["status"] == "queued"

    def test_get_baseline_cloze_returns_the_recorded_score(self) -> None:
        client, redis = self._make_client()
        cache: dict[str, JSONValue] = {
            "status": "completed",
            "total": 2627,
            "correct": 1374,
            "accuracy": 0.523,
            "chance": 0.25,
            "outcomes": [{"item_id": "page::1", "correct": True, "scores": [1.0, 3.0]}],
            "fingerprint": CLOZE_FINGERPRINT,
        }
        redis.set(baseline_cloze_key("gpt2", "file-1"), dump_json_str(cache))

        res = client.get("/runs/baselines/gpt2/cloze/file-1", headers={"X-API-Key": "test-key"})

        assert res.status_code == 200
        parsed = load_json_str(res.text)
        if not isinstance(parsed, dict):
            raise AssertionError(f"expected dict body, got {type(parsed)}")
        assert parsed["total"] == 2627
        assert parsed["hub_model_id"] == "gpt2"

    def test_get_baseline_cloze_missing_returns_404(self) -> None:
        client, _ = self._make_client()
        res = client.get("/runs/baselines/gpt2/cloze/absent", headers={"X-API-Key": "test-key"})
        assert res.status_code == 404
