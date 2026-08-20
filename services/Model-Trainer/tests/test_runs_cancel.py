"""Tests for cancelling a run, including removing it from the queue.

Setting the cancellation flag is not enough for work that has not started. A
worker that dequeues a flagged job still loads the model before reaching its
first cancellation check, so cancelling queued work used to cost a full model
load while the run went on advertising `queued`. Cancelling a RUNNING job was
always correct and still is.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.errors import ModelTrainerErrorCode
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.trainer_keys import cancel_key, job_id_key
from platform_workers.testing import FakeQueue, FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from model_trainer.worker.trainer_job_store import TrainerJobStore

from .test_training_orchestrator_resume import _install_fake_rq, _make_orchestrator

RUN_ID = "run-x"


class TestCancelDequeuesPendingWork:
    def test_a_queued_run_has_its_job_removed_from_the_queue(self, tmp_path: Path) -> None:
        """The defect: the job stayed queued and later trained for real.

        Asserted as the queue being empty rather than as the response text,
        because the old behaviour returned a perfectly agreeable response
        while leaving the job in place.
        """
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-queued-1")
        _install_fake_rq(fake_queue)
        orch, _ = _make_orchestrator(tmp_path, redis)
        redis.set(job_id_key(RUN_ID), "job-queued-1")
        _ = fake_queue.enqueue("model_trainer.worker.train_job.process_train_job")

        result = orch.cancel(RUN_ID)

        assert result["status"] == "dequeued"
        assert fake_queue.jobs == []
        redis.assert_only_called({"set", "get", "hset", "hgetall"})

    def test_a_dequeued_run_is_left_in_a_terminal_state(self, tmp_path: Path) -> None:
        """Nothing will ever run the job now, so this call owns the outcome.

        Leaving the run `queued` would strand it exactly the way a dead worker
        strands a `processing` run -- the defect this service just fixed
        elsewhere -- so the status and the traceable code are both pinned.
        """
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-queued-2")
        _install_fake_rq(fake_queue)
        orch, _ = _make_orchestrator(tmp_path, redis)
        redis.set(job_id_key(RUN_ID), "job-queued-2")
        _ = fake_queue.enqueue("model_trainer.worker.train_job.process_train_job")

        _ = orch.cancel(RUN_ID)

        status = TrainerJobStore(redis).load(RUN_ID)
        assert status is not None and status["status"] == "failed"
        assert status["error"] == ModelTrainerErrorCode.TRAINING_CANCELLED.value
        assert status["message"] == "cancelled before training started"
        redis.assert_only_called({"set", "get", "hset", "hgetall"})

    def test_the_cancellation_flag_is_set_even_when_the_job_is_dequeued(
        self, tmp_path: Path
    ) -> None:
        """The job can be taken between the lookup and the removal attempt.

        The flag is what stops it in that race, so it is set unconditionally
        rather than only on the running path.
        """
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-queued-3")
        _install_fake_rq(fake_queue)
        orch, _ = _make_orchestrator(tmp_path, redis)
        redis.set(job_id_key(RUN_ID), "job-queued-3")
        _ = fake_queue.enqueue("model_trainer.worker.train_job.process_train_job")

        _ = orch.cancel(RUN_ID)

        assert redis.get(cancel_key(RUN_ID)) == "1"
        redis.assert_only_called({"set", "get", "hset", "hgetall"})


class TestCancelFallsBackToTheFlag:
    def test_a_running_run_is_flagged_because_its_job_is_no_longer_queued(
        self, tmp_path: Path
    ) -> None:
        """A worker already holds the job, so removal finds nothing to remove.

        This is the path that always worked, and it must keep working: the
        worker stops itself at its next cancellation check.
        """
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-running-1")
        _install_fake_rq(fake_queue)
        orch, _ = _make_orchestrator(tmp_path, redis)
        redis.set(job_id_key(RUN_ID), "job-running-1")

        result = orch.cancel(RUN_ID)

        assert result["status"] == "cancellation-requested"
        assert redis.get(cancel_key(RUN_ID)) == "1"
        redis.assert_only_called({"set", "get"})

    def test_a_run_with_no_recorded_job_is_flagged_without_touching_the_queue(
        self, tmp_path: Path
    ) -> None:
        """Runs enqueued before the job id was recorded still cancel.

        The mapping did not always exist, so a run may have no job id at all.
        That must fall back to the flag rather than fail the request.
        """
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-none")
        _install_fake_rq(fake_queue)
        orch, _ = _make_orchestrator(tmp_path, redis)
        _ = fake_queue.enqueue("model_trainer.worker.train_job.process_train_job")

        result = orch.cancel(RUN_ID)

        assert result["status"] == "cancellation-requested"
        assert redis.get(cancel_key(RUN_ID)) == "1"
        assert len(fake_queue.jobs) == 1
        redis.assert_only_called({"set", "get"})


def test_cancel_endpoint_reports_the_outcome(tmp_path: Path) -> None:
    """The HTTP surface distinguishes the two outcomes for the caller."""
    settings = load_settings()
    settings["app"]["artifacts_root"] = str(tmp_path / "artifacts")
    app = create_app(settings)
    container: ServiceContainer = app.state.container
    fake_redis = FakeRedis()
    container.redis = fake_redis
    # The container builds its orchestrator eagerly against the real redis
    # client; rebuild it against the fake for this test.
    container.training_orchestrator = TrainingOrchestrator(
        settings=settings,
        redis_client=fake_redis,
        enqueuer=container.rq_enqueuer,
        model_registry=container.model_registry,
    )
    fake_queue = FakeQueue(job_id="job-route-cancel")
    _install_fake_rq(fake_queue)
    fake_redis.set(job_id_key(RUN_ID), "job-route-cancel")
    _ = fake_queue.enqueue("model_trainer.worker.train_job.process_train_job")

    client = TestClient(app)
    response = client.post(f"/runs/{RUN_ID}/cancel")

    assert response.status_code == 200
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body["status"] == "dequeued"
    assert fake_queue.jobs == []
    assert fake_redis.get(cancel_key(RUN_ID)) == "1"
    fake_redis.assert_only_called({"set", "get", "hset", "hgetall"})
