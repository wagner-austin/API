"""Tests for detecting a run whose worker died without recording anything.

The defect these cover was observed, not imagined: a training container was
recreated mid-run, and five hours later the API still reported `running` with a
297-minute-stale heartbeat, because nothing read that heartbeat. A caller
polling for a terminal state waits forever on such a run.

The clock is injected through ``_test_hooks.time_wall_clock`` rather than
patched, so these exercise the real orchestrator against real Redis-shaped
state with only time under the test's control.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.job_types import JobStatusLiteral
from platform_core.trainer_keys import heartbeat_key
from platform_workers.testing import FakeQueue, FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core.infra.paths import checkpoints_dir
from model_trainer.core.services.training.liveness import (
    WORKER_HEARTBEAT_TIMEOUT_SECONDS,
    seconds_since_last_sign_of_life,
    worker_death_message,
    worker_has_died,
)
from model_trainer.worker.trainer_job_store import TrainerJobStore

from .test_training_orchestrator_resume import (
    _install_fake_rq,
    _make_orchestrator,
    _make_request,
    _touch_checkpoint,
)

RUN_ID = "run-zombie"

# A fixed wall-clock instant, so every expectation below is arithmetic rather
# than a race against the real clock.
NOW_TS = 1_787_000_000.0
NOW_DT = datetime.fromtimestamp(NOW_TS, tz=UTC).replace(tzinfo=None)


class TestSecondsSinceLastSignOfLife:
    def test_prefers_the_heartbeat_when_one_exists(self) -> None:
        age = seconds_since_last_sign_of_life(
            last_heartbeat_ts=NOW_TS - 120.0,
            status_updated_at=NOW_DT - timedelta(hours=5),
            now_ts=NOW_TS,
        )
        assert age == 120.0

    def test_falls_back_to_the_status_write_before_the_first_heartbeat(self) -> None:
        """Corpus fetch and model setup happen before the first heartbeat.

        Without this fallback a job killed during setup would look infinitely
        alive, because a heartbeat that was never written cannot go stale.
        """
        age = seconds_since_last_sign_of_life(
            last_heartbeat_ts=None,
            status_updated_at=NOW_DT - timedelta(minutes=45),
            now_ts=NOW_TS,
        )
        assert age == pytest.approx(2700.0)

    def test_a_stamp_in_the_future_reads_as_zero_age_not_negative(self) -> None:
        """The worker and the API are different machines and may disagree.

        Clock skew must not be reported as staleness, and must never produce a
        negative age that could underflow a comparison.
        """
        age = seconds_since_last_sign_of_life(
            last_heartbeat_ts=NOW_TS + 30.0,
            status_updated_at=NOW_DT,
            now_ts=NOW_TS,
        )
        assert age == 0.0


class TestWorkerHasDied:
    @pytest.mark.parametrize("status", ["queued", "completed", "failed"])
    def test_only_a_processing_run_is_judged_by_its_heartbeat(
        self, status: JobStatusLiteral
    ) -> None:
        """A queued run has no worker yet and a terminal run needs none."""
        assert not worker_has_died(
            status=status,
            last_heartbeat_ts=NOW_TS - 100_000.0,
            status_updated_at=NOW_DT - timedelta(days=1),
            now_ts=NOW_TS,
            timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        )

    def test_a_run_heartbeating_within_the_timeout_is_alive(self) -> None:
        """The end-of-run artifact upload is silent; the worst measured was 8 minutes."""
        assert not worker_has_died(
            status="processing",
            last_heartbeat_ts=NOW_TS - 8.0 * 60.0,
            status_updated_at=NOW_DT,
            now_ts=NOW_TS,
            timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        )

    def test_the_observed_zombie_is_detected(self) -> None:
        """297 minutes stale is the real incident this whole path exists for."""
        assert worker_has_died(
            status="processing",
            last_heartbeat_ts=NOW_TS - 297.0 * 60.0,
            status_updated_at=NOW_DT - timedelta(minutes=300),
            now_ts=NOW_TS,
            timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        )

    def test_the_boundary_is_exclusive_so_exactly_the_timeout_is_still_alive(self) -> None:
        """Asserted on both sides, because an off-by-one here kills healthy runs."""
        assert not worker_has_died(
            status="processing",
            last_heartbeat_ts=NOW_TS - WORKER_HEARTBEAT_TIMEOUT_SECONDS,
            status_updated_at=NOW_DT,
            now_ts=NOW_TS,
            timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        )
        assert worker_has_died(
            status="processing",
            last_heartbeat_ts=NOW_TS - WORKER_HEARTBEAT_TIMEOUT_SECONDS - 0.001,
            status_updated_at=NOW_DT,
            now_ts=NOW_TS,
            timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        )

    def test_a_run_killed_during_setup_is_detected_without_any_heartbeat(self) -> None:
        assert worker_has_died(
            status="processing",
            last_heartbeat_ts=None,
            status_updated_at=NOW_DT - timedelta(hours=2),
            now_ts=NOW_TS,
            timeout_seconds=WORKER_HEARTBEAT_TIMEOUT_SECONDS,
        )


def test_worker_death_message_names_the_run_the_silence_and_the_remedy() -> None:
    """Read by an operator deciding what to do, so all three must be present."""
    message = worker_death_message(run_id=RUN_ID, silent_for_seconds=297.0 * 60.0)
    assert RUN_ID in message
    assert "297.0 minutes" in message
    assert "resumed" in message


def _seed_running_run(
    redis: FakeRedis,
    *,
    heartbeat_ts: float | None,
    status_age: timedelta,
) -> None:
    """Put a run in Redis that claims to be training.

    Args:
        redis: Fake Redis backing the job store.
        heartbeat_ts: Heartbeat stamp to write, or None to write none at all.
        status_age: How long ago the status was last written.
    """
    written_at = NOW_DT - status_age
    TrainerJobStore(redis).save(
        {
            "job_id": RUN_ID,
            "user_id": 42,
            "status": "processing",
            "progress": 50,
            "message": "training",
            "created_at": written_at,
            "updated_at": written_at,
            "error": None,
            "artifact_file_id": None,
        },
    )
    if heartbeat_ts is not None:
        redis.set(heartbeat_key(RUN_ID), str(heartbeat_ts))


def _freeze_clock() -> None:
    """Pin the wall clock the orchestrator reads to :data:`NOW_TS`.

    The autouse hook-reset fixture in conftest restores the real clock after
    every test, so this needs no teardown of its own.
    """
    _test_hooks.time_wall_clock = lambda: NOW_TS


frozen_clock = pytest.fixture(_freeze_clock)


class TestGetStatusReportsWorkerDeath:
    def test_a_stale_running_run_reports_failed_with_its_own_error_code(
        self, tmp_path: Path, frozen_clock: None
    ) -> None:
        """The whole point: a caller polling for a terminal state now gets one.

        Asserted as the exact status and code rather than "not running",
        because the defect produced a run that was also not completed and not
        cancelled -- it was nothing at all.
        """
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        _seed_running_run(
            redis, heartbeat_ts=NOW_TS - 297.0 * 60.0, status_age=timedelta(minutes=300)
        )

        res = orch.get_status(RUN_ID)

        assert res["status"] == "failed"
        assert res["error"] == ModelTrainerErrorCode.RUN_WORKER_DIED.value
        assert res["last_heartbeat_ts"] == NOW_TS - 297.0 * 60.0
        assert "297.0 minutes" in str(res["message"])
        # Reading a status must not write one: the truth is derived per call,
        # so nothing here may mutate the run.
        redis.assert_only_called({"hset", "set", "hgetall", "get"})

    def test_a_live_run_still_reports_running_and_no_error(
        self, tmp_path: Path, frozen_clock: None
    ) -> None:
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        _seed_running_run(redis, heartbeat_ts=NOW_TS - 60.0, status_age=timedelta(minutes=90))

        res = orch.get_status(RUN_ID)

        assert res["status"] == "running"
        assert res["error"] is None
        assert res["message"] == "training"
        redis.assert_only_called({"hset", "set", "hgetall", "get"})

    def test_a_genuinely_failed_run_surfaces_its_own_error_not_worker_death(
        self, tmp_path: Path, frozen_clock: None
    ) -> None:
        """Training failure and machine failure are different diagnoses.

        A failed run's stored error must reach the caller unchanged, so this
        also pins that `error` is not hard-wired to the worker-death code.
        """
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        TrainerJobStore(redis).save(
            {
                "job_id": RUN_ID,
                "user_id": 42,
                "status": "failed",
                "progress": 30,
                "message": "training job failed",
                "created_at": NOW_DT - timedelta(hours=9),
                "updated_at": NOW_DT - timedelta(hours=9),
                "error": "TRAINING_NAN_LOSS",
                "artifact_file_id": None,
            },
        )

        res = orch.get_status(RUN_ID)

        assert res["status"] == "failed"
        assert res["error"] == "TRAINING_NAN_LOSS"
        redis.assert_only_called({"hset", "hgetall", "get"})

    def test_a_run_killed_before_its_first_heartbeat_is_reported_dead(
        self, tmp_path: Path, frozen_clock: None
    ) -> None:
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        _seed_running_run(redis, heartbeat_ts=None, status_age=timedelta(hours=2))

        res = orch.get_status(RUN_ID)

        assert res["status"] == "failed"
        assert res["error"] == ModelTrainerErrorCode.RUN_WORKER_DIED.value
        assert res["last_heartbeat_ts"] is None
        redis.assert_only_called({"hset", "hgetall", "get"})


class TestResumeAcceptsADeadWorkersRun:
    def test_a_stale_running_run_with_a_checkpoint_resumes(
        self, tmp_path: Path, frozen_clock: None
    ) -> None:
        """Killed-mid-training runs are exactly the ones worth resuming.

        Before this, resume required status == failed, so a run killed by a
        container recreation -- checkpoint on disk, minutes from done -- had to
        have its Redis status edited by hand before it could continue.
        """
        redis = FakeRedis()
        fake_queue = FakeQueue(job_id="job-resume-zombie")
        _install_fake_rq(fake_queue)
        orch, settings = _make_orchestrator(tmp_path, redis)
        _seed_running_run(
            redis, heartbeat_ts=NOW_TS - 297.0 * 60.0, status_age=timedelta(minutes=300)
        )
        _touch_checkpoint(settings, RUN_ID)

        out = orch.enqueue_resume(RUN_ID, _make_request())

        assert out["run_id"] == RUN_ID
        assert out["job_id"] == "job-resume-zombie"
        assert len(fake_queue.jobs) == 1
        redis.assert_only_called({"hset", "set", "hgetall", "get", "delete"})

    def test_a_live_running_run_is_still_refused(self, tmp_path: Path, frozen_clock: None) -> None:
        """Resuming a run that is actually training would duplicate it."""
        redis = FakeRedis()
        orch, settings = _make_orchestrator(tmp_path, redis)
        _seed_running_run(redis, heartbeat_ts=NOW_TS - 30.0, status_age=timedelta(minutes=90))
        _touch_checkpoint(settings, RUN_ID)

        with pytest.raises(AppError) as excinfo:
            _ = orch.enqueue_resume(RUN_ID, _make_request())

        exc: AppError[ModelTrainerErrorCode] = excinfo.value
        assert exc.code == ModelTrainerErrorCode.RUN_NOT_RESUMABLE
        redis.assert_only_called({"hset", "set", "hgetall", "get"})

    def test_a_dead_workers_run_without_a_checkpoint_is_still_refused(
        self, tmp_path: Path, frozen_clock: None
    ) -> None:
        """Worker death does not conjure a checkpoint; that gate stays."""
        redis = FakeRedis()
        orch, _ = _make_orchestrator(tmp_path, redis)
        _seed_running_run(
            redis, heartbeat_ts=NOW_TS - 297.0 * 60.0, status_age=timedelta(minutes=300)
        )

        with pytest.raises(AppError) as excinfo:
            _ = orch.enqueue_resume(RUN_ID, _make_request())

        exc: AppError[ModelTrainerErrorCode] = excinfo.value
        assert exc.code == ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND
        redis.assert_only_called({"hset", "set", "hgetall", "get"})


def test_checkpoints_dir_is_where_resume_looks(tmp_path: Path) -> None:
    """Guards the helper the resume tests rely on against a silent path move."""
    redis = FakeRedis()
    _, settings = _make_orchestrator(tmp_path, redis)
    _touch_checkpoint(settings, RUN_ID)
    assert (checkpoints_dir(settings) / f"{RUN_ID}.pt").exists()
    # Building an orchestrator must not touch Redis; only using one does.
    redis.assert_only_called(set())
