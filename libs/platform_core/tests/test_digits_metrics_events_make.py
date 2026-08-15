"""Tests for digits metrics events: MakeConfigEvent."""

from __future__ import annotations

from platform_core.digits_metrics_events import (
    make_artifact_event,
    make_batch_metrics_event,
    make_best_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_epoch_metrics_event,
    make_prune_event,
    make_upload_event,
)


class TestMakeConfigEvent:
    def test_required_fields_only(self) -> None:
        ev = make_config_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            total_epochs=10,
            queue="digits-training",
        )
        assert ev["type"] == "digits.metrics.config.v1"
        assert ev["job_id"] == "j1"
        assert ev["user_id"] == 123
        assert ev["model_id"] == "m1"
        assert ev["total_epochs"] == 10
        assert ev["queue"] == "digits-training"
        assert "cpu_cores" not in ev

    def test_all_optional_fields(self) -> None:
        ev = make_config_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            total_epochs=10,
            queue="q",
            cpu_cores=8,
            optimal_threads=4,
            memory_mb=16384,
            optimal_workers=2,
            max_batch_size=64,
            device="cuda",
            batch_size=32,
            learning_rate=0.001,
            augment=True,
            aug_rotate=15.0,
            aug_translate=0.1,
            noise_prob=0.05,
            dots_prob=0.02,
        )
        assert ev["cpu_cores"] == 8
        assert ev["optimal_threads"] == 4
        assert ev["memory_mb"] == 16384
        assert ev["optimal_workers"] == 2
        assert ev["max_batch_size"] == 64
        assert ev["device"] == "cuda"
        assert ev["batch_size"] == 32
        assert ev["learning_rate"] == 0.001
        assert ev["augment"] is True
        assert ev["aug_rotate"] == 15.0
        assert ev["aug_translate"] == 0.1
        assert ev["noise_prob"] == 0.05
        assert ev["dots_prob"] == 0.02


class TestMakeBatchMetricsEvent:
    def test_creates_batch_event(self) -> None:
        ev = make_batch_metrics_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            epoch=1,
            total_epochs=10,
            batch=5,
            total_batches=100,
            batch_loss=0.5,
            batch_acc=0.95,
            avg_loss=0.4,
            samples_per_sec=128.0,
            main_rss_mb=1024,
            workers_rss_mb=512,
            worker_count=4,
            cgroup_usage_mb=2048,
            cgroup_limit_mb=4096,
            cgroup_pct=50.0,
            anon_mb=1500,
            file_mb=200,
        )
        assert ev["type"] == "digits.metrics.batch.v1"
        assert ev["batch"] == 5
        assert ev["batch_loss"] == 0.5
        assert ev["cgroup_pct"] == 50.0


class TestMakeEpochMetricsEvent:
    def test_creates_epoch_event(self) -> None:
        ev = make_epoch_metrics_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            epoch=1,
            total_epochs=10,
            train_loss=0.3,
            val_acc=0.97,
            time_s=120.5,
        )
        assert ev["type"] == "digits.metrics.epoch.v1"
        assert ev["epoch"] == 1
        assert ev["train_loss"] == 0.3
        assert ev["val_acc"] == 0.97
        assert ev["time_s"] == 120.5


class TestMakeBestMetricsEvent:
    def test_creates_best_event(self) -> None:
        ev = make_best_metrics_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            epoch=5,
            val_acc=0.98,
        )
        assert ev["type"] == "digits.metrics.best.v1"
        assert ev["epoch"] == 5
        assert ev["val_acc"] == 0.98


class TestMakeArtifactEvent:
    def test_creates_artifact_event(self) -> None:
        ev = make_artifact_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            path="/data/artifacts/model.pt",
        )
        assert ev["type"] == "digits.metrics.artifact.v1"
        assert ev["path"] == "/data/artifacts/model.pt"


class TestMakeUploadEvent:
    def test_creates_upload_event(self) -> None:
        ev = make_upload_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            status=200,
            model_bytes=1024000,
            manifest_bytes=512,
            file_id="f123",
            file_sha256="abc123",
        )
        assert ev["type"] == "digits.metrics.upload.v1"
        assert ev["status"] == 200
        assert ev["model_bytes"] == 1024000
        assert ev["file_id"] == "f123"


class TestMakePruneEvent:
    def test_creates_prune_event(self) -> None:
        ev = make_prune_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            deleted_count=3,
        )
        assert ev["type"] == "digits.metrics.prune.v1"
        assert ev["deleted_count"] == 3


class TestMakeCompletedMetricsEvent:
    def test_creates_completed_event(self) -> None:
        ev = make_completed_metrics_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            val_acc=0.985,
        )
        assert ev["type"] == "digits.metrics.completed.v1"
        assert ev["val_acc"] == 0.985
