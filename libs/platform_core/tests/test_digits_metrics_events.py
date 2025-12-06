"""Tests for digits_metrics_events module."""

from __future__ import annotations

import pytest

from platform_core.digits_metrics_events import (
    DigitsBatchMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    DigitsMetricsEventV1,
    decode_digits_metrics_event,
    encode_digits_metrics_event,
    is_artifact,
    is_batch,
    is_best,
    is_completed,
    is_config,
    is_epoch,
    is_prune,
    is_upload,
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


class TestEncodeDecodeRoundtrip:
    def test_config_roundtrip(self) -> None:
        ev = make_config_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            total_epochs=10,
            queue="q",
            batch_size=32,
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert decoded["type"] == "digits.metrics.config.v1"
        assert decoded["job_id"] == "j1"
        assert is_config(decoded)
        config_ev: DigitsConfigV1 = decoded
        assert config_ev["batch_size"] == 32

    def test_config_roundtrip_all_optional_context(self) -> None:
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
            device="cuda:0",
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_config(decoded)
        config_ev: DigitsConfigV1 = decoded
        assert config_ev["cpu_cores"] == 8
        assert config_ev["optimal_threads"] == 4
        assert config_ev["memory_mb"] == 16384
        assert config_ev["optimal_workers"] == 2
        assert config_ev["max_batch_size"] == 64
        assert config_ev["device"] == "cuda:0"

    def test_config_roundtrip_all_optional_augment(self) -> None:
        ev = make_config_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            total_epochs=10,
            queue="q",
            batch_size=32,
            learning_rate=0.001,
            augment=True,
            aug_rotate=15.0,
            aug_translate=0.1,
            noise_prob=0.05,
            dots_prob=0.02,
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_config(decoded)
        config_ev: DigitsConfigV1 = decoded
        assert config_ev["batch_size"] == 32
        assert config_ev["learning_rate"] == 0.001
        assert config_ev["augment"] is True
        assert config_ev["aug_rotate"] == 15.0
        assert config_ev["aug_translate"] == 0.1
        assert config_ev["noise_prob"] == 0.05
        assert config_ev["dots_prob"] == 0.02

    def test_batch_roundtrip(self) -> None:
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
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_batch(decoded)

    def test_epoch_roundtrip(self) -> None:
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
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_epoch(decoded)

    def test_best_roundtrip(self) -> None:
        ev = make_best_metrics_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            epoch=5,
            val_acc=0.98,
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_best(decoded)

    def test_artifact_roundtrip(self) -> None:
        ev = make_artifact_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            path="/path",
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_artifact(decoded)

    def test_upload_roundtrip(self) -> None:
        ev = make_upload_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            status=200,
            model_bytes=1024,
            manifest_bytes=128,
            file_id="f1",
            file_sha256="sha",
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_upload(decoded)

    def test_prune_roundtrip(self) -> None:
        ev = make_prune_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            deleted_count=5,
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_prune(decoded)

    def test_completed_roundtrip(self) -> None:
        ev = make_completed_metrics_event(
            job_id="j1",
            user_id=123,
            model_id="m1",
            val_acc=0.99,
        )
        encoded = encode_digits_metrics_event(ev)
        decoded = decode_digits_metrics_event(encoded)
        assert is_completed(decoded)


class TestDecodeErrors:
    def test_non_object_payload_raises(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            decode_digits_metrics_event("[]")

    def test_non_string_type_raises(self) -> None:
        with pytest.raises(ValueError, match="type must be a string"):
            decode_digits_metrics_event('{"type": 123}')

    def test_missing_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id and user_id are required"):
            decode_digits_metrics_event('{"type": "digits.metrics.config.v1", "user_id": 1}')

    def test_missing_user_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id and user_id are required"):
            decode_digits_metrics_event('{"type": "digits.metrics.config.v1", "job_id": "j1"}')

    def test_unknown_type_raises(self) -> None:
        payload = '{"type": "digits.metrics.unknown.v1", "job_id": "j1", "user_id": 1}'
        with pytest.raises(ValueError, match="unknown digits metrics event type"):
            decode_digits_metrics_event(payload)

    def test_config_missing_required_raises(self) -> None:
        with pytest.raises(ValueError, match="config event requires"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.config.v1", "job_id": "j1", "user_id": 1}'
            )

    def test_batch_missing_required_raises(self) -> None:
        payload = (
            '{"type": "digits.metrics.batch.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
        )
        with pytest.raises(ValueError, match="batch metrics event missing"):
            decode_digits_metrics_event(payload)

    def test_epoch_missing_required_raises(self) -> None:
        payload = (
            '{"type": "digits.metrics.epoch.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
        )
        with pytest.raises(ValueError, match="epoch metrics event missing"):
            decode_digits_metrics_event(payload)

    def test_best_missing_required_raises(self) -> None:
        with pytest.raises(ValueError, match="best metrics event missing"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.best.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
            )

    def test_artifact_missing_required_raises(self) -> None:
        with pytest.raises(ValueError, match="artifact event missing"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.artifact.v1", "job_id": "j1", "user_id": 1}'
            )

    def test_upload_missing_required_raises(self) -> None:
        payload = (
            '{"type": "digits.metrics.upload.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
        )
        with pytest.raises(ValueError, match="upload event missing"):
            decode_digits_metrics_event(payload)

    def test_prune_missing_required_raises(self) -> None:
        with pytest.raises(ValueError, match="prune event missing"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.prune.v1", "job_id": "j1", "user_id": 1}'
            )

    def test_completed_missing_required_raises(self) -> None:
        with pytest.raises(ValueError, match="completed metrics event missing"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.completed.v1", "job_id": "j1", "user_id": 1}'
            )


class TestTypeGuards:
    def test_is_config_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_config_event(
            job_id="j", user_id=1, model_id="m", total_epochs=1, queue="q"
        )
        assert is_config(ev)
        assert not is_batch(ev)
        assert not is_epoch(ev)
        assert not is_best(ev)
        assert not is_artifact(ev)
        assert not is_upload(ev)
        assert not is_prune(ev)
        assert not is_completed(ev)

    def test_is_batch_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_batch_metrics_event(
            job_id="j",
            user_id=1,
            model_id="m",
            epoch=1,
            total_epochs=1,
            batch=1,
            total_batches=1,
            batch_loss=0.1,
            batch_acc=0.9,
            avg_loss=0.1,
            samples_per_sec=1.0,
            main_rss_mb=1,
            workers_rss_mb=1,
            worker_count=1,
            cgroup_usage_mb=1,
            cgroup_limit_mb=1,
            cgroup_pct=1.0,
            anon_mb=1,
            file_mb=1,
        )
        assert is_batch(ev)
        assert not is_config(ev)

    def test_is_epoch_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_epoch_metrics_event(
            job_id="j",
            user_id=1,
            model_id="m",
            epoch=1,
            total_epochs=1,
            train_loss=0.1,
            val_acc=0.9,
            time_s=1.0,
        )
        assert is_epoch(ev)
        assert not is_config(ev)

    def test_is_best_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_best_metrics_event(
            job_id="j", user_id=1, model_id="m", epoch=1, val_acc=0.9
        )
        assert is_best(ev)
        assert not is_config(ev)

    def test_is_artifact_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_artifact_event(
            job_id="j", user_id=1, model_id="m", path="/p"
        )
        assert is_artifact(ev)
        assert not is_config(ev)

    def test_is_upload_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_upload_event(
            job_id="j",
            user_id=1,
            model_id="m",
            status=200,
            model_bytes=1,
            manifest_bytes=1,
            file_id="f",
            file_sha256="s",
        )
        assert is_upload(ev)
        assert not is_config(ev)

    def test_is_prune_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_prune_event(
            job_id="j", user_id=1, model_id="m", deleted_count=1
        )
        assert is_prune(ev)
        assert not is_config(ev)

    def test_is_completed_true(self) -> None:
        ev: DigitsMetricsEventV1 = make_completed_metrics_event(
            job_id="j", user_id=1, model_id="m", val_acc=0.9
        )
        assert is_completed(ev)
        assert not is_config(ev)


class TestIntCoercionToFloat:
    def test_batch_metrics_int_to_float(self) -> None:
        payload = """{
            "type": "digits.metrics.batch.v1",
            "job_id": "j1", "user_id": 1, "model_id": "m1",
            "epoch": 1, "total_epochs": 10, "batch": 1, "total_batches": 100,
            "batch_loss": 1, "batch_acc": 1, "avg_loss": 1, "samples_per_sec": 1,
            "main_rss_mb": 1, "workers_rss_mb": 1, "worker_count": 1,
            "cgroup_usage_mb": 1, "cgroup_limit_mb": 1, "cgroup_pct": 1,
            "anon_mb": 1, "file_mb": 1
        }"""
        decoded = decode_digits_metrics_event(payload)
        assert is_batch(decoded)
        batch_ev: DigitsBatchMetricsV1 = decoded
        assert type(batch_ev["batch_loss"]) is float
        assert type(batch_ev["cgroup_pct"]) is float

    def test_epoch_metrics_int_to_float(self) -> None:
        payload = """{
            "type": "digits.metrics.epoch.v1",
            "job_id": "j1", "user_id": 1, "model_id": "m1",
            "epoch": 1, "total_epochs": 10,
            "train_loss": 1, "val_acc": 1, "time_s": 1
        }"""
        decoded = decode_digits_metrics_event(payload)
        assert is_epoch(decoded)
        epoch_ev: DigitsEpochMetricsV1 = decoded
        assert type(epoch_ev["train_loss"]) is float

    def test_config_optional_int_to_float(self) -> None:
        payload = """{
            "type": "digits.metrics.config.v1",
            "job_id": "j1", "user_id": 1, "model_id": "m1",
            "total_epochs": 10, "queue": "q",
            "learning_rate": 1, "aug_rotate": 15, "aug_translate": 1,
            "noise_prob": 1, "dots_prob": 1
        }"""
        decoded = decode_digits_metrics_event(payload)
        assert is_config(decoded)
        config_ev: DigitsConfigV1 = decoded
        assert type(config_ev["learning_rate"]) is float
        assert type(config_ev["aug_rotate"]) is float


class TestTryDecodeDigitsMetricsEvent:
    def test_returns_none_for_non_dict(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_metrics_event

        assert try_decode_digits_metrics_event("[]") is None

    def test_returns_none_for_non_string_type(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_metrics_event

        assert try_decode_digits_metrics_event('{"type": 123}') is None

    def test_returns_none_for_non_metrics_type(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_metrics_event

        assert try_decode_digits_metrics_event('{"type": "digits.job.started.v1"}') is None

    def test_returns_none_for_missing_job_id(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_metrics_event

        payload = '{"type": "digits.metrics.config.v1", "user_id": 1}'
        assert try_decode_digits_metrics_event(payload) is None

    def test_returns_none_for_unknown_metrics_type(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_metrics_event

        payload = '{"type": "digits.metrics.unknown.v1", "job_id": "j1", "user_id": 1}'
        assert try_decode_digits_metrics_event(payload) is None

    def test_decodes_valid_config_event(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_metrics_event

        ev = make_config_event(job_id="j1", user_id=1, model_id="m", total_epochs=1, queue="q")
        payload = encode_digits_metrics_event(ev)
        decoded = try_decode_digits_metrics_event(payload)
        if decoded is None:
            pytest.fail("expected decoded event")
        assert is_config(decoded)
        assert decoded["type"] == "digits.metrics.config.v1"


class TestTryDecodeDigitsEvent:
    def test_returns_none_for_non_dict(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        assert try_decode_digits_event("[]") is None

    def test_returns_none_for_non_string_type(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        assert try_decode_digits_event('{"type": 123}') is None

    def test_returns_none_for_unknown_type(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        assert try_decode_digits_event('{"type": "unknown.event.v1"}') is None

    def test_decodes_job_started_event(self) -> None:
        from platform_core.digits_metrics_events import (
            is_digits_job_started,
            try_decode_digits_event,
        )

        payload = """{
            "type": "digits.job.started.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "queue": "digits-training"
        }"""
        ev = try_decode_digits_event(payload)
        if ev is None:
            pytest.fail("expected decoded event")
        assert is_digits_job_started(ev)
        assert ev["type"] == "digits.job.started.v1"

    def test_decodes_job_completed_event(self) -> None:
        from platform_core.digits_metrics_events import (
            is_digits_job_completed,
            try_decode_digits_event,
        )

        payload = """{
            "type": "digits.job.completed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "result_id": "r1",
            "result_bytes": 1024
        }"""
        ev = try_decode_digits_event(payload)
        if ev is None:
            pytest.fail("expected decoded event")
        assert is_digits_job_completed(ev)
        assert ev["type"] == "digits.job.completed.v1"

    def test_decodes_job_failed_event_user(self) -> None:
        from platform_core.digits_metrics_events import (
            is_digits_job_failed,
            try_decode_digits_event,
        )

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "user",
            "message": "Invalid input"
        }"""
        ev = try_decode_digits_event(payload)
        if ev is None:
            pytest.fail("expected decoded event")
        assert is_digits_job_failed(ev)
        assert ev["type"] == "digits.job.failed.v1"

    def test_decodes_job_failed_event_system(self) -> None:
        from platform_core.digits_metrics_events import (
            is_digits_job_failed,
            try_decode_digits_event,
        )

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "system",
            "message": "Internal error"
        }"""
        ev = try_decode_digits_event(payload)
        if ev is None:
            pytest.fail("expected decoded event")
        assert is_digits_job_failed(ev)
        assert ev["type"] == "digits.job.failed.v1"

    def test_decodes_metrics_event(self) -> None:
        from platform_core.digits_metrics_events import (
            is_digits_config,
            try_decode_digits_event,
        )

        ev = make_config_event(job_id="j1", user_id=1, model_id="m", total_epochs=1, queue="q")
        payload = encode_digits_metrics_event(ev)
        decoded = try_decode_digits_event(payload)
        if decoded is None:
            pytest.fail("expected decoded event")
        assert is_digits_config(decoded)
        assert decoded["type"] == "digits.metrics.config.v1"

    def test_returns_none_for_wrong_domain(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = """{
            "type": "digits.job.started.v1",
            "domain": "other",
            "job_id": "j1",
            "user_id": 1,
            "queue": "q"
        }"""
        assert try_decode_digits_event(payload) is None

    def test_returns_none_for_missing_queue_in_started(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = """{
            "type": "digits.job.started.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1
        }"""
        assert try_decode_digits_event(payload) is None

    def test_returns_none_for_missing_fields_in_completed(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = """{
            "type": "digits.job.completed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1
        }"""
        assert try_decode_digits_event(payload) is None

    def test_returns_none_for_missing_message_in_failed(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "user"
        }"""
        assert try_decode_digits_event(payload) is None

    def test_returns_none_for_invalid_error_kind(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "invalid",
            "message": "msg"
        }"""
        assert try_decode_digits_event(payload) is None

    def test_returns_none_for_unknown_job_event_suffix(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = """{
            "type": "digits.job.unknown.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1
        }"""
        assert try_decode_digits_event(payload) is None

    def test_returns_none_for_missing_job_id_in_metrics(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = '{"type": "digits.metrics.config.v1", "user_id": 1}'
        assert try_decode_digits_event(payload) is None

    def test_returns_none_for_missing_job_id_in_job_event(self) -> None:
        from platform_core.digits_metrics_events import try_decode_digits_event

        payload = """{
            "type": "digits.job.started.v1",
            "domain": "digits",
            "user_id": 1,
            "queue": "q"
        }"""
        assert try_decode_digits_event(payload) is None


class TestCombinedTypeGuards:
    def test_is_digits_job_started(self) -> None:
        from platform_core.digits_metrics_events import (
            DigitsEventV1,
            JobStartedV1,
            is_digits_job_started,
        )

        started: JobStartedV1 = {
            "type": "digits.job.started.v1",
            "domain": "digits",
            "job_id": "j",
            "user_id": 1,
            "queue": "q",
        }
        ev: DigitsEventV1 = started
        assert is_digits_job_started(ev)

    def test_is_digits_job_completed(self) -> None:
        from platform_core.digits_metrics_events import (
            DigitsEventV1,
            JobCompletedV1,
            is_digits_job_completed,
        )

        completed: JobCompletedV1 = {
            "type": "digits.job.completed.v1",
            "domain": "digits",
            "job_id": "j",
            "user_id": 1,
            "result_id": "r",
            "result_bytes": 1,
        }
        ev: DigitsEventV1 = completed
        assert is_digits_job_completed(ev)

    def test_is_digits_job_failed(self) -> None:
        from platform_core.digits_metrics_events import (
            DigitsEventV1,
            JobFailedV1,
            is_digits_job_failed,
        )

        failed: JobFailedV1 = {
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j",
            "user_id": 1,
            "error_kind": "user",
            "message": "m",
        }
        ev: DigitsEventV1 = failed
        assert is_digits_job_failed(ev)

    def test_is_digits_config(self) -> None:
        from platform_core.digits_metrics_events import DigitsEventV1, is_digits_config

        ev: DigitsEventV1 = make_config_event(
            job_id="j", user_id=1, model_id="m", total_epochs=1, queue="q"
        )
        assert is_digits_config(ev)

    def test_is_digits_batch(self) -> None:
        from platform_core.digits_metrics_events import DigitsEventV1, is_digits_batch

        ev: DigitsEventV1 = make_batch_metrics_event(
            job_id="j",
            user_id=1,
            model_id="m",
            epoch=1,
            total_epochs=1,
            batch=1,
            total_batches=1,
            batch_loss=0.1,
            batch_acc=0.9,
            avg_loss=0.1,
            samples_per_sec=1.0,
            main_rss_mb=1,
            workers_rss_mb=1,
            worker_count=1,
            cgroup_usage_mb=1,
            cgroup_limit_mb=1,
            cgroup_pct=1.0,
            anon_mb=1,
            file_mb=1,
        )
        assert is_digits_batch(ev)

    def test_is_digits_epoch(self) -> None:
        from platform_core.digits_metrics_events import DigitsEventV1, is_digits_epoch

        ev: DigitsEventV1 = make_epoch_metrics_event(
            job_id="j",
            user_id=1,
            model_id="m",
            epoch=1,
            total_epochs=1,
            train_loss=0.1,
            val_acc=0.9,
            time_s=1.0,
        )
        assert is_digits_epoch(ev)

    def test_is_digits_best(self) -> None:
        from platform_core.digits_metrics_events import DigitsEventV1, is_digits_best

        ev: DigitsEventV1 = make_best_metrics_event(
            job_id="j", user_id=1, model_id="m", epoch=1, val_acc=0.9
        )
        assert is_digits_best(ev)

    def test_is_digits_artifact(self) -> None:
        from platform_core.digits_metrics_events import DigitsEventV1, is_digits_artifact

        ev: DigitsEventV1 = make_artifact_event(job_id="j", user_id=1, model_id="m", path="/p")
        assert is_digits_artifact(ev)

    def test_is_digits_upload(self) -> None:
        from platform_core.digits_metrics_events import DigitsEventV1, is_digits_upload

        ev: DigitsEventV1 = make_upload_event(
            job_id="j",
            user_id=1,
            model_id="m",
            status=200,
            model_bytes=1,
            manifest_bytes=1,
            file_id="f",
            file_sha256="s",
        )
        assert is_digits_upload(ev)

    def test_is_digits_prune(self) -> None:
        from platform_core.digits_metrics_events import DigitsEventV1, is_digits_prune

        ev: DigitsEventV1 = make_prune_event(job_id="j", user_id=1, model_id="m", deleted_count=1)
        assert is_digits_prune(ev)

    def test_is_digits_completed_metrics(self) -> None:
        from platform_core.digits_metrics_events import (
            DigitsEventV1,
            is_digits_completed_metrics,
        )

        ev: DigitsEventV1 = make_completed_metrics_event(
            job_id="j", user_id=1, model_id="m", val_acc=0.9
        )
        assert is_digits_completed_metrics(ev)

    def test_type_guards_return_false_for_non_matching(self) -> None:
        from platform_core.digits_metrics_events import (
            DigitsEventV1,
            is_digits_artifact,
            is_digits_batch,
            is_digits_best,
            is_digits_completed_metrics,
            is_digits_config,
            is_digits_epoch,
            is_digits_job_completed,
            is_digits_job_failed,
            is_digits_job_started,
            is_digits_prune,
            is_digits_upload,
        )

        ev: DigitsEventV1 = make_config_event(
            job_id="j", user_id=1, model_id="m", total_epochs=1, queue="q"
        )
        assert is_digits_config(ev)
        assert not is_digits_job_started(ev)
        assert not is_digits_job_completed(ev)
        assert not is_digits_job_failed(ev)
        assert not is_digits_batch(ev)
        assert not is_digits_epoch(ev)
        assert not is_digits_best(ev)
        assert not is_digits_artifact(ev)
        assert not is_digits_upload(ev)
        assert not is_digits_prune(ev)
        assert not is_digits_completed_metrics(ev)


class TestDefaultChannel:
    def test_default_channel_value(self) -> None:
        from platform_core.digits_metrics_events import DEFAULT_DIGITS_EVENTS_CHANNEL

        assert DEFAULT_DIGITS_EVENTS_CHANNEL == "digits:events"
