"""Tests for digits metrics events: TypeGuards."""

from __future__ import annotations

from platform_core.digits_metrics_decode import (
    decode_digits_metrics_event,
    is_artifact,
    is_batch,
    is_best,
    is_completed,
    is_config,
    is_epoch,
    is_prune,
    is_upload,
)
from platform_core.digits_metrics_events import (
    DigitsBatchMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    DigitsMetricsEventV1,
    make_artifact_event,
    make_batch_metrics_event,
    make_best_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_epoch_metrics_event,
    make_prune_event,
    make_upload_event,
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
