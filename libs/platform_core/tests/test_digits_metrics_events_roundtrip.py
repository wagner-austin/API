"""Tests for digits metrics events: EncodeDecodeRoundtrip."""

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
    DigitsConfigV1,
    encode_digits_metrics_event,
    make_artifact_event,
    make_batch_metrics_event,
    make_best_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_epoch_metrics_event,
    make_prune_event,
    make_upload_event,
)


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
