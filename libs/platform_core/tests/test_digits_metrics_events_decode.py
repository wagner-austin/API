"""Tests for digits metrics events: DecodeDigitsEvent."""

from __future__ import annotations

import pytest

from platform_core.digits_metrics_events import (
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


class TestDecodeDigitsEvent:
    def test_raises_for_non_dict(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Expected JSON object"):
            decode_digits_event("[]")

    def test_raises_for_non_string_type(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Field 'type' must be a string"):
            decode_digits_event('{"type": 123}')

    def test_raises_for_unknown_type(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Unknown digits event type"):
            decode_digits_event('{"type": "unknown.event.v1", "job_id": "j", "user_id": 1}')

    def test_decodes_job_started_event(self) -> None:
        from platform_core.digits_metrics_events import (
            decode_digits_event,
            is_digits_job_started,
        )

        payload = """{
            "type": "digits.job.started.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "queue": "digits-training"
        }"""
        ev = decode_digits_event(payload)
        assert is_digits_job_started(ev)
        assert ev["type"] == "digits.job.started.v1"

    def test_decodes_job_completed_event(self) -> None:
        from platform_core.digits_metrics_events import (
            decode_digits_event,
            is_digits_job_completed,
        )

        payload = """{
            "type": "digits.job.completed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "result_id": "r1",
            "result_bytes": 1024
        }"""
        ev = decode_digits_event(payload)
        assert is_digits_job_completed(ev)
        assert ev["type"] == "digits.job.completed.v1"

    def test_decodes_job_failed_event_user(self) -> None:
        from platform_core.digits_metrics_events import (
            decode_digits_event,
            is_digits_job_failed,
        )

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "user",
            "message": "Invalid input"
        }"""
        ev = decode_digits_event(payload)
        assert is_digits_job_failed(ev)
        assert ev["type"] == "digits.job.failed.v1"

    def test_decodes_job_failed_event_system(self) -> None:
        from platform_core.digits_metrics_events import (
            decode_digits_event,
            is_digits_job_failed,
        )

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "system",
            "message": "Internal error"
        }"""
        ev = decode_digits_event(payload)
        assert is_digits_job_failed(ev)
        assert ev["type"] == "digits.job.failed.v1"

    def test_decodes_metrics_event(self) -> None:
        from platform_core.digits_metrics_events import (
            decode_digits_event,
            is_digits_config,
        )

        ev = make_config_event(job_id="j1", user_id=1, model_id="m", total_epochs=1, queue="q")
        payload = encode_digits_metrics_event(ev)
        decoded = decode_digits_event(payload)
        assert is_digits_config(decoded)
        assert decoded["type"] == "digits.metrics.config.v1"

    def test_raises_for_wrong_domain(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "digits.job.started.v1",
            "domain": "other",
            "job_id": "j1",
            "user_id": 1,
            "queue": "q"
        }"""
        with pytest.raises(JSONTypeError, match="Domain mismatch"):
            decode_digits_event(payload)

    def test_raises_for_missing_queue_in_started(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "digits.job.started.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'queue'"):
            decode_digits_event(payload)

    def test_raises_for_missing_fields_in_completed(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "digits.job.completed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'result_id'"):
            decode_digits_event(payload)

    def test_raises_for_missing_message_in_failed(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "user"
        }"""
        with pytest.raises(JSONTypeError, match="Missing required field 'message'"):
            decode_digits_event(payload)

    def test_raises_for_invalid_error_kind(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "digits.job.failed.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1,
            "error_kind": "invalid",
            "message": "msg"
        }"""
        with pytest.raises(JSONTypeError, match="Invalid error_kind 'invalid'"):
            decode_digits_event(payload)

    def test_raises_for_unknown_job_event_suffix(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = """{
            "type": "digits.job.unknown.v1",
            "domain": "digits",
            "job_id": "j1",
            "user_id": 1
        }"""
        with pytest.raises(JSONTypeError, match="Unknown digits job event type"):
            decode_digits_event(payload)

    def test_raises_for_missing_job_id(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = '{"type": "digits.metrics.config.v1", "user_id": 1}'
        with pytest.raises(JSONTypeError, match="Missing required field 'job_id'"):
            decode_digits_event(payload)

    def test_raises_for_unknown_metrics_type(self) -> None:
        from platform_core.digits_metrics_events import decode_digits_event
        from platform_core.json_utils import JSONTypeError

        payload = '{"type": "digits.metrics.unknown.v1", "job_id": "j1", "user_id": 1}'
        with pytest.raises(JSONTypeError, match="Unknown digits metrics event type"):
            decode_digits_event(payload)


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
