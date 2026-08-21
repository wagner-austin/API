"""Tests for digits metrics events: DecodeErrors."""

from __future__ import annotations

import pytest

from platform_core.digits_metrics_decode import (
    decode_digits_metrics_event,
)


class TestDecodeErrors:
    def test_non_object_payload_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Expected JSON object"):
            decode_digits_metrics_event("[]")

    def test_non_string_type_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Field 'type' must be a string"):
            decode_digits_metrics_event('{"type": 123}')

    def test_missing_job_id_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'job_id'"):
            decode_digits_metrics_event('{"type": "digits.metrics.config.v1", "user_id": 1}')

    def test_missing_user_id_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'user_id'"):
            decode_digits_metrics_event('{"type": "digits.metrics.config.v1", "job_id": "j1"}')

    def test_unknown_type_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = '{"type": "digits.metrics.unknown.v1", "job_id": "j1", "user_id": 1}'
        with pytest.raises(JSONTypeError, match="Unknown digits metrics event type"):
            decode_digits_metrics_event(payload)

    def test_config_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'model_id'"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.config.v1", "job_id": "j1", "user_id": 1}'
            )

    def test_batch_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = (
            '{"type": "digits.metrics.batch.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
        )
        with pytest.raises(JSONTypeError, match="Missing required field 'epoch'"):
            decode_digits_metrics_event(payload)

    def test_epoch_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = (
            '{"type": "digits.metrics.epoch.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
        )
        with pytest.raises(JSONTypeError, match="Missing required field 'epoch'"):
            decode_digits_metrics_event(payload)

    def test_best_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'epoch'"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.best.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
            )

    def test_artifact_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'model_id'"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.artifact.v1", "job_id": "j1", "user_id": 1}'
            )

    def test_upload_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        payload = (
            '{"type": "digits.metrics.upload.v1", "job_id": "j1", "user_id": 1, "model_id": "m1"}'
        )
        with pytest.raises(JSONTypeError, match="Missing required field 'status'"):
            decode_digits_metrics_event(payload)

    def test_prune_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'model_id'"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.prune.v1", "job_id": "j1", "user_id": 1}'
            )

    def test_completed_missing_required_raises(self) -> None:
        from platform_core.json_utils import JSONTypeError

        with pytest.raises(JSONTypeError, match="Missing required field 'model_id'"):
            decode_digits_metrics_event(
                '{"type": "digits.metrics.completed.v1", "job_id": "j1", "user_id": 1}'
            )
